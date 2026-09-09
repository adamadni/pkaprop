"""Dataset assembly, batching, training and inference for the pKa model.

Target definition
-----------------
Every ionisable proton in a molecule has a pKa. A carboxylic acid donates one
directly; an amine's proton belongs to its conjugate acid. Both are
deprotonations, so the sign convention is uniform, but they are different
*quantities* and the model keeps them in separate heads.

"Most acidic site" is then the argmin over every site's predicted pKa -- acid
sites scored by the acidic head, basic sites by the basic head. That definition
is what makes the answer well posed for a base-heavy corpus: 60% of the
training compounds are bases, and asking only about acidic protons would
mislabel the majority.

The site label is an assumption, and a stated one
-------------------------------------------------
The pKa *value* is experimental. The *site* comes from Marvin, distributed with
the source data, and is a prediction. Nobody measured which atom titrates for
these 5,994 compounds. Training therefore learns "the experimental pKa, at the
site Marvin believes is responsible". That is the same assumption every
site-level pKa model built on this corpus makes; it is recorded here rather
than hidden. SAMPL6 ships no Marvin annotations so it cannot be scored
site-wise yet; its 10 NMR-determined microstates are the right future check and
the only genuinely measured site assignments available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .features import BOND_FEATURES, MolGraph, XTBDescriptorCache, featurise
from .model import PKaGNN, PKaGNNConfig
from .records import PKaRecord

__all__ = [
    "SiteExample",
    "build_examples",
    "Batch",
    "collate",
    "train_model",
    "predict_sites",
    "most_acidic_site",
]


@dataclass(frozen=True)
class SiteExample:
    """One supervised (molecule, site, pKa) triple."""

    graph: MolGraph
    site_index: int
    pka: float
    is_acid: bool
    group_key: str
    source: str


def build_examples(
    records: Sequence[PKaRecord],
    descriptors: XTBDescriptorCache | None = None,
    *,
    on_error: str = "skip",
) -> tuple[list[SiteExample], list[PKaRecord], dict]:
    """Turn pKa records into site-level training examples.

    Records without a usable site index are dropped rather than assigned to an
    arbitrary atom: a wrong site is worse than a missing one, because it trains
    the model to attribute acidity to the wrong chemistry.

    Returns the surviving records alongside the examples, index-aligned, so the
    split machinery (which is typed for records) and the model (which consumes
    examples) can never drift out of correspondence.
    """
    examples: list[SiteExample] = []
    kept: list[PKaRecord] = []
    report = {"records": len(records), "kept": 0, "no_site": 0, "featurise_failed": 0,
              "site_out_of_range": 0, "no_type": 0}
    for record in records:
        site = record.metadata.get("marvin_predicted_atom_canonical")
        kind = record.metadata.get("marvin_predicted_type")
        if site is None:
            report["no_site"] += 1
            continue
        if kind not in ("acidic", "basic"):
            report["no_type"] += 1
            continue
        try:
            descriptor = descriptors.get(record.molecule.smiles) if descriptors else None
            graph = featurise(record.molecule.smiles, descriptor)
        except Exception:
            report["featurise_failed"] += 1
            continue
        if not 0 <= int(site) < graph.n_atoms:
            report["site_out_of_range"] += 1
            continue
        examples.append(
            SiteExample(
                graph=graph,
                site_index=int(site),
                pka=float(record.quantity.value),
                is_acid=(kind == "acidic"),
                group_key=record.group_key,
                source=record.source,
            )
        )
        kept.append(record)
        report["kept"] += 1
    return examples, kept, report


@dataclass
class Batch:
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    batch_index: torch.Tensor
    n_graphs: int
    site_rows: torch.Tensor   # index into the flattened atom tensor
    targets: torch.Tensor
    is_acid: torch.Tensor


def collate(examples: Sequence[SiteExample]) -> Batch:
    """Pack graphs into one disjoint graph, the standard GNN batching trick."""
    xs, eis, eas, bidx, rows = [], [], [], [], []
    offset = 0
    for graph_id, ex in enumerate(examples):
        g = ex.graph
        xs.append(g.atom_features)
        if g.edge_index.shape[1]:
            eis.append(g.edge_index + offset)
            eas.append(g.edge_features)
        bidx.append(np.full(g.n_atoms, graph_id, dtype=np.int64))
        rows.append(offset + ex.site_index)
        offset += g.n_atoms
    edge_index = (
        np.concatenate(eis, axis=1) if eis else np.zeros((2, 0), dtype=np.int64)
    )
    edge_features = (
        np.concatenate(eas, axis=0) if eas else np.zeros((0, BOND_FEATURES), np.float32)
    )
    return Batch(
        atom_features=torch.from_numpy(np.concatenate(xs, axis=0)),
        edge_index=torch.from_numpy(edge_index),
        edge_features=torch.from_numpy(edge_features),
        batch_index=torch.from_numpy(np.concatenate(bidx)),
        n_graphs=len(examples),
        site_rows=torch.tensor(rows, dtype=torch.long),
        targets=torch.tensor([e.pka for e in examples], dtype=torch.float32),
        is_acid=torch.tensor([e.is_acid for e in examples], dtype=torch.bool),
    )


def _masked_loss(
    acid_pred: torch.Tensor,
    base_pred: torch.Tensor,
    batch: Batch,
    criterion: nn.Module,
) -> torch.Tensor:
    """Supervise only the head that matches each record's type.

    The other head gets no gradient from this example, which is the point:
    an amine has no measured acidic pKa, so inventing a target for it would
    teach the acidic head chemistry that was never observed.
    """
    acid_at_site = acid_pred[batch.site_rows]
    base_at_site = base_pred[batch.site_rows]
    chosen = torch.where(batch.is_acid, acid_at_site, base_at_site)
    return criterion(chosen, batch.targets)


def train_model(
    train_examples: Sequence[SiteExample],
    val_examples: Sequence[SiteExample],
    *,
    hidden: int = 128,
    depth: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    max_epochs: int = 120,
    patience: int = 15,
    seed: int = 0,
    log_every: int = 5,
    verbose: bool = True,
) -> tuple[PKaGNN, dict]:
    """Fit with early stopping on validation MAE.

    Huber loss rather than MSE: the corpus contains a small number of extreme
    or mis-transcribed pKa values, and squared error would let them dominate
    the gradient. Huber keeps them informative without letting them steer.
    """
    torch.manual_seed(seed)
    config = PKaGNNConfig(
        atom_features=int(train_examples[0].graph.atom_features.shape[1]),
        bond_features=BOND_FEATURES,
        hidden=hidden,
        depth=depth,
        dropout=dropout,
    )
    model = PKaGNN(config)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=1.0)

    order = np.arange(len(train_examples))
    generator = np.random.default_rng(seed)
    history: list[dict] = []
    best = {"mae": float("inf"), "epoch": -1, "state": None}
    started = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        generator.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch_size):
            chunk = [train_examples[i] for i in order[start : start + batch_size]]
            batch = collate(chunk)
            optimiser.zero_grad()
            acid, base = model(
                batch.atom_features, batch.edge_index, batch.edge_features,
                batch.batch_index, batch.n_graphs,
            )
            loss = _masked_loss(acid, base, batch, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total += float(loss) * len(chunk)

        val_pred, val_true = predict_sites(model, val_examples, batch_size=batch_size)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        scheduler.step(val_mae)
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_mae": val_mae})
        if val_mae < best["mae"] - 1e-4:
            best = {
                "mae": val_mae,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
        if verbose and (epoch % log_every == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:>3}  train {total / len(order):.4f}  "
                  f"val MAE {val_mae:.3f}  best {best['mae']:.3f} @ {best['epoch']}",
                  flush=True)
        if epoch - best["epoch"] >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no improvement for {patience})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {
        "history": history,
        "best_val_mae": best["mae"],
        "best_epoch": best["epoch"],
        "seconds": time.perf_counter() - started,
        "config": config.__dict__,
    }


@torch.no_grad()
def predict_sites(
    model: PKaGNN, examples: Sequence[SiteExample], *, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true pKa at each example's labelled site."""
    model.eval()
    preds, trues = [], []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate(chunk)
        acid, base = model(
            batch.atom_features, batch.edge_index, batch.edge_features,
            batch.batch_index, batch.n_graphs,
        )
        chosen = torch.where(
            batch.is_acid, acid[batch.site_rows], base[batch.site_rows]
        )
        preds.append(chosen.numpy())
        trues.append(batch.targets.numpy())
    if not preds:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def most_acidic_site(
    model: PKaGNN, smiles: str, descriptors: XTBDescriptorCache | None = None
) -> dict:
    """Score every ionisable site and return the most acidic one.

    Acid sites are scored by the acidic head, basic sites by the basic head,
    and the winner is the global minimum. A molecule with no detected site
    returns `site=None` rather than a fabricated answer.
    """
    model.eval()
    descriptor = descriptors.get(smiles) if descriptors else None
    graph = featurise(smiles, descriptor)
    batch = collate(
        [SiteExample(graph, 0, 0.0, True, "", "")]
    )
    acid, base = model(
        batch.atom_features, batch.edge_index, batch.edge_features,
        batch.batch_index, batch.n_graphs,
    )
    sites = []
    for idx in graph.acid_sites:
        sites.append({"atom_index": int(idx), "kind": "acid", "pka": float(acid[idx])})
    for idx in graph.base_sites:
        sites.append({"atom_index": int(idx), "kind": "base", "pka": float(base[idx])})
    sites.sort(key=lambda s: s["pka"])
    return {
        "smiles": graph.smiles,
        "sites": sites,
        "most_acidic": sites[0] if sites else None,
    }
