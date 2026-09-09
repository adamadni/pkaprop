"""The pKa network.

Architecture and why
--------------------
**Per-atom output, not per-molecule.** The task is "which site is most acidic
and what is its pKa", so the prediction has to be attached to an atom. A
whole-molecule regressor could report a number but could never point at a site,
and pooling to a molecule vector would discard exactly the locality the answer
depends on.

**Message passing with edge conditioning.** Acidity is set by how well the
conjugate base delocalises charge, and that propagates through bonds --
inductive effects through sigma bonds, resonance through pi systems. A
convolution whose messages depend on the *bond* as well as the neighbouring
atom can distinguish those; one that only sees neighbour identities cannot tell
a conjugated path from a saturated one. Messages are therefore built from
``[h_source, h_target, edge]``.

**Depth 4.** A Hammett substituent effect reaches roughly 4-6 bonds through a
conjugated system before it decays into noise. Four rounds of message passing
give each atom a receptive field of 4 bonds, which covers para substitution on
a ring bearing the ionisable group -- the canonical long-range case. Going
deeper mostly buys over-smoothing, where every atom in a fused ring converges
to the same embedding.

**Residual updates and layer norm.** With 4 rounds on graphs of ~20 atoms,
plain stacking washes out the atom's own identity. A residual update keeps the
site's local character (an oxygen stays recognisably an oxygen) while the
message stream carries the environment.

**Two heads, one for acids and one for bases.** The same nitrogen has an
"acidic pKa" near 35 and a "basic pKa" near 10; these are different physical
quantities that happen to share an atom. One head would have to fit a strongly
bimodal target conditioned on an unstated variable. Two heads make the
condition explicit and let each specialise. Only the head matching a record's
type is supervised, so an unlabelled head contributes no gradient.

**Sum-pooled global context.** A small molecule-level vector is concatenated
into the per-atom head. Solubility-like whole-molecule effects (overall charge,
size) shift pKa systematically, and a purely local readout cannot see them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["PKaGNNConfig", "PKaGNN"]


@dataclass(frozen=True)
class PKaGNNConfig:
    atom_features: int
    bond_features: int
    hidden: int = 128
    depth: int = 4
    dropout: float = 0.1
    global_features: int = 64


class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned message passing with a residual update."""

    def __init__(self, hidden: int, bond_features: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + bond_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        if edge_index.numel() == 0:
            return self.norm(h + self.update(torch.cat([h, torch.zeros_like(h)], dim=-1)))
        src, dst = edge_index[0], edge_index[1]
        messages = self.message(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        aggregated = torch.zeros_like(h).index_add_(0, dst, messages)
        return self.norm(h + self.update(torch.cat([h, aggregated], dim=-1)))


class PKaGNN(nn.Module):
    """Predicts an acidic and a basic pKa for every atom."""

    def __init__(self, config: PKaGNNConfig):
        super().__init__()
        self.config = config
        h = config.hidden
        self.embed = nn.Sequential(nn.Linear(config.atom_features, h), nn.SiLU(), nn.Linear(h, h))
        self.layers = nn.ModuleList(
            MessagePassingLayer(h, config.bond_features, config.dropout)
            for _ in range(config.depth)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(h, config.global_features), nn.SiLU()
        )
        head_in = h + config.global_features
        self.acid_head = self._head(head_in, h, config.dropout)
        self.base_head = self._head(head_in, h, config.dropout)

    @staticmethod
    def _head(in_dim: int, hidden: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (acidic pKa, basic pKa) per atom, each shape (n_atoms,)."""
        h = self.embed(atom_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        pooled = pooled.index_add_(0, batch, h)
        context = self.global_proj(pooled)[batch]

        joined = torch.cat([h, context], dim=-1)
        return self.acid_head(joined).squeeze(-1), self.base_head(joined).squeeze(-1)
