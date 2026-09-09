"""Train and evaluate the pKa model.

Runs the whole pipeline: load, curate, split, featurise, fit, and report
against baselines on the same split. Nothing is reported without a baseline --
`ResultCard` refuses to be constructed otherwise.

    python scripts/train_pka.py --data data/raw --xtb        # with physics
    python scripts/train_pka.py --data data/raw --no-xtb     # ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from molprop.cards import Evaluation, Metric, ResultCard
from molprop.curation import audit_pka_records, summarise
from molprop.datasets.pka import load_czodrowskilab, load_sampl6
from molprop.features import XTBDescriptorCache
from molprop.splits import coverage_report, scaffold_split
from molprop.training import build_examples, predict_sites, train_model
from molprop.units import LOG10_UNIT


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def r2(pred, true) -> float:
    true = np.asarray(true)
    ss_res = float(np.sum((true - np.asarray(pred)) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def within(pred, true, tol: float) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true)) <= tol))


def evaluation(name, pred, true, split_id, description, partition="test") -> Evaluation:
    return Evaluation(
        name=name, split_id=split_id, partition=partition, description=description,
        metrics=(
            Metric("mae", mae(pred, true), LOG10_UNIT, n=len(true)),
            Metric("rmse", rmse(pred, true), LOG10_UNIT, n=len(true)),
            Metric("r2", r2(pred, true)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--xtb", dest="xtb", action="store_true", default=True)
    parser.add_argument("--no-xtb", dest="xtb", action="store_false")
    parser.add_argument("--cache", default="data/xtb_cache")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "xtb" if args.xtb else "noxtb"

    # ---- load -------------------------------------------------------------
    train_records, report = load_czodrowskilab(root / "czodrowskilab_train.sdf",
                                               source="czodrowskilab-train")
    print("training corpus\n" + report.describe())
    print("\ncuration audit at import:")
    print(summarise(audit_pka_records(train_records))[:600])
    if args.limit:
        train_records = train_records[: args.limit]

    holdout = {}
    for name, fname in (("novartis", "novartis_test.sdf"),
                        ("avlilumove", "avlilumove_test.sdf")):
        if (root / fname).exists():
            holdout[name], _ = load_czodrowskilab(root / fname, source=name)
    if (root / "sampl6_stacked.csv").exists():
        holdout["sampl6"], _ = load_sampl6(root / "sampl6_stacked.csv")

    # ---- featurise --------------------------------------------------------
    cache = XTBDescriptorCache(args.cache) if args.xtb else None
    if cache and not cache.available():
        raise SystemExit("xtb not found on PATH; rerun with --no-xtb")
    print(f"\nfeaturising ({'with' if args.xtb else 'without'} xTB descriptors)...")
    examples, kept_records, build_report = build_examples(train_records, cache)
    print(f"  {build_report}")
    if cache:
        print(f"  xtb cache: {cache.hits} hits, {cache.misses} misses, "
              f"{cache.failures} failures")

    # ---- split ------------------------------------------------------------
    # Split on the RECORDS -- the split module is typed for them -- and index
    # the examples with the same integers. build_examples returns the surviving
    # records so the two lists cannot drift out of correspondence.
    split = scaffold_split(kept_records)
    print("\n" + coverage_report(kept_records, split))
    tr = [examples[i] for i in split.train]
    va = [examples[i] for i in split.val]
    te = [examples[i] for i in split.test]

    # ---- fit --------------------------------------------------------------
    print(f"\ntraining on {len(tr):,} sites, validating on {len(va):,}")
    model, info = train_model(
        tr, va, hidden=args.hidden, depth=args.depth,
        max_epochs=args.epochs, seed=args.seed,
    )
    print(f"  best val MAE {info['best_val_mae']:.3f} at epoch {info['best_epoch']} "
          f"({info['seconds']:.0f}s)")

    # ---- evaluate ---------------------------------------------------------
    pred, true = predict_sites(model, te)

    # Baselines on the SAME split. The site-class median is the one that
    # matters: it is what you get from knowing only the functional group.
    from molprop.enumeration import find_protonation_sites

    def site_class(ex):
        for s in find_protonation_sites(ex.graph.smiles):
            if s.atom_idx == ex.site_index:
                return s.label
        return "unknown"

    classes_tr = [site_class(e) for e in tr]
    by_class: dict[str, list[float]] = {}
    for label, ex in zip(classes_tr, tr):
        by_class.setdefault(label, []).append(ex.pka)
    medians = {k: float(np.median(v)) for k, v in by_class.items()}
    fallback = float(np.median([e.pka for e in tr]))
    class_pred = np.array([medians.get(site_class(e), fallback) for e in te])
    global_pred = np.full(len(te), fallback)
    acid_median = float(np.median([e.pka for e in tr if e.is_acid]))
    base_median = float(np.median([e.pka for e in tr if not e.is_acid]))
    type_pred = np.array([acid_median if e.is_acid else base_median for e in te])

    print(f"\n{'system':<28}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'<=1':>8}{'<=2':>8}")
    print("-" * 68)
    rows = [
        ("global median", global_pred),
        ("acid/base median", type_pred),
        ("site-class median", class_pred),
        (f"pKa GNN ({tag})", pred),
    ]
    for name, p in rows:
        print(f"{name:<28}{mae(p, true):>8.2f}{rmse(p, true):>8.2f}{r2(p, true):>8.3f}"
              f"{within(p, true, 1.0):>8.0%}{within(p, true, 2.0):>8.0%}")

    card = ResultCard(
        model=evaluation(f"pKa GNN ({tag})", pred, true, split.id,
                         "edge-conditioned message passing, depth 4, per-atom "
                         "acidic/basic heads"),
        baselines=(
            evaluation("site-class median", class_pred, true, split.id,
                       "median training pKa for the site's functional-group class"),
            evaluation("acid/base median", type_pred, true, split.id,
                       "median training pKa for acids and bases separately"),
            evaluation("global median", global_pred, true, split.id,
                       "median training pKa, ignores structure entirely"),
        ),
        dataset_name="czodrowskilab-train",
        dataset_version="v1",
        dataset_sha256="see data/raw provenance",
        split=split,
        notes=("Scaffold split at molecule level; no random split anywhere. The pKa "
               "value is experimental, the SITE is a Marvin prediction distributed "
               "with the source -- see molprop.training for what that assumes."),
    )
    (out / f"result_card_{tag}.md").write_text(card.render_markdown(), encoding="utf-8")

    # ---- held-out sets ----------------------------------------------------
    holdout_scores = {}
    for name, records in holdout.items():
        ex, _, _ = build_examples(records, cache)
        if not ex:
            continue
        p, t = predict_sites(model, ex)
        holdout_scores[name] = {"n": len(t), "mae": mae(p, t), "rmse": rmse(p, t),
                                "within_1": within(p, t, 1.0)}
        print(f"held-out {name:<12} n={len(t):<5} MAE {mae(p, t):.2f}  "
              f"RMSE {rmse(p, t):.2f}  within 1 unit {within(p, t, 1.0):.0%}")

    import torch

    torch.save({"state_dict": model.state_dict(), "config": info["config"],
                "used_xtb": args.xtb}, out / f"pka_gnn_{tag}.pt")
    (out / f"metrics_{tag}.json").write_text(json.dumps({
        "split_id": split.id,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test": {name: {"mae": mae(p, true), "rmse": rmse(p, true),
                        "r2": r2(p, true), "within_1": within(p, true, 1.0)}
                 for name, p in rows},
        "holdout": holdout_scores,
        "training": {k: v for k, v in info.items() if k != "history"},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/result_card_{tag}.md, metrics_{tag}.json, pka_gnn_{tag}.pt")


if __name__ == "__main__":
    main()
