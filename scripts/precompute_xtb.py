"""Warm the GFN2-xTB descriptor cache.

Separated from training so the (slow, deterministic) physics step can run once
and be reused by every subsequent fit and ablation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from molprop.datasets.pka import load_czodrowskilab, load_sampl6
from molprop.features import XTBDescriptorCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--cache", default="data/xtb_cache")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.data)
    cache = XTBDescriptorCache(args.cache)
    if not cache.available():
        raise SystemExit("xtb not found on PATH")

    smiles: list[str] = []
    for fname in ("czodrowskilab_train.sdf", "novartis_test.sdf", "avlilumove_test.sdf"):
        if (root / fname).exists():
            records, _ = load_czodrowskilab(root / fname, source=fname)
            smiles += [r.molecule.smiles for r in records]
    if (root / "sampl6_stacked.csv").exists():
        records, _ = load_sampl6(root / "sampl6_stacked.csv")
        smiles += [r.molecule.smiles for r in records]

    unique = sorted(set(smiles))
    if args.num_shards > 1:
        unique = unique[args.shard :: args.num_shards]
    print(f"{len(unique):,} unique molecules to descriptor-ise", flush=True)
    started = time.perf_counter()
    for i, s in enumerate(unique, 1):
        cache.get(s)
        if i % 250 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {i:>6}/{len(unique):,}  {elapsed:>6.0f}s  "
                  f"({elapsed / i:.2f} s/mol)  failures={cache.failures}", flush=True)
    print(f"done in {time.perf_counter() - started:.0f}s; "
          f"{cache.misses} computed, {cache.failures} failed")


if __name__ == "__main__":
    main()
