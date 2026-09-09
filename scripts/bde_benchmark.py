"""BDE benchmark on the full ALFABET dataset.

Step 3's deliverable: a real BDE result on all 484,907 reference values, under
this project's split rules, with baselines on the same split.

Scope note. This trains the `direct` arm only. The ablation showed `direct+xtb`
is significantly better (2.68 vs 4.54 kcal/mol on a 507-molecule subset), but
extending that to the whole dataset needs GFN2 thermochemistry for all 42,577
molecules -- about 325,000 xtb jobs, roughly 60 hours locally. That is a
scheduled campaign, not something to slip into a benchmark script, so this file
reports what can be computed without new QM and says plainly what is missing.

Nothing here is comparable to ALFABET's published 0.58 kcal/mol: that is a
random molecule-level holdout, and this project uses scaffold splits only.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from molprop.cards import Evaluation, Metric, ResultCard
from molprop.chem import indexed_mol
from molprop.datasets.alfabet import ALFABET_FIDELITY, load_alfabet
from molprop.splits import coverage_report, scaffold_split
from molprop.units import KCAL_PER_MOL

FP_BITS = 1024
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=FP_BITS)
N_SCALARS = 12


def featurise_block(records, start: int, stop: int) -> sparse.csr_matrix:
    """Featurise a slice of records into a sparse row block.

    Fingerprints are sparse (tens of nonzeros in 1024 bins). Densifying half a
    million rows would need ~4 GB; CSR needs a few hundred MB.
    """
    rows, cols, vals = [], [], []
    for offset, record in enumerate(records[start:stop]):
        mol = indexed_mol(record.molecule.smiles)
        i, j = record.bond.as_tuple()
        local = _GEN.GetCountFingerprintAsNumPy(mol, fromAtoms=[i, j])
        whole = _GEN.GetCountFingerprintAsNumPy(mol)
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        bond = mol.GetBondBetweenAtoms(i, j)
        scalars = np.array(
            [
                mol.GetNumHeavyAtoms(),
                ai.GetAtomicNum(), aj.GetAtomicNum(),
                ai.GetDegree(), aj.GetDegree(),
                ai.GetTotalNumHs(), aj.GetTotalNumHs(),
                int(ai.GetIsAromatic()), int(aj.GetIsAromatic()),
                int(ai.IsInRing()), int(aj.IsInRing()),
                bond.GetBondTypeAsDouble() if bond else 0.0,
            ],
            dtype=np.float32,
        )
        dense = np.concatenate([local, whole, scalars])
        nz = np.nonzero(dense)[0]
        rows.extend([offset] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(dense[nz].tolist())
    shape = (stop - start, 2 * FP_BITS + N_SCALARS)
    return sparse.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)), shape=shape
    )


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def r2(pred, true) -> float:
    true = np.asarray(true)
    ss_res = float(np.sum((true - np.asarray(pred)) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def evaluation(name, pred, true, split_id, description) -> Evaluation:
    return Evaluation(
        name=name, split_id=split_id, partition="test", description=description,
        metrics=(
            Metric("mae", mae(pred, true), KCAL_PER_MOL, n=len(true)),
            Metric("rmse", rmse(pred, true), KCAL_PER_MOL, n=len(true)),
            Metric("r2", r2(pred, true)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--card", default="bde_benchmark_card.md")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    records, report = load_alfabet(args.csv, limit=args.limit)
    print(f"loaded {report.accepted:,} records in {time.perf_counter() - t0:.0f}s")

    t1 = time.perf_counter()
    split = scaffold_split(records)
    print(f"\n{coverage_report(records, split)}")
    print(f"split in {time.perf_counter() - t1:.0f}s")

    t2 = time.perf_counter()
    blocks = []
    step = 25_000
    for start in range(0, len(records), step):
        blocks.append(featurise_block(records, start, min(start + step, len(records))))
        print(f"  featurised {min(start + step, len(records)):,}/{len(records):,}", flush=True)
    X = sparse.vstack(blocks, format="csr")
    del blocks
    print(f"features {X.shape} nnz={X.nnz:,} ({X.data.nbytes / 1e6:.0f} MB) "
          f"in {time.perf_counter() - t2:.0f}s")

    y = np.array([r.quantity.value for r in records], dtype=float)
    bond_types = np.array([r.bond_type for r in records])
    train, test = list(split.train), list(split.test)

    import lightgbm as lgb

    params = dict(
        n_estimators=1500, learning_rate=0.05, num_leaves=255,
        min_child_samples=20, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=6, random_state=0,
    )
    t3 = time.perf_counter()
    model = lgb.LGBMRegressor(**params)
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    val = list(split.val)
    pred_val = model.predict(X[val])
    print(f"\ntrained in {time.perf_counter() - t3:.0f}s")

    truth = y[test]
    medians = pd.DataFrame({"bt": bond_types[train], "y": y[train]}).groupby("bt").y.median()
    fallback = float(np.median(y[train]))
    lookup = np.array([medians.get(b, fallback) for b in bond_types[test]])
    global_mean = np.full(len(test), float(np.mean(y[train])))

    print(f"\n{'system':<26}{'MAE':>8}{'RMSE':>9}{'R2':>8}")
    print("-" * 51)
    for name, p in (
        ("global mean", global_mean),
        ("bond-type median", lookup),
        ("direct LightGBM", pred),
    ):
        print(f"{name:<26}{mae(p, truth):>8.2f}{rmse(p, truth):>9.2f}{r2(p, truth):>8.3f}")

    print(f"\nper bond type (test MAE):")
    print(f"{'bond':<7}{'n':>9}{'lookup':>10}{'model':>10}")
    for b in sorted(set(bond_types[test])):
        m = bond_types[test] == b
        if m.sum() < 20:
            continue
        print(f"{b:<7}{int(m.sum()):>9,}{mae(lookup[m], truth[m]):>10.2f}"
              f"{mae(pred[m], truth[m]):>10.2f}")

    # A strong aggregate MAE can hide a model that is actively worse than a
    # trivial lookup on a rare stratum. Decide which strata to trust using the
    # VALIDATION partition, never the test partition.
    val_truth = y[val]
    val_lookup = np.array([medians.get(b, fallback) for b in bond_types[val]])
    trusted, distrusted = set(), {}
    for b in sorted(set(bond_types[val])):
        m = bond_types[val] == b
        if m.sum() < 10:
            continue
        model_mae, look_mae = mae(pred_val[m], val_truth[m]), mae(val_lookup[m], val_truth[m])
        if model_mae <= look_mae:
            trusted.add(b)
        else:
            distrusted[b] = (model_mae, look_mae, int(m.sum()))

    if distrusted:
        print("\nbond types where the model LOSES to the lookup on validation:")
        for b, (mm, lm, n) in distrusted.items():
            n_train = int((bond_types[train] == b).sum())
            print(f"   {b:<6} n_val={n:<5} model {mm:>6.2f} vs lookup {lm:>6.2f}   "
                  f"(only {n_train:,} training bonds)")

    hybrid = np.where(
        np.isin(bond_types[test], sorted(trusted)), pred, lookup
    )
    print(f"\n{'hybrid (val-selected fallback)':<26}{mae(hybrid, truth):>8.2f}"
          f"{rmse(hybrid, truth):>9.2f}{r2(hybrid, truth):>8.3f}")

    np.savez_compressed(
        args.card.replace(".md", "_predictions.npz"),
        test_index=np.array(test), truth=truth, model=pred,
        lookup=lookup, hybrid=hybrid, bond_type=bond_types[test],
    )

    card = ResultCard(
        model=evaluation("hybrid (model + val-selected lookup)", hybrid, truth, split.id,
                         "LightGBM where it beats the lookup on validation, "
                         "bond-type median on the strata where it does not"),
        baselines=(
            evaluation("direct LightGBM (ECFP6 + bond env)", pred, truth, split.id,
                       "gradient-boosted trees on Morgan count fingerprints of the "
                       "bond environment plus a whole-molecule fingerprint"),
            evaluation("bond-type median", lookup, truth, split.id,
                       "median training BDE per element-pair label"),
            evaluation("global mean", global_mean, truth, split.id,
                       "mean training BDE, ignores structure entirely"),
        ),
        dataset_name="alfabet-bde-db",
        dataset_version="v2",
        dataset_sha256="6b12b3414ce7097b31f633a006dd407aaf488c826c83bdf44d0db02674d4d029",
        split=split,
        fidelity=ALFABET_FIDELITY,
        notes=(
            "Scaffold split at molecule level; no random split anywhere, so this is "
            "NOT comparable to ALFABET's published 0.58 kcal/mol (a random "
            "molecule-level holdout). Direct arm only: the ablation showed direct+xtb "
            "is significantly better, but that needs GFN2 thermochemistry for all "
            "42,577 molecules (~325k xtb jobs) which has not been run."
        ),
    )
    from pathlib import Path

    Path(args.card).write_text(card.render_markdown(), encoding="utf-8")
    print(f"\ntotal {time.perf_counter() - t0:.0f}s   card -> {args.card}")


if __name__ == "__main__":
    main()
