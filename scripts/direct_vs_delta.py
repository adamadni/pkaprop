"""Direct-vs-delta ablation on ALFABET BDEs.

The design question: is GFN2-xTB's error a smoother function of structure than
the BDE itself? If it is, learning the correction beats learning the value. That
is not obvious -- xTB is off by 15-35 kcal/mol -- so it gets measured rather
than assumed.

Three arms, identical features, identical hyper-parameters, identical split:

  direct       features                -> BDE
  direct+xtb   features + xTB BDE      -> BDE
  delta        features                -> (BDE - xTB BDE), predict xTB + delta

`direct+xtb` is the arm that separates "the xTB number carries information" from
"the delta parameterisation is the right way to use it". Without it, a delta win
is uninterpretable.

Baselines on the same split: bond-type median, and raw uncorrected xTB.

The split is scaffold-based at molecule level. There is deliberately no random
split anywhere, which means these numbers are NOT comparable to ALFABET's
published 0.58 kcal/mol -- that figure is a random molecule-level holdout.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from molprop.cards import Evaluation, Metric, ResultCard
from molprop.chem import BondRef, indexed_mol
from molprop.datasets.alfabet import ALFABET_FIDELITY
from molprop.records import BDERecord, MoleculeRef
from molprop.splits import coverage_report, scaffold_split
from molprop.units import GAS, KCAL_PER_MOL, Quantity, QuantityKind

FP_BITS = 1024
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=FP_BITS)


def featurise(smiles: str, atom_i: int, atom_j: int) -> np.ndarray:
    """Local bond environment plus a whole-molecule fingerprint.

    The bond-centred block is what a BDE actually depends on; the molecular
    block gives the model somewhere to put longer-range effects.
    """
    mol = indexed_mol(smiles)
    local = _GEN.GetCountFingerprintAsNumPy(mol, fromAtoms=[atom_i, atom_j])
    whole = _GEN.GetCountFingerprintAsNumPy(mol)
    ai, aj = mol.GetAtomWithIdx(atom_i), mol.GetAtomWithIdx(atom_j)
    bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
    scalars = np.array(
        [
            mol.GetNumHeavyAtoms(),
            ai.GetAtomicNum(),
            aj.GetAtomicNum(),
            ai.GetDegree(),
            aj.GetDegree(),
            ai.GetTotalNumHs(),
            aj.GetTotalNumHs(),
            int(ai.GetIsAromatic()),
            int(aj.GetIsAromatic()),
            int(ai.IsInRing()),
            int(aj.IsInRing()),
            bond.GetBondTypeAsDouble() if bond else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate([local, whole, scalars]).astype(np.float32)


def build_records(df: pd.DataFrame) -> list[BDERecord]:
    refs: dict[str, MoleculeRef] = {}
    records = []
    for row in df.itertuples():
        ref = refs.get(row.molecule)
        if ref is None:
            ref = MoleculeRef.from_smiles(row.molecule)
            refs[row.molecule] = ref
        records.append(
            BDERecord(
                molecule=ref,
                quantity=Quantity(
                    float(row.bde_dft), KCAL_PER_MOL, QuantityKind.BDE, GAS, 298.15
                ),
                fidelity=ALFABET_FIDELITY,
                source="alfabet-bde-db",
                bond=BondRef(int(row.atom_i), int(row.atom_j)),
            )
        )
    return records


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def r2(pred, true) -> float:
    true = np.asarray(true)
    ss_res = float(np.sum((true - np.asarray(pred)) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def paired_bootstrap(errors_a, errors_b, n_resamples: int = 20000, seed: int = 0):
    """Paired bootstrap CI on the MAE difference between two arms.

    Every arm predicts the same test points, so the comparison must be paired --
    unpaired error bars would be far wider than the actual uncertainty on the
    difference. Note this resamples *test points to estimate uncertainty*; it is
    not a random split, and it never touches the train/test assignment.
    """
    d = np.asarray(errors_a) - np.asarray(errors_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluation(name: str, pred, true, split_id: str, description: str) -> Evaluation:
    return Evaluation(
        name=name,
        split_id=split_id,
        partition="test",
        description=description,
        metrics=(
            Metric("mae", mae(pred, true), KCAL_PER_MOL, n=len(true)),
            Metric("rmse", rmse(pred, true), KCAL_PER_MOL, n=len(true)),
            Metric("r2", r2(pred, true)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", help="csv from xtb_bde_campaign.py")
    parser.add_argument("--card", default=None, help="write the result card here")
    args = parser.parse_args()

    df = pd.read_csv(args.pairs)
    print(f"{len(df):,} paired BDEs over {df.molecule.nunique():,} molecules")

    records = build_records(df)
    split = scaffold_split(records)
    print("\n" + coverage_report(records, split))

    X = np.vstack([featurise(r.molecule, r.atom_i, r.atom_j) for r in df.itertuples()])
    y = df.bde_dft.to_numpy(dtype=float)
    x_xtb = df.bde_xtb.to_numpy(dtype=float)

    train, test = list(split.train), list(split.test)
    print(f"\ntrain {len(train):,} bonds / {len({records[i].group_key for i in train}):,} molecules"
          f"   test {len(test):,} bonds / {len({records[i].group_key for i in test}):,} molecules")

    import lightgbm as lgb

    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        verbose=-1,
        n_jobs=4,
        random_state=0,
    )

    def fit_predict(features_train, target_train, features_test):
        model = lgb.LGBMRegressor(**params)
        model.fit(features_train, target_train)
        return model.predict(features_test)

    results: dict[str, np.ndarray] = {}

    # arm 1: direct
    results["direct"] = fit_predict(X[train], y[train], X[test])

    # arm 2: direct with the xTB value as an extra feature
    Xx = np.hstack([X, x_xtb.reshape(-1, 1).astype(np.float32)])
    results["direct+xtb"] = fit_predict(Xx[train], y[train], Xx[test])

    # arm 3: delta
    delta = y - x_xtb
    results["delta"] = x_xtb[test] + fit_predict(X[train], delta[train], X[test])

    truth = y[test]

    # baselines on the same split
    bond_types = df.bond_type.to_numpy()
    medians = (
        pd.DataFrame({"bt": bond_types[train], "y": y[train]})
        .groupby("bt")
        .y.median()
    )
    fallback = float(np.median(y[train]))
    baseline_lookup = np.array([medians.get(b, fallback) for b in bond_types[test]])

    print(f"\n{'system':<22} {'MAE':>8} {'RMSE':>8} {'R2':>7}")
    print("-" * 48)
    for name, pred in (
        ("bond-type median", baseline_lookup),
        ("raw GFN2-xTB", x_xtb[test]),
        ("direct", results["direct"]),
        ("direct+xtb", results["direct+xtb"]),
        ("delta", results["delta"]),
    ):
        print(f"{name:<22} {mae(pred, truth):>7.2f}  {rmse(pred, truth):>7.2f}  "
              f"{r2(pred, truth):>6.3f}")

    print(f"\nxTB error on test: mean {np.mean(x_xtb[test] - truth):+.2f}, "
          f"std {np.std(x_xtb[test] - truth):.2f} kcal/mol")
    print(f"BDE spread on test: std {np.std(truth):.2f} kcal/mol")

    # Paired comparison against the direct arm. Point estimates this close are
    # meaningless without it.
    abs_err = {k: np.abs(v - truth) for k, v in results.items()}
    print("\npaired MAE difference vs the direct arm (negative = better):")
    print(f"{'arm':<14}{'diff':>10}{'95% CI':>22}   verdict")
    for arm in ("direct+xtb", "delta"):
        mean, lo, hi = paired_bootstrap(abs_err[arm], abs_err["direct"])
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        print(f"{arm:<14}{mean:>+10.3f}   [{lo:>+6.3f}, {hi:>+6.3f}]   {verdict}")

    # Where does a correction help? The xTB offset is near-constant within a
    # bond family but differs sharply between families.
    print("\nper bond type (test MAE):")
    bt_test = bond_types[test]
    print(f"{'bond':<7}{'n':>6}" + "".join(f"{k:>10}" for k in ("xtb", "direct", "delta")))
    for b in sorted(set(bt_test)):
        m = bt_test == b
        if m.sum() < 5:
            continue
        row = f"{b:<7}{int(m.sum()):>6}"
        for pred in (x_xtb[test], results["direct"], results["delta"]):
            row += f"{mae(np.asarray(pred)[m], truth[m]):>10.2f}"
        print(row)

    card = ResultCard(
        model=evaluation(
            "delta LightGBM (xTB -> M06-2X)", results["delta"], truth, split.id,
            "predicts DFT - xTB from bond-environment fingerprints, adds xTB back",
        ),
        baselines=(
            evaluation("direct LightGBM", results["direct"], truth, split.id,
                       "same features and hyper-parameters, predicts the BDE itself"),
            evaluation("direct+xtb LightGBM", results["direct+xtb"], truth, split.id,
                       "direct arm with the xTB BDE appended as one extra feature"),
            evaluation("bond-type median", baseline_lookup, truth, split.id,
                       "median training BDE per element-pair label"),
            evaluation("raw GFN2-xTB", x_xtb[test], truth, split.id,
                       "uncorrected base method, no learning"),
        ),
        dataset_name="alfabet-bde-db",
        dataset_version="v2 (subset)",
        dataset_sha256="subset-see-notes",
        split=split,
        notes=(
            "Scaffold split at molecule level. NOT comparable to ALFABET's published "
            "0.58 kcal/mol, which is a random molecule-level holdout. Subset of "
            f"{df.molecule.nunique()} molecules selected deterministically by sha256 "
            "of canonical SMILES."
        ),
    )
    if args.card:
        from pathlib import Path

        Path(args.card).write_text(card.render_markdown(), encoding="utf-8")
        print(f"\nresult card -> {args.card}")


if __name__ == "__main__":
    main()
