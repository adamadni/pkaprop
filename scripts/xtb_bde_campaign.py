"""Compute GFN2-xTB BDEs for a deterministic subset of ALFABET molecules.

Molecule selection is by sha256 of the canonical SMILES, not by an RNG. It has
to be reproducible and it must not be a *split* -- which partition a molecule
lands in is decided later by `scaffold_split`, never here.

Species-level caching matters: a molecule with 11 bonds needs 1 parent job and
22 fragment jobs, but the fragments repeat heavily across bonds (every X-H
homolysis shares the same [H]), so the same species is otherwise recomputed
many times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from molprop.bde import BDEError, bde_from_enthalpies, homolysis_fragments
from molprop.calc import CalculationError, Task
from molprop.chem import BondRef, canonical_smiles
from molprop.conformers import generate_conformers
from molprop.datasets.alfabet import canonical_bond
from molprop.engines import XTBEngine
from molprop.fidelity import ConformerProtocol, Fidelity, Tier
from molprop.records import MoleculeRef
from molprop.thermo import ensemble_quantity

PROTOCOL = ConformerProtocol(
    generator="rdkit_etkdgv3",
    ensemble_treatment="lowest",
    n_confs_requested=10,
    rmsd_prune_angstrom=0.5,
    rmsd_atoms="all",
    optimizer="mmff94s",
    seed=42,
)
GFN2 = Fidelity(
    tier=Tier.SEMIEMPIRICAL,
    method="GFN2-xTB",
    conformers=PROTOCOL,
    software="xtb",
    software_version="6.7.1",
)


def selection_key(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


class SpeciesCache:
    """Enthalpy per (smiles, charge, multiplicity), computed at most once."""

    def __init__(self, engine: XTBEngine):
        self.engine = engine
        self._cache: dict[tuple[str, int, int], object] = {}
        self.jobs = 0
        self.wall = 0.0
        self.failures: dict[tuple[str, int, int], str] = {}

    def enthalpy(self, smiles: str, charge: int, multiplicity: int):
        key = (smiles, charge, multiplicity)
        if key in self._cache:
            return self._cache[key]
        if key in self.failures:
            return None
        try:
            ensemble = generate_conformers(
                smiles, PROTOCOL, charge=charge, multiplicity=multiplicity
            )
            results = self.engine.run(ensemble, GFN2, Task.OPT_FREQ)
            self.jobs += len(results)
            self.wall += sum(r.usage.wall_seconds for r in results)
            thermo = ensemble_quantity(results, treatment=PROTOCOL.ensemble_treatment)
            self._cache[key] = thermo.enthalpy
            return thermo.enthalpy
        except (CalculationError, BDEError, ValueError, RuntimeError) as exc:
            self.failures[key] = f"{type(exc).__name__}: {exc}"
            return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="ALFABET rdf_data_190531.csv")
    parser.add_argument("--molecules", type=int, default=400)
    parser.add_argument("--out", default="xtb_bdes.csv")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--max-heavy", type=int, default=10)
    parser.add_argument(
        "--bond-types",
        default=None,
        help="comma-separated ALFABET bond_type labels to compute, e.g. 'H-O'. "
             "A targeted campaign only cleaves the bonds it cares about, so a "
             "molecule costs 2-3 jobs instead of 20.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.bond_types:
        wanted = {b.strip() for b in args.bond_types.split(",")}
        unknown = wanted - set(df.bond_type.unique())
        if unknown:
            raise SystemExit(f"unknown bond types {sorted(unknown)}; "
                             f"known: {sorted(df.bond_type.unique())}")
        df = df[df.bond_type.isin(wanted)]
        print(f"targeting bond types {sorted(wanted)}: {len(df):,} candidate bonds "
              f"over {df.molecule.nunique():,} molecules")
    molecules = sorted(df.molecule.unique(), key=selection_key)[: args.molecules]
    chosen = set(molecules)
    rows = df[df.molecule.isin(chosen)]
    print(f"selected {len(molecules):,} molecules -> {len(rows):,} reference BDEs")

    engine = XTBEngine(n_cores=args.cores, imaginary_retries=3)
    cache = SpeciesCache(engine)

    out = []
    start = time.perf_counter()
    for n, (smiles, group) in enumerate(rows.groupby("molecule"), 1):
        canonical = canonical_smiles(smiles)
        parent = cache.enthalpy(canonical, 0, 1)
        if parent is None:
            continue
        for row in group.itertuples():
            try:
                bond = canonical_bond(smiles, int(row.bond_index))
                a, b = homolysis_fragments(canonical, bond)
            except Exception:
                continue
            ha = cache.enthalpy(a.molecule.smiles, a.charge, a.multiplicity)
            hb = cache.enthalpy(b.molecule.smiles, b.charge, b.multiplicity)
            if ha is None or hb is None:
                continue
            try:
                value = bde_from_enthalpies(parent, ha, hb)
            except Exception:
                continue
            out.append(
                {
                    "molecule": canonical,
                    "atom_i": bond.atom_i,
                    "atom_j": bond.atom_j,
                    "bond_type": row.bond_type,
                    "bde_dft": float(row.bde),
                    "bde_xtb": value.value,
                    "rid": int(row.rid),
                }
            )
        if n % 10 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"  {n:>4}/{len(molecules)} molecules  {len(out):>6} BDEs  "
                f"{cache.jobs:>6} jobs  {elapsed:>6.0f}s  "
                f"({elapsed / max(n, 1):.2f} s/molecule)",
                flush=True,
            )
            # Checkpoint: a long campaign must survive being cut short.
            pd.DataFrame(out).to_csv(args.out, index=False)

    frame = pd.DataFrame(out)
    frame.to_csv(args.out, index=False)
    elapsed = time.perf_counter() - start

    print(f"\nwrote {len(frame):,} paired BDEs to {args.out}")
    print(f"coverage: {len(frame):,}/{len(rows):,} reference bonds "
          f"({100 * len(frame) / max(len(rows), 1):.1f}%)")
    print(f"xtb jobs: {cache.jobs:,}   engine wall {cache.wall:.0f}s   total {elapsed:.0f}s")
    print(f"failed species: {len(cache.failures)}")
    for key, why in list(cache.failures.items())[:8]:
        print(f"   {key}: {why[:90]}")
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "n_molecules_requested": args.molecules,
                "n_pairs": len(frame),
                "n_reference_bonds": len(rows),
                "xtb_jobs": cache.jobs,
                "wall_seconds": elapsed,
                "fidelity_id": GFN2.id,
                "protocol": PROTOCOL.describe(),
                "failed_species": len(cache.failures),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
