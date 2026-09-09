"""ALFABET / BDE-db import.

Source: St. John, Guan, Kim, Kim, Paton, "Prediction of organic homolytic bond
dissociation enthalpies at near chemical accuracy with sub-second computational
cost", Nat. Commun. 11, 2328 (2020). Data: figshare 10.6084/m9.figshare.10248932
(``rdf_data_190531.csv``, MIT licence).

Protocol as stated by the source (not from memory):

* M06-2X/def2-TZVP in Gaussian 16.
* Starting geometries: RDKit conformers with MMFF94s, between 100 and 1000 per
  molecule depending on rotatable-bond count; the lowest force-field conformer
  seeded the DFT optimisation.
* Geometries optimised at M06-2X/def2-TZVP with the Hessian evaluated; species
  with imaginary frequencies were discarded.
* Enthalpies at 298 K and 1 atm, with unscaled ZPE and thermal corrections.
* 42,577 CHNO molecules of at most 10 heavy atoms; every single bond not in a
  ring was cleaved.
* ~10% of calculations discarded, outliers removed at >3 IQR, and dissociations
  creating a new stereocentre excluded.

Two facts worth carrying forward:

* The published 0.58 kcal/mol MAE is on a **random molecule-level** holdout
  (40,577 train / 1000 validation / 1000 test parent molecules). It is not a
  scaffold-split number, so it is not directly comparable to a result produced
  under this project's split rules. Compare against it only on a like-for-like
  split, or say plainly that the splits differ.
* Molecules average ~11 bonds, so records must be grouped by molecule when
  splitting. `BDERecord.group_key` already does this.

Atom indexing: ALFABET's ``bond_index`` indexes into
``Chem.AddHs(Chem.MolFromSmiles(molecule))`` using its SMILES exactly as
written. About 1% of those SMILES are not RDKit-canonical, and this package
keys everything off the canonical form, so indices are remapped through
canonical atom ranks. The mapping is verified by reproducing ALFABET's own
fragment SMILES.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rdkit import Chem

from ..chem import BondRef, bond_label, canonical_smiles, indexed_mol
from ..fidelity import ConformerProtocol, Fidelity, Tier
from ..records import BDERecord, MoleculeRef
from ..units import GAS, KCAL_PER_MOL, Quantity, QuantityKind

__all__ = [
    "ALFABET_PROVENANCE",
    "ALFABET_FIDELITY",
    "ImportReport",
    "canonical_bond",
    "load_alfabet",
]


ALFABET_PROVENANCE = {
    "source_url": "https://ndownloader.figshare.com/files/18506018",
    "doi": "10.6084/m9.figshare.10248932",
    "license": "MIT",
    "file": "rdf_data_190531.csv",
    "md5": "2e876544fda84a4b7d44d9f792ad50bd",
    "paper_doi": "10.1038/s41467-020-16201-z",
}

#: The conformer search is recorded as "provided": the upstream protocol is
#: known (RDKit + MMFF94s, 100-1000 conformers, lowest taken) but no seed was
#: published, so it is not reproducible here and must not claim to be.
ALFABET_CONFORMERS = ConformerProtocol(
    generator="provided",
    ensemble_treatment="lowest",
    optimizer="mmff94s",
)

ALFABET_FIDELITY = Fidelity(
    tier=Tier.DFT,
    method="M06-2X",
    basis="def2-TZVP",
    conformers=ALFABET_CONFORMERS,
    software="Gaussian",
    software_version="16",
    notes="BDE-db reference level; enthalpies at 298 K, 1 atm, unscaled ZPE",
)


@functools.lru_cache(maxsize=100_000)
def _rank_map(original_smiles: str) -> tuple[Chem.Mol, tuple[int, ...]]:
    """Map original explicit-H atom indices to canonical explicit-H indices.

    Canonical atom ranks give a deterministic correspondence between the two
    orderings, including for symmetric molecules where a substructure match
    would be ambiguous.
    """
    original = Chem.AddHs(Chem.MolFromSmiles(original_smiles))
    canonical = indexed_mol(canonical_smiles(original_smiles))
    ranks_original = list(Chem.CanonicalRankAtoms(original, breakTies=True))
    ranks_canonical = list(Chem.CanonicalRankAtoms(canonical, breakTies=True))
    position = {rank: index for index, rank in enumerate(ranks_canonical)}
    return original, tuple(position[rank] for rank in ranks_original)


def canonical_bond(original_smiles: str, bond_index: int) -> BondRef:
    """Translate an ALFABET bond index into this package's atom indexing."""
    original, mapping = _rank_map(original_smiles)
    if not 0 <= bond_index < original.GetNumBonds():
        raise ValueError(
            f"bond index {bond_index} out of range for {original_smiles} "
            f"({original.GetNumBonds()} bonds)"
        )
    bond = original.GetBondWithIdx(int(bond_index))
    return BondRef(mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()])


@dataclass
class ImportReport:
    """What survived the import, and why anything did not."""

    total_rows: int = 0
    accepted: int = 0
    rejected_bond_type: int = 0
    rejected_parse: int = 0
    rejected_other: int = 0
    fragment_checked: int = 0
    fragment_mismatch: int = 0
    examples: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if len(self.examples) < 10:
            self.examples.append(message)

    def describe(self) -> str:
        lines = [
            f"rows read           {self.total_rows:>9,}",
            f"accepted            {self.accepted:>9,}",
            f"rejected bond_type  {self.rejected_bond_type:>9,}",
            f"rejected parse      {self.rejected_parse:>9,}",
            f"rejected other      {self.rejected_other:>9,}",
        ]
        if self.fragment_checked:
            lines.append(
                f"fragment spot-check {self.fragment_checked:>9,} rows, "
                f"{self.fragment_mismatch} mismatched"
            )
        lines += [f"  {e}" for e in self.examples]
        return "\n".join(lines)


def _rows(csv_path: Path) -> Iterator[dict]:
    import csv

    with open(csv_path, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def load_alfabet(
    csv_path: str | Path,
    *,
    limit: int | None = None,
    source: str = "alfabet-bde-db",
    verify_fragments_every: int = 0,
) -> tuple[list[BDERecord], ImportReport]:
    """Read BDE-db into `BDERecord`s.

    Every row is gated on the bond type: the element pair implied by the
    remapped atom indices must equal the ``bond_type`` column ALFABET published.
    That is cheap enough to run on all 485k rows and catches any index error.
    `verify_fragments_every` additionally reconstructs the homolysis products
    for every Nth row and compares them with ALFABET's own fragment SMILES.
    """
    from ..bde import homolysis_fragments

    csv_path = Path(csv_path)
    report = ImportReport()
    records: list[BDERecord] = []
    molecules: dict[str, MoleculeRef] = {}

    for index, row in enumerate(_rows(csv_path)):
        if limit is not None and report.total_rows >= limit:
            break
        report.total_rows += 1
        smiles = row["molecule"]
        try:
            bond = canonical_bond(smiles, int(row["bond_index"]))
            canonical = canonical_smiles(smiles)
            reference = molecules.get(canonical)
            if reference is None:
                reference = MoleculeRef.from_smiles(canonical)
                molecules[canonical] = reference
        except Exception as exc:
            report.rejected_parse += 1
            report.note(f"parse {smiles!r}: {type(exc).__name__}: {exc}")
            continue

        observed = bond_label(indexed_mol(canonical), bond.atom_i, bond.atom_j)
        if observed != row["bond_type"]:
            report.rejected_bond_type += 1
            report.note(
                f"bond_type {smiles!r} idx {row['bond_index']}: "
                f"ALFABET says {row['bond_type']}, indices give {observed}"
            )
            continue

        if verify_fragments_every and index % verify_fragments_every == 0:
            report.fragment_checked += 1
            try:
                a, b = homolysis_fragments(canonical, bond)
                produced = {a.molecule.smiles, b.molecule.smiles}
                expected = {
                    canonical_smiles(row["fragment1"]),
                    canonical_smiles(row["fragment2"]),
                }
                if produced != expected:
                    report.fragment_mismatch += 1
                    report.note(
                        f"fragments {smiles!r} idx {row['bond_index']}: "
                        f"expected {sorted(expected)}, produced {sorted(produced)}"
                    )
            except Exception as exc:
                report.fragment_mismatch += 1
                report.note(f"fragments {smiles!r}: {type(exc).__name__}: {exc}")

        try:
            records.append(
                BDERecord(
                    molecule=reference,
                    quantity=Quantity(
                        float(row["bde"]),
                        KCAL_PER_MOL,
                        QuantityKind.BDE,
                        GAS,
                        298.15,
                    ),
                    fidelity=ALFABET_FIDELITY,
                    source=source,
                    source_ref=f"rid:{row['rid']}",
                    bond=bond,
                    metadata={"alfabet_bond_index": int(row["bond_index"])},
                )
            )
            report.accepted += 1
        except Exception as exc:
            report.rejected_other += 1
            report.note(f"record {smiles!r}: {type(exc).__name__}: {exc}")

    return records, report
