"""pKa dataset import.

Two sources, both fetched with their provenance rather than described from
memory:

**SAMPL6** (github.com/samplchallenges/SAMPL6, MIT). 24 molecules, 31
macroscopic pKa values with SEM. Conditions from the challenge instructions:
three replicates at 25 C in ionic-strength-adjusted water with 0.15 M KCl on a
Sirius T3. The source is explicit that "this method produces estimates of
macroscopic pKas". 15 of the 31 values come from a methanol-cosolvent protocol
(30/40/50% methanol) extrapolated to pure water by Yasuda-Shedlovsky; the
source measured both ways where it could and found differences up to 0.4 pKa
units, mean 0.05, with no obvious bias. That is a different measurement
condition and is recorded as such.

**Baltruschat & Czodrowski** (github.com/czodrowskilab/Machine-learning-meets-pKa,
MIT). ChEMBL25, DataWarrior and Novartis values combined, plus two held-out
test sets. Temperature and ionic strength are not distributed with the data.

Three curation decisions this module makes, each of which would be a silent
error if made the other way:

1. **`marvin_atom` is not a site.** The SDFs carry a `marvin_atom` index and a
   `marvin_pKa_type`, but those are *predictions* from Marvin, not measured
   site assignments, while the `pKa` value is macroscopic experiment. Recording
   them as `site_atom_idx` would manufacture microscopic data that nobody
   measured. They go in metadata.

2. **Molecule identity is the neutral parent.** The Novartis file stores
   cations (`[NH+]`) while others store neutral forms, so the same compound
   would otherwise get two identities, break deduplication, and let one
   compound straddle a train/test boundary.

3. **Every record is macroscopic, step 1.** These sets are curated to one value
   per compound. Where a compound is polyprotic, that is an *assumption* about
   which equilibrium was reported, and `curation.audit_pka_records` raises
   `incomplete-titration` so the size of the problem is measured rather than
   assumed away.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from ..chem import MoleculeError, canonical_smiles
from ..fidelity import Fidelity, Tier
from ..records import MoleculeRef, PKaRecord
from ..units import LOG10_UNIT, Quantity, QuantityKind, water

__all__ = [
    "SAMPL6_PROVENANCE",
    "CZODROWSKILAB_PROVENANCE",
    "EXPERIMENT",
    "neutral_parent",
    "PKaImportReport",
    "load_sampl6",
    "load_czodrowskilab",
    "map_atom_to_canonical",
]

SAMPL6_PROVENANCE = {
    "source_url": "https://github.com/samplchallenges/SAMPL6",
    "license": "MIT",
    "paper_doi": "10.1007/s10822-018-0168-0",
    "conditions": "25 C, 0.15 M KCl, Sirius T3, macroscopic pKa",
}

CZODROWSKILAB_PROVENANCE = {
    "source_url": "https://github.com/czodrowskilab/Machine-learning-meets-pKa",
    "license": "MIT",
    "conditions": "not distributed with the data",
}

EXPERIMENT = Fidelity(tier=Tier.EXPERIMENT, method="experiment")

_UNCHARGER = rdMolStandardize.Uncharger()


def map_atom_to_canonical(mol: Chem.Mol, atom_idx: int) -> int | None:
    """Translate an atom index from a source mol into this package's ordering.

    `marvin_atom` indexes the SDF atom block, but records key on the canonical
    SMILES of the NEUTRAL PARENT -- a different molecule with a different atom
    order. Reusing the integer would silently point at the wrong atom, which is
    the same class of bug that the ALFABET bond indices had.

    Neutralisation is done in place so atom order is preserved, then canonical
    ranks give a deterministic correspondence that is robust to symmetry.
    """
    try:
        neutral = rdMolStandardize.ChargeParent(Chem.Mol(mol))
    except Exception:
        return None
    if neutral is None or neutral.GetNumAtoms() != mol.GetNumAtoms():
        # ChargeParent may also strip salts, which reorders/removes atoms; in
        # that case the index cannot be trusted and is dropped rather than guessed.
        return None
    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(neutral, isomericSmiles=True))
    if canonical is None or canonical.GetNumAtoms() != neutral.GetNumAtoms():
        return None
    source_ranks = list(Chem.CanonicalRankAtoms(neutral, breakTies=True))
    target_ranks = list(Chem.CanonicalRankAtoms(canonical, breakTies=True))
    position = {rank: i for i, rank in enumerate(target_ranks)}
    if not 0 <= atom_idx < len(source_ranks):
        return None
    return position.get(source_ranks[atom_idx])


@functools.lru_cache(maxsize=100_000)
def neutral_parent(smiles: str) -> str:
    """Canonical SMILES of the neutral parent form.

    Datasets disagree about which protonation state to store. Without this a
    compound recorded as a cation in one source and a neutral in another gets
    two identities, which defeats duplicate detection and lets one compound
    appear in both train and test.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeError(f"could not parse {smiles!r}")
    try:
        parent = rdMolStandardize.ChargeParent(mol)
    except Exception:
        parent = _UNCHARGER.uncharge(mol)
    if parent is None:
        raise MoleculeError(f"could not neutralise {smiles!r}")
    return Chem.MolToSmiles(parent, isomericSmiles=True, canonical=True)


@dataclass
class PKaImportReport:
    total: int = 0
    accepted: int = 0
    rejected_parse: int = 0
    rejected_value: int = 0
    charge_normalised: int = 0
    examples: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if len(self.examples) < 8:
            self.examples.append(message)

    def describe(self) -> str:
        lines = [
            f"rows read           {self.total:>8,}",
            f"accepted            {self.accepted:>8,}",
            f"rejected parse      {self.rejected_parse:>8,}",
            f"rejected value      {self.rejected_value:>8,}",
            f"charge-normalised   {self.charge_normalised:>8,}",
        ]
        return "\n".join(lines + [f"  {e}" for e in self.examples])


def _pka_quantity(value: float, *, ionic_strength_M, temperature_K):
    return Quantity(
        float(value),
        LOG10_UNIT,
        QuantityKind.PKA,
        water(ionic_strength_M=ionic_strength_M),
        temperature_K,
    )


def load_sampl6(
    stacked_csv: str | Path, *, source: str = "sampl6-pka"
) -> tuple[list[PKaRecord], PKaImportReport]:
    """Load the SAMPL6 macroscopic pKa values.

    Values measured by cosolvent extrapolation are tagged in metadata; they are
    a different measurement condition from the 100% water titrations even
    though both are reported as aqueous pKa.
    """
    import csv

    report = PKaImportReport()
    records: list[PKaRecord] = []
    with open(stacked_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # pKa IDs look like SM01_pKa1; the trailing integer is the titration step.
    for row in rows:
        report.total += 1
        raw = row["canonical isomeric SMILES"]
        try:
            parent = neutral_parent(raw)
            if parent != canonical_smiles(raw):
                report.charge_normalised += 1
            reference = MoleculeRef.from_smiles(parent)
        except Exception as exc:
            report.rejected_parse += 1
            report.note(f"{raw!r}: {type(exc).__name__}: {exc}")
            continue
        try:
            value = float(row["pKa mean"])
            step = int(row["pKa ID"].rsplit("pKa", 1)[1])
        except Exception as exc:
            report.rejected_value += 1
            report.note(f"{row.get('pKa ID')!r}: {exc}")
            continue

        assay = row["Assay Type"]
        cosolvent = "Yasuda-Shedlovsky" in assay
        sem = row.get("pKa SEM")
        records.append(
            PKaRecord(
                molecule=reference,
                quantity=_pka_quantity(value, ionic_strength_M=0.15, temperature_K=298.15),
                fidelity=EXPERIMENT,
                source=source,
                source_ref=row["pKa ID"],
                uncertainty=(
                    _pka_quantity(abs(float(sem)), ionic_strength_M=0.15, temperature_K=298.15)
                    if sem not in (None, "", "nan")
                    else None
                ),
                resolution="macro",
                step_index=step,
                date="2018-01-01",
                metadata={
                    "assay_type": assay,
                    "cosolvent_extrapolation": cosolvent,
                    "molecule_id": row["Molecule ID"],
                    "as_published_smiles": raw,
                    "instrument": "Sirius T3",
                },
            )
        )
        report.accepted += 1
    return records, report


def load_czodrowskilab(
    sdf_path: str | Path,
    *,
    source: str,
    temperature_K: float | None = None,
    ionic_strength_M: float | None = None,
) -> tuple[list[PKaRecord], PKaImportReport]:
    """Load one of the Baltruschat/Czodrowski SDF files.

    `temperature_K` and `ionic_strength_M` default to None because the source
    does not distribute them. That is deliberate: leaving them unset makes
    `audit_pka_records` raise, which surfaces the gap instead of inventing a
    plausible 298.15 K.
    """
    report = PKaImportReport()
    records: list[PKaRecord] = []
    supplier = Chem.SDMolSupplier(str(sdf_path))
    for index, mol in enumerate(supplier):
        report.total += 1
        if mol is None:
            report.rejected_parse += 1
            report.note(f"record {index}: RDKit returned None")
            continue
        props = mol.GetPropsAsDict()
        try:
            value = float(props["pKa"])
        except Exception:
            report.rejected_value += 1
            report.note(f"record {index}: no usable pKa property")
            continue
        raw = Chem.MolToSmiles(mol)
        try:
            parent = neutral_parent(raw)
            if parent != canonical_smiles(raw):
                report.charge_normalised += 1
            reference = MoleculeRef.from_smiles(parent)
        except Exception as exc:
            report.rejected_parse += 1
            report.note(f"record {index}: {type(exc).__name__}: {exc}")
            continue

        records.append(
            PKaRecord(
                molecule=reference,
                quantity=_pka_quantity(
                    value, ionic_strength_M=ionic_strength_M, temperature_K=temperature_K
                ),
                fidelity=EXPERIMENT,
                source=source,
                source_ref=str(props.get("ID", index)),
                resolution="macro",
                step_index=1,
                metadata={
                    # Marvin's site and type are PREDICTIONS, not measurements.
                    # They must never become site_atom_idx on a macroscopic value.
                    "marvin_predicted_atom": props.get("marvin_atom"),
                    # ... and the raw index is useless downstream without this
                    # remap into the canonical neutral-parent atom ordering.
                    "marvin_predicted_atom_canonical": (
                        map_atom_to_canonical(mol, int(props["marvin_atom"]))
                        if props.get("marvin_atom") is not None
                        else None
                    ),
                    "marvin_predicted_pka": props.get("marvin_pKa"),
                    "marvin_predicted_type": props.get("marvin_pKa_type"),
                    "original_dataset": props.get("original_dataset"),
                    "as_published_smiles": raw,
                },
            )
        )
        report.accepted += 1
    return records, report
