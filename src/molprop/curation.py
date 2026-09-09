"""Curation contract for pKa and tautomer data.

This is the schema-defining half of the data work: what a source must supply
before its records may be registered, and the checks that run over a candidate
batch. It does not download or parse anything -- that belongs to step 4, once
these decisions are settled.

The checks encode the failure modes that make aggregated literature pKa and
tautomer sets untrustworthy, in rough order of how often they bite:

* macroscopic labels used as if they were site-specific
* mixed or unstated solvent, temperature and ionic strength
* the same compound appearing twice with irreconcilable values
* "tautomer pairs" that are not related by tautomerism at all
* qualitative observations ("only the keto form is seen") encoded as numbers
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .records import PKaRecord, PropertyRecord, TautomerRecord

__all__ = [
    "SourceSpec",
    "KNOWN_PKA_SOURCES",
    "KNOWN_TAUTOMER_SOURCES",
    "CurationIssue",
    "audit_pka_records",
    "audit_tautomer_records",
    "ionisable_sites",
    "are_tautomers",
]

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class SourceSpec:
    """What a data source provides, and what it does not.

    `url`, `doi` and `n_records` are left unset here on purpose: they get filled
    in during step 4 from the artefact actually downloaded, not from memory, so
    that a manifest never carries a provenance field nobody verified.
    """

    name: str
    resolution: Literal["macro", "micro", "mixed", "n/a"]
    caveats: tuple[str, ...]
    url: str | None = None
    doi: str | None = None
    license: str | None = None
    n_records: int | None = None


#: Candidate pKa sources. Sizes and identifiers are deliberately absent -- fill
#: them in from the downloaded artefact during step 4.
KNOWN_PKA_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="sampl6-pka",
        resolution="micro",
        caveats=(
            "small (tens of compounds) but the microstate assignments are careful",
            "blind-challenge design makes it the right external validation set, "
            "not a training set",
        ),
    ),
    SourceSpec(
        name="sampl7-pka",
        resolution="micro",
        caveats=("same shape as SAMPL6; keep as held-out validation",),
    ),
    SourceSpec(
        name="iupac-dissociation-constants",
        resolution="macro",
        caveats=(
            "digitised from print; transcription errors are common",
            "many entries are mixed-solvent or non-298.15 K and must not be pooled",
            "ionic strength frequently unstated",
        ),
    ),
    SourceSpec(
        name="chembl-derived",
        resolution="macro",
        caveats=(
            "assay conditions vary enormously; requires aggressive filtering",
            "the same compound often appears with several irreconcilable values",
        ),
    ),
    SourceSpec(
        name="novartis-az-published",
        resolution="macro",
        caveats=(
            "measured under a consistent in-house protocol, which is its main value",
            "drug-like and therefore polyprotic; step assignment must be checked",
        ),
    ),
    SourceSpec(
        name="datawarrior-pka",
        resolution="macro",
        caveats=("aggregated from several primary sources; provenance per record is thin",),
    ),
)

#: Candidate tautomer sources. Quantitative ratios are genuinely scarce.
KNOWN_TAUTOMER_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="tautobase",
        resolution="n/a",
        caveats=(
            "many entries are qualitative or solvent-heterogeneous",
            "a substantial fraction are not quantitative ratios at all",
        ),
    ),
    SourceSpec(
        name="wahl-sander",
        resolution="n/a",
        caveats=("small and careful; treat as validation",),
    ),
    SourceSpec(
        name="sampl2",
        resolution="n/a",
        caveats=("blind-challenge tautomer ratios; very small",),
    ),
)


@dataclass(frozen=True)
class CurationIssue:
    severity: Severity
    code: str
    message: str
    record_ids: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{len(self.record_ids)} records]" if self.record_ids else ""
        return f"{self.severity.upper()} {self.code}: {self.message}{where}"


# --------------------------------------------------------------------------
# Structural helpers
# --------------------------------------------------------------------------

def ionisable_sites(smiles: str) -> list[tuple[str, int]]:
    """(group label, atom index) for plausible ionisable centres.

    Delegates to `enumeration.find_protonation_sites`, which is validated
    against SAMPL6. An earlier version of this function used a small ad-hoc
    SMARTS list that over-reported badly -- it called 197 of the 280 Novartis
    compounds polyprotic when the source had curated them as monoprotic -- so
    any polyprotic count derived from it was an upper bound, not a measurement.

    Atoms are reported once even when they are both an acid and a base site (an
    aniline nitrogen is both), because the question this answers is "how many
    ionisable centres does the molecule have", not "how many equilibria".

    Still an upper bound on the count that matters: it does not filter to the
    measurable pH window, so an aniline N-H whose deprotonation sits near pKa 30
    is counted alongside the ring nitrogen that titrates at 5.
    """
    from .enumeration import find_protonation_sites

    try:
        sites = find_protonation_sites(smiles)
    except Exception:
        return []
    # When one atom is both an acid and a base site, report the equilibrium
    # that actually titrates. An aliphatic amine protonates around pKa 10; its
    # N-H deprotonation sits near 35 and never appears in a titration.
    out_of_window_acids = {"amine_NH", "aniline_NH", "alcohol"}
    best: dict[int, str] = {}
    for site in sites:
        current = best.get(site.atom_idx)
        if current is None or current in out_of_window_acids:
            best[site.atom_idx] = site.label
    return sorted((label, idx) for idx, label in best.items())


def are_tautomers(smiles_a: str, smiles_b: str, *, max_tautomers: int = 200) -> bool:
    """True if B appears in RDKit's tautomer enumeration of A.

    Used to catch "tautomer pairs" that are actually different compounds or
    different protonation states. RDKit's enumerator is incomplete, so a False
    is a warning to check by hand, not a verdict.
    """
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return False
    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = max_tautomers
    enumerator = rdMolStandardize.TautomerEnumerator(params)
    target = Chem.MolToSmiles(mol_b, canonical=True)
    for taut in enumerator.Enumerate(mol_a):
        if Chem.MolToSmiles(taut, canonical=True) == target:
            return True
    # Fall back on comparing canonical tautomers, which is weaker but catches
    # pairs the enumeration truncated.
    try:
        return Chem.MolToSmiles(enumerator.Canonicalize(mol_a)) == Chem.MolToSmiles(
            enumerator.Canonicalize(mol_b)
        )
    except Exception:  # pragma: no cover - RDKit can throw on odd valences
        return False


# --------------------------------------------------------------------------
# Batch audits
# --------------------------------------------------------------------------


def _conflicting_duplicates(
    records: Sequence[PropertyRecord], tolerance: float
) -> list[CurationIssue]:
    by_slot: dict[str, list[PropertyRecord]] = defaultdict(list)
    for r in records:
        by_slot[r.record_id].append(r)
    issues = []
    for slot, group in sorted(by_slot.items()):
        if len(group) < 2:
            continue
        values = [r.quantity.canonical_value for r in group]
        spread = max(values) - min(values)
        if spread > tolerance:
            issues.append(
                CurationIssue(
                    "error",
                    "conflicting-duplicate",
                    f"{len(group)} records share slot {slot} but span "
                    f"{spread:.2f} units ({min(values):.2f} to {max(values):.2f}); "
                    f"pick one with a documented rule or drop the compound",
                    tuple(r.record_id for r in group),
                )
            )
    return issues


def audit_pka_records(
    records: Sequence[PKaRecord],
    *,
    duplicate_tolerance: float = 0.5,
    plausible_range: tuple[float, float] = (-2.0, 16.0),
) -> list[CurationIssue]:
    """Check a candidate pKa batch against the curation contract."""
    issues: list[CurationIssue] = []
    if not records:
        return issues

    no_temp = [r for r in records if r.quantity.temperature_K is None]
    if no_temp:
        issues.append(
            CurationIssue(
                "error",
                "missing-temperature",
                "pKa without a temperature cannot be converted to a free energy",
                tuple(r.record_id for r in no_temp),
            )
        )

    solvents = {r.quantity.phase.solvent for r in records}
    if len(solvents) > 1:
        issues.append(
            CurationIssue(
                "error",
                "mixed-solvent",
                f"batch mixes solvents {sorted(s or '?' for s in solvents)}; "
                f"mixed-solvent pKa are not comparable to aqueous ones",
            )
        )

    no_ionic = [r for r in records if r.quantity.phase.ionic_strength_M is None]
    if no_ionic:
        issues.append(
            CurationIssue(
                "warning",
                "missing-ionic-strength",
                "ionic strength unstated; acceptable for a single-source batch, "
                "not for a pooled one",
                tuple(r.record_id for r in no_ionic),
            )
        )

    lo, hi = plausible_range
    out_of_range = [r for r in records if not lo <= r.quantity.value <= hi]
    if out_of_range:
        issues.append(
            CurationIssue(
                "warning",
                "implausible-value",
                f"values outside [{lo}, {hi}] are usually extrapolations or unit "
                f"errors rather than measurements",
                tuple(r.record_id for r in out_of_range),
            )
        )

    # Macroscopic records on polyprotic compounds: check the titration is whole.
    macro = [r for r in records if r.resolution == "macro"]
    steps: dict[str, set[int]] = defaultdict(set)
    for r in macro:
        steps[r.molecule.smiles].add(r.step_index or 0)
    incomplete = []
    for smiles, present in sorted(steps.items()):
        n_sites = len(ionisable_sites(smiles))
        if n_sites > 1 and len(present) < 2:
            incomplete.append(smiles)
        if present and sorted(present) != list(range(1, max(present) + 1)):
            issues.append(
                CurationIssue(
                    "error",
                    "non-contiguous-steps",
                    f"{smiles} has macro steps {sorted(present)}; a missing step "
                    f"means an unrecorded equilibrium, not an absent one",
                )
            )
    if incomplete:
        issues.append(
            CurationIssue(
                "warning",
                "incomplete-titration",
                f"{len(incomplete)} polyprotic compounds carry a single macro pKa; "
                f"a site model trained on these is being fitted to whichever "
                f"equilibrium the experimenter happened to report",
            )
        )

    # Microscopic records: the named site must actually bear a proton.
    bad_site = []
    for r in records:
        if r.resolution != "micro" or r.protonated_smiles is None:
            continue
        mol = Chem.AddHs(Chem.MolFromSmiles(r.molecule.smiles))
        idx = r.site_atom_idx
        if idx is None or idx >= mol.GetNumAtoms():
            bad_site.append(r)
            continue
        atom = mol.GetAtomWithIdx(idx)
        if not any(n.GetSymbol() == "H" for n in atom.GetNeighbors()):
            bad_site.append(r)
    if bad_site:
        issues.append(
            CurationIssue(
                "error",
                "site-has-no-proton",
                "microscopic pKa names a site that carries no hydrogen in the "
                "reference structure; the index is probably against a different "
                "atom ordering",
                tuple(r.record_id for r in bad_site),
            )
        )

    mixed_resolution = {r.resolution for r in records}
    if len(mixed_resolution) > 1:
        issues.append(
            CurationIssue(
                "error",
                "mixed-resolution",
                "batch mixes macroscopic and microscopic pKa; they are different "
                "targets and must be registered as separate datasets",
            )
        )

    no_date = [r for r in records if not r.date]
    if no_date:
        issues.append(
            CurationIssue(
                "warning",
                "missing-date",
                "records without a date cannot take part in a time split",
                tuple(r.record_id for r in no_date),
            )
        )

    issues.extend(_conflicting_duplicates(records, duplicate_tolerance))
    return issues


def audit_tautomer_records(
    records: Sequence[TautomerRecord],
    *,
    duplicate_tolerance: float = 0.3,
    saturation_threshold: float = 4.0,
    check_tautomerism: bool = True,
) -> list[CurationIssue]:
    """Check a candidate tautomer-ratio batch against the curation contract."""
    issues: list[CurationIssue] = []
    if not records:
        return issues

    no_temp = [r for r in records if r.quantity.temperature_K is None]
    if no_temp:
        issues.append(
            CurationIssue(
                "error",
                "missing-temperature",
                "a log ratio without a temperature is not a free energy",
                tuple(r.record_id for r in no_temp),
            )
        )

    solvents = {r.quantity.phase.solvent for r in records}
    if len(solvents) > 1:
        issues.append(
            CurationIssue(
                "error",
                "mixed-solvent",
                f"tautomer ratios are strongly solvent dependent; this batch mixes "
                f"{sorted(s or '?' for s in solvents)}",
            )
        )

    saturated = [r for r in records if abs(r.quantity.value) >= saturation_threshold]
    if saturated:
        issues.append(
            CurationIssue(
                "warning",
                "saturated-ratio",
                f"|log K| >= {saturation_threshold} usually encodes a qualitative "
                f"observation ('only one form is seen') as a number; these are "
                f"censored values and should be modelled as such or excluded",
                tuple(r.record_id for r in saturated),
            )
        )

    if check_tautomerism:
        not_tautomers = [
            r
            for r in records
            if not are_tautomers(r.tautomer_a_smiles, r.tautomer_b_smiles)
        ]
        if not_tautomers:
            issues.append(
                CurationIssue(
                    "warning",
                    "not-a-tautomer-pair",
                    "RDKit's enumeration does not connect these structures; either "
                    "the enumerator is incomplete for this chemotype or the pair is "
                    "mislabelled -- check by hand before registering",
                    tuple(r.record_id for r in not_tautomers),
                )
            )

    issues.extend(_conflicting_duplicates(records, duplicate_tolerance))
    return issues


def summarise(issues: Sequence[CurationIssue]) -> str:
    if not issues:
        return "no curation issues"
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    lines = [f"{len(errors)} error(s), {len(warnings)} warning(s)"]
    for issue in (*errors, *warnings):
        lines.append(f"  {issue}")
    return "\n".join(lines)
