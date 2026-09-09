"""pKa import: the three decisions that would otherwise be silent errors."""

from __future__ import annotations

import textwrap

import pytest

from molprop.curation import audit_pka_records
from molprop.datasets.pka import (
    CZODROWSKILAB_PROVENANCE,
    SAMPL6_PROVENANCE,
    load_sampl6,
    neutral_parent,
)
from molprop.units import QuantityKind

STACKED = textwrap.dedent(
    """\
    Assay Type,Experimental Molecule ID,Molecule ID,canonical isomeric SMILES,pKa ID,pKa SEM,pKa mean
    UV-metric pKa,M01,SM01,c1cc2c(cc1O)c3c(o2)C(=O)NCCC3,SM01_pKa1,0.01,9.53
    "UV-metric psKa, Yasuda-Shedlovsky Extrapolation",M02,SM02,c1ccc2c(c1)c(ncn2)Nc3cccc(c3)C(F)(F)F,SM02_pKa1,0.01,5.03
    UV-metric pKa,M14,SM14,Cc1ccccc1NC(=O)c1ccc(N)cc1,SM14_pKa1,0.01,2.58
    UV-metric pKa,M14,SM14,Cc1ccccc1NC(=O)c1ccc(N)cc1,SM14_pKa2,0.03,5.30
    """
)


@pytest.fixture
def stacked(tmp_path):
    path = tmp_path / "sampl6.csv"
    path.write_text(STACKED, encoding="utf-8")
    return path


# -- charge normalisation --------------------------------------------------


def test_charged_and_neutral_forms_get_one_identity():
    """The Novartis file stores cations; others store neutral forms."""
    assert neutral_parent("C[NH+](C)C") == neutral_parent("CN(C)C")


def test_carboxylate_normalises_to_the_acid():
    assert neutral_parent("CC(=O)[O-]") == neutral_parent("CC(=O)O")


def test_neutral_parent_preserves_the_skeleton():
    assert neutral_parent("Brc1ccc(C2CN3C=CSC3=[NH+]2)cc1").count("Br") == 1


def test_unparseable_smiles_raises():
    from molprop.chem import MoleculeError

    with pytest.raises(MoleculeError):
        neutral_parent("not_a_molecule[[[")


# -- SAMPL6 ----------------------------------------------------------------


def test_records_are_macroscopic(stacked):
    records, _ = load_sampl6(stacked)
    assert records and all(r.resolution == "macro" for r in records)
    assert all(r.site_atom_idx is None for r in records)


def test_step_index_comes_from_the_pka_id(stacked):
    records, _ = load_sampl6(stacked)
    steps = {r.source_ref: r.step_index for r in records}
    assert steps["SM14_pKa1"] == 1
    assert steps["SM14_pKa2"] == 2


def test_conditions_are_recorded(stacked):
    records, _ = load_sampl6(stacked)
    q = records[0].quantity
    assert q.kind is QuantityKind.PKA
    assert q.phase.solvent == "water"
    assert q.phase.ionic_strength_M == pytest.approx(0.15)
    assert q.temperature_K == pytest.approx(298.15)


def test_cosolvent_extrapolation_is_flagged(stacked):
    records, _ = load_sampl6(stacked)
    by_ref = {r.source_ref: r for r in records}
    assert by_ref["SM02_pKa1"].metadata["cosolvent_extrapolation"] is True
    assert by_ref["SM01_pKa1"].metadata["cosolvent_extrapolation"] is False


def test_sem_becomes_uncertainty(stacked):
    records, _ = load_sampl6(stacked)
    by_ref = {r.source_ref: r for r in records}
    assert by_ref["SM14_pKa2"].uncertainty.value == pytest.approx(0.03)


def test_published_smiles_is_kept(stacked):
    records, _ = load_sampl6(stacked)
    assert all("as_published_smiles" in r.metadata for r in records)


def test_a_clean_sampl6_batch_has_no_audit_errors(stacked):
    records, _ = load_sampl6(stacked)
    errors = [i for i in audit_pka_records(records) if i.severity == "error"]
    assert not errors, [str(e) for e in errors]


def test_multiprotic_molecule_keeps_contiguous_steps(stacked):
    records, _ = load_sampl6(stacked)
    codes = {i.code for i in audit_pka_records(records)}
    assert "non-contiguous-steps" not in codes


# -- provenance ------------------------------------------------------------


def test_provenance_records_conditions_or_says_it_cannot():
    assert "0.15 M KCl" in SAMPL6_PROVENANCE["conditions"]
    assert CZODROWSKILAB_PROVENANCE["conditions"] == "not distributed with the data"


def test_missing_temperature_is_an_error_not_a_guess():
    """Leaving temperature unset must surface, not be filled with 298.15."""
    from molprop.datasets.pka import EXPERIMENT
    from molprop.records import MoleculeRef, PKaRecord
    from molprop.units import LOG10_UNIT, Quantity, water

    record = PKaRecord(
        molecule=MoleculeRef.from_smiles("CC(=O)O"),
        quantity=Quantity(4.76, LOG10_UNIT, QuantityKind.PKA, water(), None),
        fidelity=EXPERIMENT,
        source="czodrowskilab-train",
        resolution="macro",
        step_index=1,
    )
    codes = {i.code for i in audit_pka_records([record])}
    assert "missing-temperature" in codes
