"""Record-level invariants: a pKa must declare macro vs micro, and stay coherent."""

from __future__ import annotations

import pytest
from conftest import find_atom

from molprop.chem import MoleculeError
from molprop.records import MoleculeRef, PKaRecord
from molprop.units import (
    GAS,
    KCAL_PER_MOL,
    LOG10_UNIT,
    Quantity,
    QuantityKind,
    UnitError,
    water,
)

PHENOL = "Oc1ccccc1"
ACETIC = "CC(=O)O"


def pka(value=4.76):
    return Quantity(value, LOG10_UNIT, QuantityKind.PKA, water(), 298.15)


# -- molecule identity -----------------------------------------------------


def test_molecule_ref_canonicalises():
    a = MoleculeRef.from_smiles("c1ccccc1O")
    b = MoleculeRef.from_smiles("Oc1ccccc1")
    assert a == b
    assert a.inchikey.startswith("ISWSIDIOOBJBQZ")


def test_atom_indexing_is_deterministic():
    ref = MoleculeRef.from_smiles(PHENOL)
    first = [a.GetSymbol() for a in ref.indexed().GetAtoms()]
    second = [a.GetSymbol() for a in MoleculeRef.from_smiles("c1ccccc1O").indexed().GetAtoms()]
    assert first == second


# -- pKa -------------------------------------------------------------------


def test_macro_pka_record(experiment):
    rec = PKaRecord(
        molecule=MoleculeRef.from_smiles(ACETIC),
        quantity=pka(4.76),
        fidelity=experiment,
        source="iupac",
        resolution="macro",
        step_index=1,
    )
    assert rec.as_dict()["resolution"] == "macro"


def test_micro_pka_record(experiment):
    rec = PKaRecord(
        molecule=MoleculeRef.from_smiles(ACETIC),
        quantity=pka(4.76),
        fidelity=experiment,
        source="sampl6",
        resolution="micro",
        site_atom_idx=find_atom(ACETIC, "O", with_h=True),
        protonated_smiles="CC(=O)O",
        deprotonated_smiles="CC(=O)[O-]",
    )
    assert rec.deprotonated_smiles == "CC(=O)[O-]"


def test_resolution_must_be_declared(experiment):
    with pytest.raises(ValueError, match="resolution"):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=pka(),
            fidelity=experiment,
            source="iupac",
        )


def test_macro_pka_cannot_name_a_site(experiment):
    with pytest.raises(ValueError, match="microscopic claim"):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=pka(),
            fidelity=experiment,
            source="iupac",
            resolution="macro",
            step_index=1,
            site_atom_idx=3,
        )


def test_micro_pka_needs_both_microstates(experiment):
    with pytest.raises(ValueError, match="both microstate SMILES"):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=pka(),
            fidelity=experiment,
            source="sampl6",
            resolution="micro",
            site_atom_idx=find_atom(ACETIC, "O", with_h=True),
            protonated_smiles="CC(=O)O",
        )


def test_deprotonation_must_lower_the_charge_by_one(experiment):
    with pytest.raises(MoleculeError, match="lower the formal charge"):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=pka(),
            fidelity=experiment,
            source="sampl6",
            resolution="micro",
            site_atom_idx=find_atom(ACETIC, "O", with_h=True),
            protonated_smiles="CC(=O)O",
            deprotonated_smiles="CC(=O)O",
        )


def test_pka_must_be_solution_phase(experiment):
    gas_pka = Quantity(4.76, LOG10_UNIT, QuantityKind.PKA, GAS, 298.15)
    with pytest.raises(UnitError):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=gas_pka,
            fidelity=experiment,
            source="iupac",
            resolution="macro",
            step_index=1,
        )


# -- uncertainty and identity ----------------------------------------------


def test_uncertainty_must_share_a_dimension(experiment):
    with pytest.raises(UnitError, match="dimension"):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=pka(),
            fidelity=experiment,
            source="iupac",
            resolution="macro",
            step_index=1,
            uncertainty=Quantity(0.1, KCAL_PER_MOL, QuantityKind.GIBBS, water(), 298.15),
        )


def test_negative_uncertainty_is_rejected(experiment):
    with pytest.raises(UnitError, match="non-negative"):
        PKaRecord(
            molecule=MoleculeRef.from_smiles(ACETIC),
            quantity=pka(),
            fidelity=experiment,
            source="iupac",
            resolution="macro",
            step_index=1,
            uncertainty=Quantity(-0.1, LOG10_UNIT, QuantityKind.PKA, water(), 298.15),
        )


def test_record_id_distinguishes_titration_steps(experiment):
    ref = MoleculeRef.from_smiles("NCCC(=O)O")
    first = PKaRecord(molecule=ref, quantity=pka(3.6), fidelity=experiment,
                      source="iupac", resolution="macro", step_index=1)
    second = PKaRecord(molecule=ref, quantity=pka(10.2), fidelity=experiment,
                       source="iupac", resolution="macro", step_index=2)
    assert first.record_id != second.record_id


def test_record_id_is_stable_across_values(experiment):
    """The id names the measurement slot, not the value."""
    ref = MoleculeRef.from_smiles(ACETIC)
    a = PKaRecord(molecule=ref, quantity=pka(4.76), fidelity=experiment,
                  source="iupac", resolution="macro", step_index=1)
    b = PKaRecord(molecule=ref, quantity=pka(4.80), fidelity=experiment,
                  source="iupac", resolution="macro", step_index=1)
    assert a.record_id == b.record_id
