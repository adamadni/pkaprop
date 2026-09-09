"""Record-level invariants: bonds are real, pKa declare resolution, tautomers match."""

from __future__ import annotations

import pytest
from conftest import find_atom, find_bond

from molprop.chem import BondRef, MoleculeError
from molprop.records import BDERecord, MoleculeRef, PKaRecord, TautomerRecord
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


def bde(value=86.7):
    return Quantity(value, KCAL_PER_MOL, QuantityKind.BDE, GAS, 298.15)


def pka(value=4.76):
    return Quantity(value, LOG10_UNIT, QuantityKind.PKA, water(), 298.15)


def logkt(value=0.5):
    return Quantity(value, LOG10_UNIT, QuantityKind.LOG_TAUTOMER_RATIO, water(), 298.15)


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


# -- BDE -------------------------------------------------------------------


def test_bde_record_round_trips(dft_gas):
    ref = MoleculeRef.from_smiles(PHENOL)
    rec = BDERecord(
        molecule=ref,
        quantity=bde(86.7),
        fidelity=dft_gas,
        source="alfabet",
        bond=find_bond(PHENOL, "O", "H"),
    )
    assert rec.bond_type == "H-O"
    assert rec.is_acidic_xh
    assert rec.group_key == ref.inchikey
    assert rec.as_dict()["bond_type"] == "H-O"


def test_bde_on_a_ch_bond_is_not_acidic(dft_gas):
    rec = BDERecord(
        molecule=MoleculeRef.from_smiles(PHENOL),
        quantity=bde(112.9),
        fidelity=dft_gas,
        source="alfabet",
        bond=find_bond(PHENOL, "C", "H"),
    )
    assert rec.bond_type == "C-H"
    assert not rec.is_acidic_xh


def test_bde_requires_a_real_bond(dft_gas):
    with pytest.raises(MoleculeError, match="no bond between"):
        BDERecord(
            molecule=MoleculeRef.from_smiles(PHENOL),
            quantity=bde(),
            fidelity=dft_gas,
            source="alfabet",
            bond=BondRef(0, 5),
        )


def test_bde_rejects_an_out_of_range_index(dft_gas):
    with pytest.raises(MoleculeError, match="out of range"):
        BDERecord(
            molecule=MoleculeRef.from_smiles(PHENOL),
            quantity=bde(),
            fidelity=dft_gas,
            source="alfabet",
            bond=BondRef(0, 999),
        )


def test_bde_must_be_gas_phase(dft_water):
    with pytest.raises(UnitError):
        BDERecord(
            molecule=MoleculeRef.from_smiles(PHENOL),
            quantity=bde(),
            fidelity=dft_water,
            source="alfabet",
            bond=find_bond(PHENOL, "O", "H"),
        )


def test_a_pka_quantity_cannot_be_a_bde_record(dft_gas):
    with pytest.raises(UnitError, match="requires a quantity of kind"):
        BDERecord(
            molecule=MoleculeRef.from_smiles(PHENOL),
            quantity=pka(),
            fidelity=dft_gas,
            source="alfabet",
            bond=find_bond(PHENOL, "O", "H"),
        )


def test_gas_fidelity_rejects_a_solution_value(dft_gas):
    solution_bde = Quantity(86.7, KCAL_PER_MOL, QuantityKind.BDE, water(), 298.15)
    with pytest.raises(UnitError, match="gas-phase"):
        BDERecord(
            molecule=MoleculeRef.from_smiles(PHENOL),
            quantity=solution_bde,
            fidelity=dft_gas,
            source="alfabet",
            bond=find_bond(PHENOL, "O", "H"),
        )


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


# -- tautomers -------------------------------------------------------------


def test_tautomer_record(experiment):
    rec = TautomerRecord(
        molecule=MoleculeRef.from_smiles("CC(=O)CC(C)=O"),
        quantity=logkt(0.8),
        fidelity=experiment,
        source="tautobase",
        tautomer_a_smiles="CC(=O)CC(C)=O",
        tautomer_b_smiles="CC(O)=CC(C)=O",
    )
    assert "|" in rec.group_key


def test_tautomers_must_share_a_formula(experiment):
    with pytest.raises(MoleculeError, match="molecular formula"):
        TautomerRecord(
            molecule=MoleculeRef.from_smiles("CC(=O)CC(C)=O"),
            quantity=logkt(),
            fidelity=experiment,
            source="tautobase",
            tautomer_a_smiles="CC(=O)CC(C)=O",
            tautomer_b_smiles="CCC(O)=CC(C)=O",
        )


def test_tautomers_must_differ(experiment):
    with pytest.raises(MoleculeError, match="same structure"):
        TautomerRecord(
            molecule=MoleculeRef.from_smiles("CC(=O)CC(C)=O"),
            quantity=logkt(),
            fidelity=experiment,
            source="tautobase",
            tautomer_a_smiles="CC(=O)CC(C)=O",
            tautomer_b_smiles="CC(=O)CC(C)=O",
        )


def test_tautomer_group_key_is_order_independent(experiment):
    def make(a, b):
        return TautomerRecord(
            molecule=MoleculeRef.from_smiles("CC(=O)CC(C)=O"),
            quantity=logkt(),
            fidelity=experiment,
            source="tautobase",
            tautomer_a_smiles=a,
            tautomer_b_smiles=b,
        )

    keto, enol = "CC(=O)CC(C)=O", "CC(O)=CC(C)=O"
    assert make(keto, enol).group_key == make(enol, keto).group_key


# -- uncertainty -----------------------------------------------------------


def test_uncertainty_must_share_a_dimension(dft_gas):
    with pytest.raises(UnitError, match="dimension"):
        BDERecord(
            molecule=MoleculeRef.from_smiles(PHENOL),
            quantity=bde(),
            fidelity=dft_gas,
            source="alfabet",
            bond=find_bond(PHENOL, "O", "H"),
            uncertainty=Quantity(0.1, LOG10_UNIT, QuantityKind.PKA, GAS, 298.15),
        )


def test_record_id_is_stable_and_bond_specific(dft_gas):
    ref = MoleculeRef.from_smiles(PHENOL)
    oh = BDERecord(
        molecule=ref, quantity=bde(), fidelity=dft_gas, source="alfabet",
        bond=find_bond(PHENOL, "O", "H"),
    )
    ch = BDERecord(
        molecule=ref, quantity=bde(), fidelity=dft_gas, source="alfabet",
        bond=find_bond(PHENOL, "C", "H"),
    )
    assert oh.record_id != ch.record_id
    assert oh.record_id == BDERecord(
        molecule=ref, quantity=bde(99.0), fidelity=dft_gas, source="alfabet",
        bond=find_bond(PHENOL, "O", "H"),
    ).record_id, "the id identifies the measurement slot, not the value"
