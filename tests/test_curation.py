"""Curation contract checks for pKa and tautomer batches."""

from __future__ import annotations

import pytest
from conftest import find_atom

from molprop.curation import (
    KNOWN_PKA_SOURCES,
    audit_pka_records,
    ionisable_sites,
    summarise,
)
from molprop.records import MoleculeRef, PKaRecord
from molprop.units import LOG10_UNIT, Phase, Quantity, QuantityKind, water


def pka_q(value=4.76, phase=None, t=298.15):
    return Quantity(value, LOG10_UNIT, QuantityKind.PKA, phase or water(0.15), t)


def macro(smiles, value=4.76, step=1, phase=None, source="iupac", date="2001-01-01", **kw):
    return PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=pka_q(value, phase),
        source=source,
        resolution="macro",
        step_index=step,
        date=date,
        **kw,
    )


def codes(issues):
    return {i.code for i in issues}


# -- source specs ----------------------------------------------------------


def test_source_specs_state_caveats_and_leave_provenance_unset():
    for spec in KNOWN_PKA_SOURCES:
        assert spec.caveats, f"{spec.name} records no caveats"
        assert spec.doi is None and spec.n_records is None, (
            "provenance must be filled from the downloaded artefact, not from memory"
        )


# -- ionisable sites -------------------------------------------------------


def test_ionisable_sites_finds_the_obvious_groups():
    names = {n for n, _ in ionisable_sites("NCCC(=O)O")}
    assert "carboxylic_acid" in names
    assert "aliphatic_amine" in names, (
        "an amine must be reported by its in-window equilibrium (protonation), "
        "not by its ~pKa 35 N-H deprotonation"
    )


def test_each_ionisable_atom_is_reported_once():
    """An aniline N is both an acid and a base site; it is one centre."""
    sites = ionisable_sites("Nc1ccccc1")
    assert len({idx for _, idx in sites}) == len(sites)


def test_ionisable_sites_finds_phenols():
    assert any(n == "phenol" for n, _ in ionisable_sites("Oc1ccccc1"))


def test_ionisable_sites_is_empty_for_an_alkane():
    assert ionisable_sites("CCCC") == []


# -- pKa audit -------------------------------------------------------------


def test_clean_batch_has_no_errors(experiment):
    records = [
        macro("CC(=O)O", 4.76, fidelity=experiment),
        macro("OC(=O)c1ccccc1", 4.20, fidelity=experiment),
    ]
    issues = audit_pka_records(records)
    assert not [i for i in issues if i.severity == "error"], summarise(issues)


def test_mixed_solvent_is_an_error(experiment):
    records = [
        macro("CC(=O)O", 4.76, fidelity=experiment),
        macro(
            "OC(=O)c1ccccc1",
            9.40,
            phase=Phase("solution", "methanol", 0.15),
            fidelity=experiment,
        ),
    ]
    assert "mixed-solvent" in codes(audit_pka_records(records))


def test_mixed_resolution_is_an_error(experiment):
    smiles = "CC(=O)O"
    micro = PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=pka_q(4.76),
        fidelity=experiment,
        source="sampl6",
        resolution="micro",
        site_atom_idx=find_atom(smiles, "O", with_h=True),
        protonated_smiles="CC(=O)O",
        deprotonated_smiles="CC(=O)[O-]",
        date="2018-01-01",
    )
    records = [macro("OC(=O)c1ccccc1", fidelity=experiment), micro]
    assert "mixed-resolution" in codes(audit_pka_records(records))


def test_non_contiguous_steps_are_an_error(experiment):
    records = [
        macro("NCCC(=O)O", 3.6, step=1, fidelity=experiment),
        macro("NCCC(=O)O", 10.2, step=3, fidelity=experiment),
    ]
    assert "non-contiguous-steps" in codes(audit_pka_records(records))


def test_polyprotic_with_one_step_is_flagged(experiment):
    records = [macro("NCCC(=O)O", 3.6, step=1, fidelity=experiment)]
    assert "incomplete-titration" in codes(audit_pka_records(records))


def test_monoprotic_with_one_step_is_not_flagged(experiment):
    records = [macro("CC(=O)O", 4.76, step=1, fidelity=experiment)]
    assert "incomplete-titration" not in codes(audit_pka_records(records))


def test_implausible_values_are_flagged(experiment):
    records = [macro("CC(=O)O", 41.0, fidelity=experiment)]
    assert "implausible-value" in codes(audit_pka_records(records))


def test_missing_date_blocks_a_time_split(experiment):
    records = [macro("CC(=O)O", date=None, fidelity=experiment)]
    assert "missing-date" in codes(audit_pka_records(records))


def test_conflicting_duplicates_are_an_error(experiment):
    records = [
        macro("CC(=O)O", 4.76, fidelity=experiment),
        macro("CC(=O)O", 6.90, fidelity=experiment),
    ]
    issues = audit_pka_records(records)
    assert "conflicting-duplicate" in codes(issues)


def test_agreeing_duplicates_are_not_flagged(experiment):
    records = [
        macro("CC(=O)O", 4.76, fidelity=experiment),
        macro("CC(=O)O", 4.80, fidelity=experiment),
    ]
    assert "conflicting-duplicate" not in codes(audit_pka_records(records))


def test_a_site_without_a_proton_is_caught(experiment):
    smiles = "CC(=O)O"
    # index 1 is the carbonyl carbon: no hydrogen on it
    rec = PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=pka_q(4.76),
        fidelity=experiment,
        source="sampl6",
        resolution="micro",
        site_atom_idx=1,
        protonated_smiles="CC(=O)O",
        deprotonated_smiles="CC(=O)[O-]",
        date="2018-01-01",
    )
    assert "site-has-no-proton" in codes(audit_pka_records([rec]))


def test_summarise_reports_counts(experiment):
    records = [macro("CC(=O)O", 41.0, date=None, fidelity=experiment)]
    text = summarise(audit_pka_records(records))
    assert "warning" in text and "implausible-value" in text


def test_summarise_is_quiet_when_clean():
    assert summarise([]) == "no curation issues"
