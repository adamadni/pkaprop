"""Macro/micro pKa aggregation, including the statistical factor."""

from __future__ import annotations

import pytest

from molprop.chem import MoleculeError
from molprop.microstates import (
    Microstate,
    MicrostateEnsemble,
    check_ensemble_formula,
    macro_pka_from_microstates,
    micro_pka_from_macro,
    microstate_populations,
)

# 2-methylsuccinic acid: two distinguishable carboxyls, so the two
# singly-deprotonated microstates are genuinely different structures.
DIACID = "CC(C(=O)O)CC(=O)O"
DIACID_A = "CC(C(=O)[O-])CC(=O)O"
DIACID_B = "CC(C(=O)O)CC(=O)[O-]"
DIACID_AB = "CC(C(=O)[O-])CC(=O)[O-]"


def diacid_ensemble(pka_a: float = 4.0, pka_b: float = 4.0) -> MicrostateEnsemble:
    """Free energies in pK units, relative to the fully protonated state."""
    return MicrostateEnsemble(
        states=(
            Microstate(DIACID, n_protons=2, relative_g_pk_units=0.0),
            Microstate(DIACID_A, n_protons=1, relative_g_pk_units=pka_a),
            Microstate(DIACID_B, n_protons=1, relative_g_pk_units=pka_b),
            Microstate(DIACID_AB, n_protons=0, relative_g_pk_units=pka_a + pka_b),
        )
    )


def test_two_equivalent_sites_give_the_statistical_correction():
    """Two independent sites at micro pKa 4.0 -> macro 3.699 and 4.301.

    Level 2: one state at g=0        -> log10 Z2 = 0
    Level 1: two states at g=4       -> log10 Z1 = log10(2e-4) = -3.699
    Level 0: one state at g=8        -> log10 Z0 = -8
    pKa1 = Z2 - Z1 = 3.699,  pKa2 = Z1 - Z0 = 4.301
    """
    results = macro_pka_from_microstates(diacid_ensemble(4.0, 4.0))
    assert [step for step, _ in results] == [1, 2]
    assert results[0][1].value == pytest.approx(4.0 - 0.30103, abs=1e-4)
    assert results[1][1].value == pytest.approx(4.0 + 0.30103, abs=1e-4)


def test_macro_pkas_sum_to_the_micro_pkas():
    """Whatever the site energies, the two macro constants conserve total dG."""
    results = macro_pka_from_microstates(diacid_ensemble(3.5, 5.5))
    assert sum(pka.value for _, pka in results) == pytest.approx(3.5 + 5.5)


def test_a_dominant_site_pins_the_first_macro_pka():
    """One site far more acidic: macro pKa1 approaches that site's micro pKa."""
    results = macro_pka_from_microstates(diacid_ensemble(3.0, 8.0))
    assert results[0][1].value == pytest.approx(3.0, abs=1e-4)


def test_single_site_macro_equals_micro():
    ensemble = MicrostateEnsemble(
        states=(
            Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),
            Microstate("CC(=O)[O-]", n_protons=0, relative_g_pk_units=4.76),
        )
    )
    (step, pka), = macro_pka_from_microstates(ensemble)
    assert step == 1
    assert pka.value == pytest.approx(4.76)


def test_macro_pka_is_a_solution_phase_pka_quantity():
    ensemble = MicrostateEnsemble(
        states=(
            Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),
            Microstate("CC(=O)[O-]", n_protons=0, relative_g_pk_units=4.76),
        )
    )
    (_, pka), = macro_pka_from_microstates(ensemble)
    assert pka.kind.value == "pKa"
    assert pka.phase.medium == "solution"
    assert pka.temperature_K == pytest.approx(298.15)


# -- the guards ------------------------------------------------------------


def test_duplicate_microstates_are_rejected():
    with pytest.raises(MoleculeError, match="duplicate microstate"):
        MicrostateEnsemble(
            states=(
                Microstate(DIACID, n_protons=2, relative_g_pk_units=0.0),
                Microstate(DIACID_A, n_protons=1, relative_g_pk_units=4.0),
                Microstate(DIACID_A, n_protons=1, relative_g_pk_units=4.0),
            )
        )


def test_charge_must_track_proton_count():
    with pytest.raises(MoleculeError, match="inconsistent"):
        MicrostateEnsemble(
            states=(
                Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),
                # neutral, but claims to have lost a proton
                Microstate("CCO", n_protons=0, relative_g_pk_units=4.0),
            )
        )


def test_missing_protonation_level_is_an_error():
    """A skipped level means an unenumerated microstate, not a zero-population one."""
    ensemble = MicrostateEnsemble(
        states=(
            Microstate(DIACID, n_protons=2, relative_g_pk_units=0.0),
            Microstate(DIACID_AB, n_protons=0, relative_g_pk_units=8.0),
        )
    )
    with pytest.raises(ValueError, match="contiguous"):
        macro_pka_from_microstates(ensemble)


def test_a_single_level_has_no_macro_pka():
    ensemble = MicrostateEnsemble(
        states=(Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),)
    )
    with pytest.raises(ValueError, match="two protonation levels"):
        macro_pka_from_microstates(ensemble)


def test_formula_check_catches_a_different_compound():
    """Charge bookkeeping alone will not notice a wrong structure."""
    ensemble = MicrostateEnsemble(
        states=(
            Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),
            Microstate("CCC(=O)[O-]", n_protons=0, relative_g_pk_units=4.8),
        )
    )
    with pytest.raises(MoleculeError, match="heavy-atom composition"):
        check_ensemble_formula(ensemble)


def test_formula_check_passes_for_a_real_ensemble():
    check_ensemble_formula(diacid_ensemble())


# -- populations -----------------------------------------------------------


def test_populations_sum_to_one_and_favour_acetate_at_ph_74():
    ensemble = MicrostateEnsemble(
        states=(
            Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),
            Microstate("CC(=O)[O-]", n_protons=0, relative_g_pk_units=4.76),
        )
    )
    pops = microstate_populations(ensemble, ph=7.4)
    assert sum(pops.values()) == pytest.approx(1.0)
    assert pops["CC(=O)[O-]"] > 0.99


def test_populations_flip_below_the_pka():
    ensemble = MicrostateEnsemble(
        states=(
            Microstate("CC(=O)O", n_protons=1, relative_g_pk_units=0.0),
            Microstate("CC(=O)[O-]", n_protons=0, relative_g_pk_units=4.76),
        )
    )
    pops = microstate_populations(ensemble, ph=2.0)
    assert pops["CC(=O)O"] > 0.99


def test_macro_to_micro_is_refused():
    with pytest.raises(NotImplementedError, match="underdetermined"):
        micro_pka_from_macro(4.2, 9.1)
