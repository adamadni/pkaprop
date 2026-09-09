"""Unit mismatches must raise, not silently compute."""

from __future__ import annotations

import math

import pytest

from molprop.units import (
    EV,
    GAS,
    HARTREE,
    KCAL_PER_MOL,
    KJ_PER_MOL,
    LOG10_UNIT,
    Phase,
    Quantity,
    QuantityKind,
    UnitError,
    log10_to_molar_energy,
    molar_energy_to_log10,
    rt_ln10_kcal_per_mol,
    water,
)


def g(value, unit=KCAL_PER_MOL, phase=GAS, t=298.15):
    return Quantity(value, unit, QuantityKind.GIBBS, phase, t)


# -- conversions -----------------------------------------------------------


def test_energy_conversions_are_exact():
    q = g(1.0, KCAL_PER_MOL)
    assert q.to(KJ_PER_MOL).value == pytest.approx(4.184)
    assert q.to(HARTREE).value == pytest.approx(1 / 627.5094740631)
    assert q.to(EV).value == pytest.approx(1 / 23.060547830619026)
    assert q.to(KJ_PER_MOL).to(KCAL_PER_MOL).value == pytest.approx(1.0)


def test_one_log_unit_is_1p364_kcal_per_mol():
    assert rt_ln10_kcal_per_mol(298.15) == pytest.approx(1.3642, abs=1e-4)


def test_pka_to_free_energy_round_trip():
    pka = Quantity(4.0, LOG10_UNIT, QuantityKind.PKA, water(), 298.15)
    dg = log10_to_molar_energy(pka)
    assert dg.kind is QuantityKind.DELTA_GIBBS
    assert dg.value == pytest.approx(5.4568, abs=1e-3)
    back = molar_energy_to_log10(dg)
    assert back.value == pytest.approx(4.0)
    assert back.kind is QuantityKind.PKA


def test_tautomer_ratio_uses_the_same_constant():
    ratio = Quantity(1.0, LOG10_UNIT, QuantityKind.LOG_TAUTOMER_RATIO, water(), 298.15)
    assert log10_to_molar_energy(ratio).value == pytest.approx(rt_ln10_kcal_per_mol())


# -- the guards ------------------------------------------------------------


def test_cannot_convert_across_dimensions():
    pka = Quantity(4.0, LOG10_UNIT, QuantityKind.PKA, water(), 298.15)
    with pytest.raises(UnitError, match="cannot convert"):
        pka.to(KCAL_PER_MOL)


def test_cannot_add_enthalpy_to_gibbs():
    h = Quantity(10.0, KCAL_PER_MOL, QuantityKind.ENTHALPY, GAS, 298.15)
    with pytest.raises(UnitError, match="cannot add"):
        g(10.0) + h


def test_cannot_mix_phases():
    with pytest.raises(UnitError, match="across phases"):
        g(1.0, phase=GAS) + g(1.0, phase=water())


def test_cannot_mix_temperatures():
    with pytest.raises(UnitError, match="different temperatures"):
        g(1.0, t=298.15) + g(1.0, t=310.0)


def test_bare_float_is_rejected():
    with pytest.raises(UnitError, match="explicit units"):
        g(1.0) + 2.0


def test_kind_must_match_unit_dimension():
    with pytest.raises(UnitError, match="requires"):
        Quantity(1.0, LOG10_UNIT, QuantityKind.GIBBS, GAS, 298.15)


def test_subtraction_produces_a_delta_kind():
    d = g(10.0) - g(4.0)
    assert d.kind is QuantityKind.DELTA_GIBBS
    assert d.value == pytest.approx(6.0)
    # and a delta minus a delta stays a delta
    assert (d - d).kind is QuantityKind.DELTA_GIBBS


def test_enthalpy_cannot_become_a_pka():
    dh = Quantity(5.0, KCAL_PER_MOL, QuantityKind.DELTA_ENTHALPY, water(), 298.15)
    with pytest.raises(UnitError, match="entropic term"):
        molar_energy_to_log10(dh)


def test_conversion_across_units_before_addition_is_transparent():
    total = g(1.0, KCAL_PER_MOL) + g(4.184, KJ_PER_MOL)
    assert total.unit is KCAL_PER_MOL
    assert total.value == pytest.approx(2.0)


def test_enthalpy_and_gibbs_require_a_temperature():
    with pytest.raises(UnitError, match="requires an explicit temperature"):
        Quantity(1.0, KCAL_PER_MOL, QuantityKind.GIBBS, GAS, None)
    # electronic energy is a 0 K quantity and may omit it
    Quantity(1.0, KCAL_PER_MOL, QuantityKind.ELECTRONIC_ENERGY, GAS, None)


def test_phase_requires_a_solvent():
    with pytest.raises(UnitError, match="explicit solvent"):
        Phase("solution")
    with pytest.raises(UnitError, match="cannot carry a solvent"):
        Phase("gas", "water")


def test_solvent_identity_is_part_of_the_phase():
    aqueous = water()
    methanolic = Phase("solution", "methanol")
    with pytest.raises(UnitError, match="across phases"):
        g(1.0, phase=aqueous) + g(1.0, phase=methanolic)


def test_ionic_strength_distinguishes_conditions():
    with pytest.raises(UnitError, match="across phases"):
        g(1.0, phase=water(0.1)) + g(1.0, phase=water(0.15))


def test_non_finite_values_are_rejected():
    with pytest.raises(UnitError, match="finite"):
        g(math.nan)


def test_serialisation_round_trip():
    q = Quantity(4.2, LOG10_UNIT, QuantityKind.PKA, water(0.15, 7.4), 310.0)
    assert Quantity.from_dict(q.as_dict()) == q
