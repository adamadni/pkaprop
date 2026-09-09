"""Step 2: conformers, calculation specs, thermochemistry, BDE assembly.

Tests that need the xtb binary are marked and skipped when it is absent, so the
suite still runs on a machine without a QM stack.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from molprop.bde import BDEError, bde_from_enthalpies, homolysis_fragments
from molprop.calc import (
    CalculationError,
    CalculationResult,
    CalculationSpec,
    ResourceUsage,
    Task,
    geometry_hash,
)
from molprop.chem import BondRef, indexed_mol
from molprop.conformers import assert_deterministic, generate_conformers
from molprop.engines import XTBEngine
from molprop.fidelity import ConformerProtocol, Fidelity, FidelityError, Tier
from molprop.geometry import (
    Conformer,
    ConformerEnsemble,
    GeometryError,
    check_multiplicity,
    electron_count,
)
from molprop.records import MoleculeRef
from molprop.thermo import (
    ThermoError,
    boltzmann_weights,
    ensemble_quantity,
    monatomic_entropy_kcal_per_mol_k,
    monatomic_thermochemistry,
)
from molprop.units import (
    GAS,
    HARTREE,
    KCAL_PER_MOL,
    Quantity,
    QuantityKind,
    UnitError,
    water,
)

_engine = XTBEngine()
needs_xtb = pytest.mark.skipif(not _engine.is_available(), reason="xtb not installed")


def bond_between(smiles: str, a: str, b: str) -> BondRef:
    mol = indexed_mol(smiles)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pair = sorted(
            (mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol())
        )
        if pair == sorted((a, b)):
            return BondRef(i, j)
    raise AssertionError(f"no {a}-{b} bond in {smiles}")


@pytest.fixture
def protocol() -> ConformerProtocol:
    return ConformerProtocol(
        generator="rdkit_etkdgv3",
        ensemble_treatment="lowest",
        n_confs_requested=10,
        rmsd_prune_angstrom=0.5,
        optimizer="mmff94s",
        seed=42,
    )


@pytest.fixture
def gfn2(protocol) -> Fidelity:
    return Fidelity(tier=Tier.SEMIEMPIRICAL, method="GFN2-xTB", conformers=protocol)


# -- spin bookkeeping ------------------------------------------------------


def test_odd_electron_species_cannot_be_a_singlet():
    with pytest.raises(GeometryError, match="multiplicity 2, not 1"):
        check_multiplicity(electron_count(("C", "H", "H", "H"), 0), 1)


def test_even_electron_species_cannot_be_a_doublet():
    with pytest.raises(GeometryError, match="incompatible"):
        check_multiplicity(electron_count(("C", "H", "H", "H", "H"), 0), 2)


def test_methyl_radical_is_a_doublet():
    check_multiplicity(electron_count(("C", "H", "H", "H"), 0), 2)


def test_charge_shifts_the_electron_count():
    assert electron_count(("O", "H"), -1) == 10


# -- geometry --------------------------------------------------------------


def test_ensemble_rejects_a_wrong_atom_ordering():
    ref = MoleculeRef.from_smiles("CO")
    with pytest.raises(GeometryError, match="element ordering"):
        ConformerEnsemble(
            molecule=ref,
            elements=("O", "C", "H", "H", "H", "H"),
            conformers=[],
            protocol=ConformerProtocol(generator="none", ensemble_treatment="lowest"),
        )


def test_ensemble_rejects_a_wrong_atom_count(protocol):
    ref = MoleculeRef.from_smiles("CO")
    elements = tuple(a.GetSymbol() for a in ref.indexed().GetAtoms())
    with pytest.raises(GeometryError, match="has 2 atoms"):
        ConformerEnsemble(
            molecule=ref,
            elements=elements,
            conformers=[Conformer(coords=np.zeros((2, 3)))],
            protocol=protocol,
        )


def test_ensemble_refuses_mixed_energy_kinds(protocol):
    ref = MoleculeRef.from_smiles("CO")
    elements = tuple(a.GetSymbol() for a in ref.indexed().GetAtoms())
    ff = Quantity(0.0, KCAL_PER_MOL, QuantityKind.FORCE_FIELD_ENERGY, GAS, None)
    qm = Quantity(-1.0, HARTREE, QuantityKind.ELECTRONIC_ENERGY, GAS, None)
    with pytest.raises(UnitError, match="mixes energy kinds"):
        ConformerEnsemble(
            molecule=ref,
            elements=elements,
            conformers=[
                Conformer(coords=np.zeros((6, 3)), energy=ff),
                Conformer(coords=np.zeros((6, 3)), energy=qm),
            ],
            protocol=protocol,
        )


def test_geometry_hash_is_insensitive_to_float_noise():
    coords = np.random.default_rng(0).normal(size=(5, 3))
    a = geometry_hash(coords, ("C", "H", "H", "H", "H"))
    b = geometry_hash(coords + 1e-9, ("C", "H", "H", "H", "H"))
    assert a == b


def test_geometry_hash_notices_a_real_change():
    coords = np.zeros((2, 3))
    moved = coords.copy()
    moved[1, 0] = 0.1
    assert geometry_hash(coords, ("H", "H")) != geometry_hash(moved, ("H", "H"))


# -- conformer generation --------------------------------------------------


def test_force_field_energies_cannot_mix_with_qm(protocol):
    ensemble = generate_conformers("CCO", protocol)
    assert ensemble.energy_kind is QuantityKind.FORCE_FIELD_ENERGY
    qm = Quantity(-1.0, HARTREE, QuantityKind.ELECTRONIC_ENERGY, GAS, None)
    with pytest.raises(UnitError, match="cannot subtract|cannot add"):
        ensemble.conformers[0].energy - qm


def test_generation_is_deterministic(protocol):
    assert_deterministic("OCCCCO", protocol, repeats=3)


def test_a_flexible_molecule_yields_several_conformers():
    protocol = ConformerProtocol(
        generator="rdkit_etkdgv3",
        ensemble_treatment="all",
        n_confs_requested=50,
        rmsd_prune_angstrom=0.5,
        optimizer="mmff94s",
        seed=42,
    )
    assert len(generate_conformers("OCCCCO", protocol)) > 3


def test_all_atom_rmsd_keeps_more_rotamers_than_heavy_atom():
    def count(atoms):
        protocol = ConformerProtocol(
            generator="rdkit_etkdgv3",
            ensemble_treatment="all",
            n_confs_requested=50,
            rmsd_prune_angstrom=0.5,
            rmsd_atoms=atoms,
            optimizer="mmff94s",
            seed=42,
        )
        return len(generate_conformers("OCCCCO", protocol))

    assert count("all") >= count("heavy")


def test_a_lone_atom_is_handled(protocol):
    ensemble = generate_conformers("[H]", protocol, multiplicity=2)
    assert len(ensemble) == 1
    assert ensemble.n_atoms == 1
    assert ensemble.metadata["monatomic"] is True


def test_crest_is_not_silently_substituted():
    protocol = ConformerProtocol(
        generator="crest", ensemble_treatment="all", seed=1
    )
    with pytest.raises(FidelityError, match="rdkit_etkdgv3"):
        generate_conformers("CCO", protocol)


# -- resource accounting ---------------------------------------------------


def test_unpriced_run_reports_no_cost_rather_than_zero():
    assert ResourceUsage(wall_seconds=10.0).cost_usd is None


def test_cost_uses_core_hours():
    usage = ResourceUsage(wall_seconds=3600.0, n_cores=2, usd_per_core_hour=0.10)
    assert usage.core_hours == pytest.approx(2.0)
    assert usage.cost_usd == pytest.approx(0.20)


def test_usage_adds():
    a = ResourceUsage(wall_seconds=1.0, n_cores=2, usd_per_core_hour=0.1)
    b = ResourceUsage(wall_seconds=2.0, n_cores=4, usd_per_core_hour=0.1)
    total = a + b
    assert total.wall_seconds == pytest.approx(3.0)
    assert total.n_cores == 4
    assert total.usd_per_core_hour == 0.1


def test_adding_differently_priced_usage_drops_the_rate():
    a = ResourceUsage(wall_seconds=1.0, usd_per_core_hour=0.1)
    b = ResourceUsage(wall_seconds=1.0, usd_per_core_hour=0.5)
    assert (a + b).cost_usd is None


# -- calculation specs -----------------------------------------------------


def test_spec_identity_includes_the_geometry(gfn2):
    ref = MoleculeRef.from_smiles("CO")
    base = dict(molecule=ref, fidelity=gfn2, task=Task.SINGLE_POINT)
    a = CalculationSpec(geometry_hash="aaaa", **base)
    b = CalculationSpec(geometry_hash="bbbb", **base)
    assert a.id != b.id


def test_spec_identity_includes_the_spin_state(gfn2):
    ref = MoleculeRef.from_smiles("[CH3]")
    base = dict(molecule=ref, fidelity=gfn2, task=Task.SINGLE_POINT, geometry_hash="x")
    assert CalculationSpec(multiplicity=2, **base).id != CalculationSpec(
        multiplicity=4, **base
    ).id


def test_single_point_cannot_carry_thermochemistry(gfn2):
    spec = CalculationSpec(
        molecule=MoleculeRef.from_smiles("CO"),
        fidelity=gfn2,
        task=Task.SINGLE_POINT,
        geometry_hash="x",
    )
    h = Quantity(-1.0, HARTREE, QuantityKind.ENTHALPY, GAS, 298.15)
    with pytest.raises(CalculationError, match="cannot produce H or G"):
        CalculationResult(
            spec=spec,
            converged=True,
            usage=ResourceUsage(1.0),
            engine="x",
            engine_version="1",
            enthalpy=h,
        )


def test_result_rejects_a_phase_mismatch(protocol):
    solvated = Fidelity(
        tier=Tier.SEMIEMPIRICAL,
        method="GFN2-xTB",
        solvation_model="ALPB",
        solvation_solvent="water",
        conformers=protocol,
    )
    spec = CalculationSpec(
        molecule=MoleculeRef.from_smiles("CO"),
        fidelity=solvated,
        task=Task.SINGLE_POINT,
        geometry_hash="x",
    )
    gas_energy = Quantity(-1.0, HARTREE, QuantityKind.ELECTRONIC_ENERGY, GAS, None)
    with pytest.raises(UnitError, match="reported in gas"):
        CalculationResult(
            spec=spec,
            converged=True,
            usage=ResourceUsage(1.0),
            engine="x",
            engine_version="1",
            energy=gas_energy,
        )


# -- thermochemistry -------------------------------------------------------


def test_boltzmann_weights_sum_to_one():
    energies = [
        Quantity(v, KCAL_PER_MOL, QuantityKind.GIBBS, GAS, 298.15) for v in (0.0, 1.0, 2.0)
    ]
    weights = boltzmann_weights(energies)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_a_one_kcal_gap_gives_the_expected_ratio():
    energies = [
        Quantity(v, KCAL_PER_MOL, QuantityKind.GIBBS, GAS, 298.15) for v in (0.0, 1.0)
    ]
    w = boltzmann_weights(energies)
    assert w[0] / w[1] == pytest.approx(math.exp(1.0 / (1.987204259e-3 * 298.15)), rel=1e-6)


def test_weights_refuse_mixed_phases():
    energies = [
        Quantity(0.0, KCAL_PER_MOL, QuantityKind.GIBBS, GAS, 298.15),
        Quantity(1.0, KCAL_PER_MOL, QuantityKind.GIBBS, water(), 298.15),
    ]
    with pytest.raises(UnitError, match="mixes phases"):
        boltzmann_weights(energies)


def test_self_weighting_an_electronic_energy_is_refused():
    from molprop.thermo import boltzmann_average

    energies = [
        Quantity(v, KCAL_PER_MOL, QuantityKind.ELECTRONIC_ENERGY, GAS, 298.15)
        for v in (0.0, 1.0)
    ]
    with pytest.raises(UnitError, match="entropic differences"):
        boltzmann_average(energies)


def test_hydrogen_atom_entropy_matches_the_tabulated_value():
    s = monatomic_entropy_kcal_per_mol_k("H", 298.15, multiplicity=2)
    assert s * 1000 == pytest.approx(27.4, abs=0.1)


def test_monatomic_enthalpy_correction_is_five_halves_rt():
    e = Quantity(-0.5, HARTREE, QuantityKind.ELECTRONIC_ENERGY, GAS, None)
    h, g = monatomic_thermochemistry(e, "H", temperature_K=298.15, multiplicity=2)
    correction = h.to(KCAL_PER_MOL).value - e.to(KCAL_PER_MOL).value
    assert correction == pytest.approx(2.5 * 1.987204259e-3 * 298.15, rel=1e-6)
    assert g.canonical_value < h.canonical_value


def test_monatomic_thermo_needs_an_electronic_energy():
    h = Quantity(-1.0, HARTREE, QuantityKind.ENTHALPY, GAS, 298.15)
    with pytest.raises(UnitError, match="electronic energy"):
        monatomic_thermochemistry(h, "H")


# -- homolysis -------------------------------------------------------------


def test_xh_homolysis_gives_a_radical_and_a_hydrogen_atom():
    a, b = homolysis_fragments("CO", bond_between("CO", "O", "H"))
    smiles = {a.molecule.smiles, b.molecule.smiles}
    assert smiles == {"C[O]", "[H]"}
    assert a.multiplicity == 2 and b.multiplicity == 2


def test_cc_homolysis_of_ethane_gives_two_methyls():
    a, b = homolysis_fragments("CC", bond_between("CC", "C", "C"))
    assert a.molecule.smiles == "[CH3]" and b.molecule.smiles == "[CH3]"


def test_ring_bonds_are_refused():
    with pytest.raises(BDEError, match="in a ring"):
        homolysis_fragments("c1ccccc1", bond_between("c1ccccc1", "C", "C"))


def test_charged_species_are_refused():
    with pytest.raises(BDEError, match="charged species"):
        homolysis_fragments("CC(=O)[O-]", bond_between("CC(=O)[O-]", "C", "H"))


# -- BDE assembly ----------------------------------------------------------


def h(value, phase=GAS, t=298.15):
    return Quantity(value, KCAL_PER_MOL, QuantityKind.ENTHALPY, phase, t)


def test_bde_is_products_minus_parent():
    value = bde_from_enthalpies(h(-100.0), h(-40.0), h(-15.0))
    assert value.kind is QuantityKind.BDE
    assert value.value == pytest.approx(45.0)


def test_bde_refuses_an_electronic_energy():
    e = Quantity(-100.0, KCAL_PER_MOL, QuantityKind.ELECTRONIC_ENERGY, GAS, 298.15)
    with pytest.raises(UnitError, match="must be an enthalpy"):
        bde_from_enthalpies(e, h(-40.0), h(-15.0))


def test_bde_refuses_a_solution_phase_fragment():
    with pytest.raises(UnitError, match="gas-phase"):
        bde_from_enthalpies(h(-100.0), h(-40.0, phase=water()), h(-15.0))


def test_bde_refuses_mismatched_temperatures():
    with pytest.raises(UnitError, match="different temperatures"):
        bde_from_enthalpies(h(-100.0), h(-40.0, t=310.0), h(-15.0))


# -- engine (needs xtb) ----------------------------------------------------


@needs_xtb
def test_engine_reports_a_version():
    assert XTBEngine().version()


@needs_xtb
def test_engine_refuses_a_fidelity_it_cannot_run(protocol):
    dft = Fidelity(
        tier=Tier.DFT, method="M06-2X", basis="def2-TZVP", conformers=protocol
    )
    assert not XTBEngine().supports(dft, Task.SINGLE_POINT)


@needs_xtb
def test_engine_runs_thermochemistry(gfn2, protocol):
    ensemble = generate_conformers("CO", protocol)
    result = XTBEngine(n_cores=2).run(ensemble, gfn2, Task.OPT_FREQ)[0]
    result.require_usable()
    assert result.energy.unit is HARTREE
    assert result.enthalpy.kind is QuantityKind.ENTHALPY
    assert result.gibbs.kind is QuantityKind.GIBBS
    assert result.n_imaginary == 0
    assert result.usage.wall_seconds > 0


@needs_xtb
def test_uhf_flag_carries_unpaired_electrons_not_multiplicity(gfn2, protocol):
    ensemble = generate_conformers("[CH3]", protocol, multiplicity=2)
    result = XTBEngine().run(ensemble, gfn2, Task.SINGLE_POINT)[0]
    assert "--uhf 1" in result.metadata["command"]


@needs_xtb
def test_hydrogen_atom_gets_analytic_thermochemistry(gfn2, protocol):
    ensemble = generate_conformers("[H]", protocol, multiplicity=2)
    result = XTBEngine().run(ensemble, gfn2, Task.OPT_FREQ)[0]
    result.require_usable()
    assert result.enthalpy is not None and result.gibbs is not None
    assert any("monatomic" in w for w in result.warnings)


@needs_xtb
def test_saddle_points_are_refused_without_mode_following(gfn2, protocol):
    """The methoxy radical is a Jahn-Teller case: GFN2 finds a saddle point."""
    ensemble = generate_conformers("C[O]", protocol, multiplicity=2)
    result = XTBEngine(imaginary_retries=0).run(ensemble, gfn2, Task.OPT_FREQ)[0]
    assert result.n_imaginary and result.n_imaginary > 0
    assert result.is_minimum is False
    with pytest.raises(CalculationError, match="imaginary frequencies"):
        result.require_usable()


@needs_xtb
def test_mode_following_escapes_the_saddle_point(gfn2, protocol):
    ensemble = generate_conformers("C[O]", protocol, multiplicity=2)
    result = XTBEngine(imaginary_retries=3).run(ensemble, gfn2, Task.OPT_FREQ)[0]
    assert result.n_imaginary == 0
    result.require_usable()
    assert any("imaginary mode" in w for w in result.warnings)


@needs_xtb
def test_end_to_end_bde(gfn2):
    from molprop.bde import compute_bde

    out = compute_bde(
        "Oc1ccccc1",
        bond_between("Oc1ccccc1", "O", "H"),
        gfn2,
        XTBEngine(n_cores=2, imaginary_retries=3),
    )
    assert out["bde"].kind is QuantityKind.BDE
    # GFN2 overestimates X-H BDEs badly; this only asserts the pipeline is
    # producing a bond energy rather than nonsense.
    assert 50.0 < out["bde"].value < 200.0
    assert out["n_jobs"] == 3
    assert out["usage"].wall_seconds > 0


@needs_xtb
def test_engine_has_a_bounded_default_timeout():
    """A campaign must not be able to stall for an hour on one molecule."""
    assert XTBEngine().timeout_seconds <= 600.0


@needs_xtb
def test_a_timeout_is_reported_as_a_failure_not_a_result(gfn2, protocol):
    ensemble = generate_conformers("CCCCCCO", protocol)
    result = XTBEngine(timeout_seconds=1e-3).run(ensemble, gfn2, Task.OPT_FREQ)[0]
    assert not result.converged
    assert any("timed out" in w for w in result.warnings)
    with pytest.raises(CalculationError):
        result.require_usable()
