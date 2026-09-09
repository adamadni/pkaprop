""""DFT" is not a fidelity label, and the conformer protocol is part of it."""

from __future__ import annotations

import dataclasses

import pytest

from molprop.fidelity import (
    ConformerProtocol,
    DeltaSpec,
    Fidelity,
    FidelityError,
    FidelityLadder,
    Tier,
)


def test_dft_without_a_basis_is_rejected(confs):
    with pytest.raises(FidelityError, match="not a fidelity label"):
        Fidelity(tier=Tier.DFT, method="B3LYP", conformers=confs)


def test_qm_tier_requires_a_conformer_protocol():
    with pytest.raises(FidelityError, match="conformer protocol"):
        Fidelity(tier=Tier.DFT, method="B3LYP", basis="6-31G*")


def test_experiment_cannot_carry_a_level_of_theory():
    with pytest.raises(FidelityError, match="level of theory"):
        Fidelity(tier=Tier.EXPERIMENT, method="experiment", basis="def2-TZVP")


def test_solvation_model_and_solvent_travel_together(confs):
    with pytest.raises(FidelityError, match="together"):
        Fidelity(
            tier=Tier.DFT,
            method="wB97X-D",
            basis="def2-TZVPD",
            solvation_model="SMD",
            conformers=confs,
        )


def test_stochastic_conformer_generation_requires_a_seed():
    with pytest.raises(FidelityError, match="explicit seed"):
        ConformerProtocol(generator="rdkit_etkdgv3", ensemble_treatment="lowest")


def test_boltzmann_averaging_requires_an_energy_window():
    with pytest.raises(FidelityError, match="energy_window"):
        ConformerProtocol(
            generator="crest", ensemble_treatment="boltzmann", seed=1
        )


def test_id_changes_with_the_conformer_protocol(dft_gas, confs):
    lowest = dft_gas
    ensemble = dataclasses.replace(
        dft_gas,
        conformers=dataclasses.replace(
            confs, ensemble_treatment="boltzmann", energy_window_kcal_per_mol=6.0
        ),
    )
    assert lowest.id != ensemble.id, (
        "single-lowest and Boltzmann-averaged are different fidelities"
    )


def test_id_is_stable_across_construction(dft_gas):
    twin = Fidelity(
        tier=Tier.DFT,
        method="M06-2X",
        basis="def2-TZVP",
        conformers=dft_gas.conformers,
        software="ORCA",
        software_version="6.0",
    )
    assert twin.id == dft_gas.id


def test_free_text_notes_do_not_change_identity(dft_gas):
    assert dataclasses.replace(dft_gas, notes="rerun on new nodes").id == dft_gas.id


def test_geometry_source_is_part_of_the_identity(dft_gas, xtb_gas):
    single_point = dataclasses.replace(dft_gas, geometry_source=xtb_gas.id)
    assert single_point.id != dft_gas.id


def test_label_reads_like_a_method_string(dft_water):
    assert dft_water.label() == "wB97X-D/def2-TZVPD/SMD(water)"


def test_ladder_rejects_duplicate_ranks(xtb_gas, dft_gas):
    ladder = FidelityLadder("bde").add(xtb_gas, 0)
    with pytest.raises(FidelityError, match="rank 0 already taken"):
        ladder.add(dft_gas, 0)


def test_ladder_reference_is_the_top_rung(bde_ladder, dft_gas):
    assert bde_ladder.reference.id == dft_gas.id


def test_unregistered_fidelity_cannot_be_ranked(bde_ladder, dft_water):
    with pytest.raises(FidelityError, match="not registered"):
        bde_ladder.rank(dft_water)


def test_delta_target_must_outrank_the_base(bde_ladder, xtb_gas, dft_gas):
    DeltaSpec(base=xtb_gas, target=dft_gas, ladder=bde_ladder)
    with pytest.raises(FidelityError, match="must outrank"):
        DeltaSpec(base=dft_gas, target=xtb_gas, ladder=bde_ladder)


def test_pka_delta_targets_experiment(pka_ladder, dft_water, experiment):
    spec = DeltaSpec(base=dft_water, target=experiment, ladder=pka_ladder)
    assert spec.target.tier is Tier.EXPERIMENT


def test_delta_across_phases_is_refused(confs, xtb_gas, dft_water):
    ladder = FidelityLadder("mixed").add(xtb_gas, 0).add(dft_water, 10)
    with pytest.raises(FidelityError, match="solvation free energy"):
        DeltaSpec(base=xtb_gas, target=dft_water, ladder=ladder)


def test_round_trip_through_dict(dft_water):
    assert Fidelity.from_dict(dft_water.as_dict()).id == dft_water.id
