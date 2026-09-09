"""Shared fixtures and helpers."""

from __future__ import annotations

import pytest

from molprop.chem import indexed_mol
from molprop.fidelity import ConformerProtocol, Fidelity, Tier
from molprop.records import MoleculeRef, PKaRecord
from molprop.units import LOG10_UNIT, Quantity, QuantityKind, water


@pytest.fixture
def confs() -> ConformerProtocol:
    return ConformerProtocol(
        generator="rdkit_etkdgv3",
        ensemble_treatment="lowest",
        n_confs_requested=50,
        rmsd_prune_angstrom=0.5,
        optimizer="mmff94s",
        seed=42,
    )


@pytest.fixture
def xtb_gas(confs) -> Fidelity:
    return Fidelity(
        tier=Tier.SEMIEMPIRICAL,
        method="GFN2-xTB",
        conformers=confs,
        software="xtb",
        software_version="6.7.1",
    )


@pytest.fixture
def dft_gas(confs) -> Fidelity:
    return Fidelity(
        tier=Tier.DFT,
        method="M06-2X",
        basis="def2-TZVP",
        conformers=confs,
        software="ORCA",
        software_version="6.0",
    )


@pytest.fixture
def dft_water(confs) -> Fidelity:
    return Fidelity(
        tier=Tier.DFT,
        method="wB97X-D",
        basis="def2-TZVPD",
        solvation_model="SMD",
        solvation_solvent="water",
        conformers=confs,
    )


@pytest.fixture
def experiment() -> Fidelity:
    return Fidelity(tier=Tier.EXPERIMENT, method="experiment")


@pytest.fixture
def pka_ladder(dft_water, experiment):
    from molprop.fidelity import FidelityLadder

    return FidelityLadder("pka").add(dft_water, 5).add(experiment, 10)


def find_atom(canonical: str, symbol: str, *, with_h: bool = False) -> int:
    """Index of the first atom of `symbol`, optionally requiring a bonded H."""
    mol = indexed_mol(canonical)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != symbol:
            continue
        if with_h and not any(n.GetSymbol() == "H" for n in atom.GetNeighbors()):
            continue
        return atom.GetIdx()
    raise AssertionError(f"no {symbol} atom in {canonical}")


def macro_record(smiles: str, value: float, fidelity, *, step: int = 1, **kw) -> PKaRecord:
    """A macroscopic pKa record, the shape most of the corpus takes."""
    return PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=Quantity(value, LOG10_UNIT, QuantityKind.PKA, water(0.15), 298.15),
        fidelity=fidelity,
        source="test",
        resolution="macro",
        step_index=step,
        **kw,
    )


def micro_record(
    smiles: str, value: float, fidelity, *, protonated: str, deprotonated: str, **kw
) -> PKaRecord:
    """A microscopic pKa record naming the site that loses the proton."""
    return PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=Quantity(value, LOG10_UNIT, QuantityKind.PKA, water(0.15), 298.15),
        fidelity=fidelity,
        source="test",
        resolution="micro",
        site_atom_idx=find_atom(smiles, "O", with_h=True),
        protonated_smiles=protonated,
        deprotonated_smiles=deprotonated,
        **kw,
    )
