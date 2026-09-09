"""Shared fixtures. Kept small and explicit."""

from __future__ import annotations

import pytest

from molprop.chem import BondRef, indexed_mol
from molprop.fidelity import ConformerProtocol, Fidelity, FidelityLadder, Tier


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
def bde_ladder(xtb_gas, dft_gas) -> FidelityLadder:
    return FidelityLadder("bde").add(xtb_gas, 0).add(dft_gas, 10)


@pytest.fixture
def pka_ladder(dft_water, experiment) -> FidelityLadder:
    return FidelityLadder("pka").add(dft_water, 5).add(experiment, 10)


def find_bond(canonical: str, sym_a: str, sym_b: str) -> BondRef:
    """First bond in the indexed mol joining the two given elements."""
    mol = indexed_mol(canonical)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pair = sorted((mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()))
        if pair == sorted((sym_a, sym_b)):
            return BondRef(i, j)
    raise AssertionError(f"no {sym_a}-{sym_b} bond in {canonical}")


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
