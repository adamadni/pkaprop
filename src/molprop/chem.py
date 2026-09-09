"""RDKit helpers with deterministic atom indexing.

Atom-index stability matters here more than usual: a BDE record names a bond by
atom indices and a microscopic pKa names a site by atom index. Those indices are
meaningless unless the mapping from an identity string to an atom ordering is
fixed. The convention used everywhere in this package:

    identity   = canonical isomeric SMILES (RDKit, no explicit hydrogens)
    heavy mol  = Chem.MolFromSmiles(identity)
    indexed mol = Chem.AddHs(heavy mol)

`AddHs` appends hydrogens after the heavy atoms in heavy-atom order, so indices
into the indexed mol are a deterministic function of the identity string alone.
Every atom index stored in a record is an index into that indexed mol.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

__all__ = [
    "MoleculeError",
    "canonical_smiles",
    "inchikey",
    "heavy_mol",
    "indexed_mol",
    "bond_label",
    "molecular_formula",
    "net_charge",
    "murcko_scaffold",
    "atom_environment",
    "BondRef",
]

RDLogger.DisableLog("rdApp.*")


class MoleculeError(ValueError):
    """Raised when a structure cannot be parsed or violates an invariant."""


def _parse(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeError(f"RDKit could not parse SMILES {smiles!r}")
    return mol


def canonical_smiles(smiles: str) -> str:
    """Canonical isomeric SMILES. This string is the molecule's identity."""
    return Chem.MolToSmiles(_parse(smiles), isomericSmiles=True, canonical=True)


def inchikey(smiles: str) -> str:
    key = Chem.MolToInchiKey(_parse(smiles))
    if not key:
        raise MoleculeError(f"could not generate InChIKey for {smiles!r}")
    return key


def heavy_mol(canonical: str) -> Chem.Mol:
    """Heavy-atom mol parsed from a canonical SMILES."""
    return _parse(canonical)


@functools.lru_cache(maxsize=100_000)
def _indexed_mol_cached(canonical: str) -> Chem.Mol:
    return Chem.AddHs(_parse(canonical))


def indexed_mol(canonical: str) -> Chem.Mol:
    """Explicit-hydrogen mol defining the atom indexing used by all records.

    Parsing dominates bulk import (a BDE dataset has ~11 bonds per molecule, so
    the same structure is rebuilt many times). The parse is cached and a fresh
    copy handed out, so callers may still mutate what they get.
    """
    return Chem.Mol(_indexed_mol_cached(canonical))


def bond_label(mol: Chem.Mol, atom_i: int, atom_j: int) -> str:
    """Element-pair label for a bond, e.g. 'C-H', 'N-H', 'C-O'.

    Sorted so the label does not depend on which index was given first.
    """
    n = mol.GetNumAtoms()
    for idx in (atom_i, atom_j):
        if not 0 <= idx < n:
            raise MoleculeError(f"atom index {idx} out of range for {n}-atom molecule")
    if mol.GetBondBetweenAtoms(atom_i, atom_j) is None:
        raise MoleculeError(f"no bond between atoms {atom_i} and {atom_j}")
    a = mol.GetAtomWithIdx(atom_i).GetSymbol()
    b = mol.GetAtomWithIdx(atom_j).GetSymbol()
    return "-".join(sorted((a, b)))


def molecular_formula(smiles: str) -> str:
    return rdMolDescriptors.CalcMolFormula(_parse(smiles))


def net_charge(smiles: str) -> int:
    return Chem.GetFormalCharge(_parse(smiles))


@functools.lru_cache(maxsize=200_000)
def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; empty string for acyclic molecules.

    Cached: a split calls this once per record, but a BDE dataset has ~11
    records per molecule.
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(_parse(smiles))
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else ""


def atom_environment(canonical: str, atom_idx: int, radius: int = 2) -> str:
    """Canonical SMILES of the substructure within `radius` bonds of an atom.

    This is the grouping key for a reaction-centre split. For a local property
    like pKa, the holdout unit that answers "does this generalise to new
    chemistry" is the environment of the ionisable site -- not the scaffold,
    which a molecule can change entirely while keeping the same aniline.
    """
    mol = indexed_mol(canonical)
    if not 0 <= atom_idx < mol.GetNumAtoms():
        raise MoleculeError(
            f"atom index {atom_idx} out of range for {mol.GetNumAtoms()}-atom molecule"
        )
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    if not env:
        # Isolated atom at this radius: fall back to the atom itself.
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()
    amap: dict[int, int] = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    if atom_idx not in amap:
        return mol.GetAtomWithIdx(atom_idx).GetSymbol()
    return Chem.MolToSmiles(submol, rootedAtAtom=amap[atom_idx], canonical=True)


@dataclass(frozen=True)
class BondRef:
    """A bond named by atom indices into the indexed (explicit-H) mol."""

    atom_i: int
    atom_j: int

    def __post_init__(self) -> None:
        if self.atom_i == self.atom_j:
            raise MoleculeError("a bond needs two distinct atoms")
        if self.atom_i < 0 or self.atom_j < 0:
            raise MoleculeError("atom indices must be non-negative")
        # Store in a canonical order so (i,j) and (j,i) are the same bond.
        if self.atom_i > self.atom_j:
            lo, hi = self.atom_j, self.atom_i
            object.__setattr__(self, "atom_i", lo)
            object.__setattr__(self, "atom_j", hi)

    def as_tuple(self) -> tuple[int, int]:
        return (self.atom_i, self.atom_j)
