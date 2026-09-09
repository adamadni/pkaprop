"""Property records.

One record is one measured-or-computed number, with the molecule it belongs to,
the fidelity that produced it, and enough structure to be split without leakage.

Every record carries a `group_key` -- the unit that must never be broken across
a train/test boundary. It is the molecule, because one molecule contributes a
record per ionisable site and scattering those across folds leaks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from .chem import (
    MoleculeError,
    canonical_smiles,
    inchikey,
    indexed_mol,
    net_charge,
)
from .fidelity import Fidelity
from .units import Quantity, QuantityKind, UnitError

__all__ = ["MoleculeRef", "PropertyRecord", "PKaRecord"]


def _stable_id(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class MoleculeRef:
    """Molecular identity. `smiles` is canonical and defines atom indexing."""

    smiles: str
    inchikey: str

    @classmethod
    def from_smiles(cls, smiles: str) -> "MoleculeRef":
        canon = canonical_smiles(smiles)
        return cls(smiles=canon, inchikey=inchikey(canon))

    def indexed(self):
        """Explicit-hydrogen mol that record atom indices refer to."""
        return indexed_mol(self.smiles)

    @property
    def formula(self) -> str:
        return molecular_formula(self.smiles)

    @property
    def charge(self) -> int:
        return net_charge(self.smiles)


@dataclass(frozen=True)
class PropertyRecord:
    """Base record. Subclasses add the structure specific to a property."""

    molecule: MoleculeRef
    quantity: Quantity
    fidelity: Fidelity
    source: str
    source_ref: str | None = None
    uncertainty: Quantity | None = None
    #: ISO-8601 date the value was published/measured; required for time splits.
    date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    #: kinds a subclass will accept; None means "any". ClassVar so it stays out
    #: of __init__ and out of serialisation.
    _allowed_kinds: ClassVar[tuple[QuantityKind, ...] | None] = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("every record must name its source dataset")
        allowed = self._allowed_kinds
        if allowed and self.quantity.kind not in allowed:
            raise UnitError(
                f"{type(self).__name__} requires a quantity of kind "
                f"{[k.value for k in allowed]}, got {self.quantity.kind.value}"
            )
        if self.uncertainty is not None:
            if self.uncertainty.kind.dimension is not self.quantity.kind.dimension:
                raise UnitError(
                    "uncertainty must share the dimension of the value it qualifies"
                )
            if self.uncertainty.canonical_value < 0:
                raise UnitError("uncertainty must be non-negative")
        # A solution-phase fidelity must not carry a gas-phase value, and the
        # reverse. This is the ensemble-mismatch guard at record level.
        if self.fidelity.is_solution_phase and self.quantity.phase.medium != "solution":
            raise UnitError(
                f"fidelity {self.fidelity.label()} is solution-phase but the value "
                f"is reported in {self.quantity.phase}"
            )
        if not self.fidelity.is_solution_phase and self.quantity.phase.medium != "gas":
            raise UnitError(
                f"fidelity {self.fidelity.label()} is gas-phase but the value "
                f"is reported in {self.quantity.phase}"
            )

    @property
    def group_key(self) -> str:
        """Split-integrity unit. Overridden only with a very good reason."""
        return self.molecule.inchikey

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "smiles": self.molecule.smiles,
            "fidelity": self.fidelity.id,
            "source": self.source,
            "kind": self.quantity.kind.value,
        }

    @property
    def record_id(self) -> str:
        return _stable_id(self._identity_payload())

    def as_dict(self) -> dict[str, Any]:
        d = {
            "record_type": type(self).__name__,
            "record_id": self.record_id,
            "molecule": {"smiles": self.molecule.smiles, "inchikey": self.molecule.inchikey},
            "quantity": self.quantity.as_dict(),
            "fidelity": self.fidelity.as_dict(),
            "source": self.source,
            "source_ref": self.source_ref,
            "uncertainty": self.uncertainty.as_dict() if self.uncertainty else None,
            "date": self.date,
            "group_key": self.group_key,
            "metadata": self.metadata,
        }
        d.update(self._extra_dict())
        return d

    def _extra_dict(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class PKaRecord(PropertyRecord):
    """An acid dissociation constant, macroscopic or microscopic.

    Both are always written as a *deprotonation*: the pKa of a base is the pKa
    of its conjugate acid, so the direction is uniform and no sign convention
    has to be remembered.
    """

    resolution: Literal["macro", "micro"] | None = None
    #: macro only -- 1 for the first proton lost, 2 for the second, ...
    step_index: int | None = None
    #: micro only -- index into the indexed mol of the atom losing the proton
    site_atom_idx: int | None = None
    #: micro only -- the two microstates, canonical SMILES with formal charges
    protonated_smiles: str | None = None
    deprotonated_smiles: str | None = None

    _allowed_kinds: ClassVar[tuple[QuantityKind, ...] | None] = (QuantityKind.PKA,)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.resolution not in ("macro", "micro"):
            raise ValueError(
                "pKa records must declare resolution='macro' or 'micro'; "
                "experimental labels are macroscopic and site models are microscopic, "
                "and pooling them without aggregation is a silent target mismatch"
            )
        if self.quantity.phase.medium != "solution":
            raise UnitError("a pKa is a solution-phase quantity")

        if self.resolution == "macro":
            if self.step_index is None or self.step_index < 1:
                raise ValueError("a macroscopic pKa needs a step_index >= 1")
            if self.site_atom_idx is not None:
                raise ValueError(
                    "a macroscopic pKa cannot name a site; that is a microscopic claim"
                )
            if self.protonated_smiles or self.deprotonated_smiles:
                raise ValueError("a macroscopic pKa cannot name microstates")
        else:
            if self.site_atom_idx is None:
                raise ValueError("a microscopic pKa needs the site_atom_idx")
            if self.step_index is not None:
                raise ValueError("a microscopic pKa has no step index")
            if not (self.protonated_smiles and self.deprotonated_smiles):
                raise ValueError(
                    "a microscopic pKa needs both microstate SMILES; the site index "
                    "alone does not identify the equilibrium"
                )
            mol = self.molecule.indexed()
            if not 0 <= self.site_atom_idx < mol.GetNumAtoms():
                raise MoleculeError(
                    f"site_atom_idx {self.site_atom_idx} out of range for "
                    f"{mol.GetNumAtoms()}-atom indexed molecule"
                )
            prot = canonical_smiles(self.protonated_smiles)
            deprot = canonical_smiles(self.deprotonated_smiles)
            object.__setattr__(self, "protonated_smiles", prot)
            object.__setattr__(self, "deprotonated_smiles", deprot)
            if net_charge(deprot) != net_charge(prot) - 1:
                raise MoleculeError(
                    f"deprotonation must lower the formal charge by exactly 1: "
                    f"{prot} (q={net_charge(prot)}) -> {deprot} (q={net_charge(deprot)})"
                )

    def _identity_payload(self) -> dict[str, Any]:
        p = super()._identity_payload()
        p["resolution"] = self.resolution
        p["step_index"] = self.step_index
        p["site"] = self.site_atom_idx
        return p

    def _extra_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "step_index": self.step_index,
            "site_atom_idx": self.site_atom_idx,
            "protonated_smiles": self.protonated_smiles,
            "deprotonated_smiles": self.deprotonated_smiles,
        }
