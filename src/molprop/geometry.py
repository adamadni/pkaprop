"""3D structures, tied to the package's atom-indexing convention.

A conformer's coordinate array is ordered to match `chem.indexed_mol` for the
molecule's canonical SMILES. That is what makes a BDE record's bond indices and
a microscopic pKa's site index meaningful against a geometry: the same integer
refers to the same atom everywhere in the pipeline.

Charge and multiplicity travel with the geometry rather than being inferred.
Radical fragments are the whole point of the BDE track, and a fragment silently
treated as closed-shell is the classic way a BDE pipeline produces plausible,
wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from .chem import MoleculeError, indexed_mol
from .fidelity import ConformerProtocol
from .records import MoleculeRef
from .units import GAS, KCAL_PER_MOL, Phase, Quantity, QuantityKind, UnitError

__all__ = [
    "GeometryError",
    "Conformer",
    "ConformerEnsemble",
    "electron_count",
    "check_multiplicity",
]

#: Atomic numbers by element symbol, for the elements this project touches.
_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
    "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


class GeometryError(ValueError):
    """Raised on a malformed geometry or an impossible charge/spin state."""


def electron_count(elements: Sequence[str], charge: int) -> int:
    """Total electrons for a set of element symbols at a given charge."""
    total = 0
    for symbol in elements:
        try:
            total += _Z[symbol]
        except KeyError:
            raise GeometryError(
                f"element {symbol!r} has no atomic number in this table; add it"
            ) from None
    return total - charge


def check_multiplicity(n_electrons: int, multiplicity: int) -> None:
    """Raise unless the spin multiplicity is possible for this electron count.

    An even-electron species cannot be a doublet and an odd-electron species
    cannot be a singlet. This catches the single most common BDE pipeline bug:
    generating a radical fragment and leaving multiplicity at 1.
    """
    if multiplicity < 1:
        raise GeometryError(f"multiplicity must be >= 1, got {multiplicity}")
    n_unpaired = multiplicity - 1
    if n_unpaired > n_electrons:
        raise GeometryError(
            f"multiplicity {multiplicity} needs {n_unpaired} unpaired electrons "
            f"but the species has only {n_electrons}"
        )
    if (n_electrons % 2) != (n_unpaired % 2):
        parity = "odd" if n_electrons % 2 else "even"
        raise GeometryError(
            f"{n_electrons} electrons is {parity}, which is incompatible with "
            f"multiplicity {multiplicity}; an open-shell fragment needs "
            f"multiplicity 2, not 1"
        )


@dataclass
class Conformer:
    """One 3D structure with an optional energy of a stated kind."""

    coords: np.ndarray  # (n_atoms, 3), angstrom
    energy: Quantity | None = None
    converged: bool = True
    label: str | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.coords, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise GeometryError(f"coords must be (n_atoms, 3), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise GeometryError("coords contain non-finite values")
        arr.flags.writeable = False
        self.coords = arr
        if self.energy is not None and self.energy.kind.dimension.value != "molar_energy":
            raise UnitError(
                f"a conformer energy must be a molar energy, got {self.energy.kind.value}"
            )

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])


@dataclass
class ConformerEnsemble:
    """An ordered set of conformers for one species at one charge and spin.

    `elements` is the atom ordering every coordinate array and every record
    atom index refers to; it is checked against the molecule's indexed mol on
    construction so the two can never drift apart.
    """

    molecule: MoleculeRef
    elements: tuple[str, ...]
    conformers: list[Conformer]
    protocol: ConformerProtocol
    charge: int = 0
    multiplicity: int = 1
    phase: Phase = GAS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = tuple(a.GetSymbol() for a in indexed_mol(self.molecule.smiles).GetAtoms())
        if tuple(self.elements) != expected:
            raise GeometryError(
                "element ordering does not match the molecule's indexed mol; "
                "coordinates must follow Chem.AddHs(Chem.MolFromSmiles(canonical)) "
                f"order.\n  expected: {expected}\n  got:      {tuple(self.elements)}"
            )
        for i, conf in enumerate(self.conformers):
            if conf.n_atoms != len(self.elements):
                raise GeometryError(
                    f"conformer {i} has {conf.n_atoms} atoms, expected "
                    f"{len(self.elements)}"
                )
        check_multiplicity(electron_count(self.elements, self.charge), self.multiplicity)
        kinds = {c.energy.kind for c in self.conformers if c.energy is not None}
        if len(kinds) > 1:
            raise UnitError(
                f"ensemble mixes energy kinds {sorted(k.value for k in kinds)}; "
                f"a force-field energy and a semiempirical one are not comparable"
            )

    # -- basics ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.conformers)

    def __iter__(self) -> Iterator[Conformer]:
        return iter(self.conformers)

    def __getitem__(self, i: int) -> Conformer:
        return self.conformers[i]

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def is_open_shell(self) -> bool:
        return self.multiplicity > 1

    @property
    def energy_kind(self) -> QuantityKind | None:
        for c in self.conformers:
            if c.energy is not None:
                return c.energy.kind
        return None

    def energies(self, unit=KCAL_PER_MOL) -> list[Quantity]:
        missing = [i for i, c in enumerate(self.conformers) if c.energy is None]
        if missing:
            raise UnitError(
                f"conformers {missing[:5]} have no energy; an ensemble cannot be "
                f"ranked or Boltzmann-weighted until every member is evaluated"
            )
        return [c.energy.to(unit) for c in self.conformers]

    def sorted_by_energy(self) -> "ConformerEnsemble":
        ordered = sorted(self.conformers, key=lambda c: c.energy.canonical_value)
        return ConformerEnsemble(
            molecule=self.molecule,
            elements=self.elements,
            conformers=list(ordered),
            protocol=self.protocol,
            charge=self.charge,
            multiplicity=self.multiplicity,
            phase=self.phase,
            metadata=dict(self.metadata),
        )

    def lowest(self) -> Conformer:
        if not self.conformers:
            raise GeometryError("ensemble is empty")
        self.energies()  # raises if any energy is missing
        return min(self.conformers, key=lambda c: c.energy.canonical_value)

    def relative_energies(self, unit=KCAL_PER_MOL) -> list[Quantity]:
        """Energies relative to the lowest conformer, as delta quantities."""
        lowest = self.lowest().energy
        return [c.energy - lowest for c in self.conformers]

    def within_window(self, window_kcal_per_mol: float) -> "ConformerEnsemble":
        if window_kcal_per_mol < 0:
            raise GeometryError("energy window must be non-negative")
        rel = self.relative_energies()
        keep = [
            c for c, d in zip(self.conformers, rel) if d.canonical_value <= window_kcal_per_mol
        ]
        return ConformerEnsemble(
            molecule=self.molecule,
            elements=self.elements,
            conformers=keep,
            protocol=self.protocol,
            charge=self.charge,
            multiplicity=self.multiplicity,
            phase=self.phase,
            metadata=dict(self.metadata),
        )

    def with_energies(
        self, energies: Sequence[Quantity], *, converged: Sequence[bool] | None = None
    ) -> "ConformerEnsemble":
        """Return a copy carrying new energies, e.g. after a single-point stage."""
        if len(energies) != len(self.conformers):
            raise GeometryError(
                f"got {len(energies)} energies for {len(self.conformers)} conformers"
            )
        flags = list(converged) if converged is not None else [c.converged for c in self.conformers]
        new = [
            Conformer(coords=c.coords, energy=e, converged=f, label=c.label)
            for c, e, f in zip(self.conformers, energies, flags)
        ]
        return ConformerEnsemble(
            molecule=self.molecule,
            elements=self.elements,
            conformers=new,
            protocol=self.protocol,
            charge=self.charge,
            multiplicity=self.multiplicity,
            phase=self.phase,
            metadata=dict(self.metadata),
        )

    # -- serialisation -----------------------------------------------------

    def to_xyz(self, index: int = 0, comment: str | None = None) -> str:
        conf = self.conformers[index]
        header = comment
        if header is None:
            energy = conf.energy
            header = (
                f"{self.molecule.smiles} charge={self.charge} mult={self.multiplicity}"
                + (f" E={energy.value:.10f} {energy.unit}" if energy else "")
            )
        lines = [str(self.n_atoms), header]
        for symbol, (x, y, z) in zip(self.elements, conf.coords):
            lines.append(f"{symbol:<3s} {x:>18.10f} {y:>18.10f} {z:>18.10f}")
        return "\n".join(lines) + "\n"

    def all_to_xyz(self) -> str:
        return "".join(self.to_xyz(i) for i in range(len(self.conformers)))


def ensemble_from_xyz_block(
    molecule: MoleculeRef,
    xyz: str,
    protocol: ConformerProtocol,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> ConformerEnsemble:
    """Parse a (possibly multi-frame) XYZ block into an ensemble.

    Element ordering is validated against the molecule, so an engine that
    reorders atoms cannot silently corrupt the index convention.
    """
    lines = xyz.splitlines()
    conformers: list[Conformer] = []
    elements: tuple[str, ...] | None = None
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].strip())
        except ValueError:
            raise MoleculeError(f"expected an atom count at line {i + 1}") from None
        block = lines[i + 2 : i + 2 + n]
        if len(block) != n:
            raise MoleculeError(f"truncated XYZ frame at line {i + 1}")
        syms, coords = [], []
        for row in block:
            parts = row.split()
            syms.append(parts[0])
            coords.append([float(v) for v in parts[1:4]])
        if elements is None:
            elements = tuple(syms)
        elif tuple(syms) != elements:
            raise GeometryError("frames in this XYZ block have different atom orderings")
        conformers.append(Conformer(coords=np.array(coords)))
        i += 2 + n
    if elements is None:
        raise MoleculeError("no frames found in XYZ block")
    return ConformerEnsemble(
        molecule=molecule,
        elements=elements,
        conformers=conformers,
        protocol=protocol,
        charge=charge,
        multiplicity=multiplicity,
    )
