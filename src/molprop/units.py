"""Typed physical quantities.

Project constraint: every property value carries explicit units, and a unit,
kind, phase or temperature mismatch raises rather than silently computing.

The three silent-error classes this module exists to prevent:

1. Unit confusion     -- kcal/mol vs kJ/mol vs hartree vs eV.
2. Kind confusion     -- electronic energy E used where enthalpy H or Gibbs G
                         was meant. These differ by thermal/entropic terms and
                         are numerically plausible as each other.
3. Ensemble confusion -- a gas-phase value combined with a solution-phase one,
                         or values at different temperatures.

There is exactly one place in the codebase where a pKa or a log tautomer ratio
becomes a free energy: `log10_to_molar_energy` / `molar_energy_to_log10`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

__all__ = [
    "UnitError",
    "Dimension",
    "Unit",
    "QuantityKind",
    "Phase",
    "GAS",
    "water",
    "Quantity",
    "GAS_CONSTANT_KCAL_PER_MOL_K",
    "STANDARD_TEMPERATURE_K",
    "rt_ln10_kcal_per_mol",
    "log10_to_molar_energy",
    "molar_energy_to_log10",
    "unit_from_symbol",
    "KCAL_PER_MOL",
    "KJ_PER_MOL",
    "J_PER_MOL",
    "HARTREE",
    "EV",
    "LOG10_UNIT",
    "DIMENSIONLESS",
]


class UnitError(ValueError):
    """Raised on any unit, kind, phase or temperature incompatibility."""


# --------------------------------------------------------------------------
# Physical constants (CODATA 2018). Defined once; never re-typed inline.
# --------------------------------------------------------------------------

#: Molar gas constant in kcal/(mol*K).  8.314462618 J/(mol*K) / 4184 J/kcal.
GAS_CONSTANT_KCAL_PER_MOL_K = 8.314462618 / 4184.0

#: Thermochemical standard temperature.
STANDARD_TEMPERATURE_K = 298.15

_LN10 = math.log(10.0)


def rt_ln10_kcal_per_mol(temperature_K: float = STANDARD_TEMPERATURE_K) -> float:
    """RT*ln(10) in kcal/mol -- the value of one log10 unit.

    At 298.15 K this is 1.3642 kcal/mol. One pKa unit and one log tautomer
    ratio unit are both worth exactly this much free energy.
    """
    if temperature_K <= 0:
        raise UnitError(f"temperature must be positive, got {temperature_K} K")
    return GAS_CONSTANT_KCAL_PER_MOL_K * temperature_K * _LN10


class Dimension(Enum):
    MOLAR_ENERGY = "molar_energy"
    LOG10 = "log10"
    DIMENSIONLESS = "dimensionless"


@dataclass(frozen=True)
class Unit:
    symbol: str
    dimension: Dimension
    #: multiply a value in this unit by this to reach the canonical unit
    to_canonical: float

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.symbol


# Canonical molar-energy unit is kcal/mol. Canonical log unit is log10.
KCAL_PER_MOL = Unit("kcal/mol", Dimension.MOLAR_ENERGY, 1.0)
KJ_PER_MOL = Unit("kJ/mol", Dimension.MOLAR_ENERGY, 1.0 / 4.184)
J_PER_MOL = Unit("J/mol", Dimension.MOLAR_ENERGY, 1.0 / 4184.0)
HARTREE = Unit("hartree", Dimension.MOLAR_ENERGY, 627.5094740631)
EV = Unit("eV", Dimension.MOLAR_ENERGY, 23.060547830619026)
LOG10_UNIT = Unit("log10", Dimension.LOG10, 1.0)
DIMENSIONLESS = Unit("1", Dimension.DIMENSIONLESS, 1.0)

_UNITS_BY_SYMBOL: dict[str, Unit] = {
    u.symbol: u
    for u in (KCAL_PER_MOL, KJ_PER_MOL, J_PER_MOL, HARTREE, EV, LOG10_UNIT, DIMENSIONLESS)
}
# Aliases people actually type. Deliberately narrow -- an unrecognised symbol
# must raise, not fall back to a guess.
_UNITS_BY_SYMBOL.update(
    {
        "kcal_per_mol": KCAL_PER_MOL,
        "kcal/mole": KCAL_PER_MOL,
        "kJ_per_mol": KJ_PER_MOL,
        "Eh": HARTREE,
        "hartrees": HARTREE,
        "pK": LOG10_UNIT,
        "log_units": LOG10_UNIT,
    }
)


def unit_from_symbol(symbol: str) -> Unit:
    try:
        return _UNITS_BY_SYMBOL[symbol]
    except KeyError:
        known = sorted({u.symbol for u in _UNITS_BY_SYMBOL.values()})
        raise UnitError(f"unknown unit {symbol!r}; known units: {known}") from None


class QuantityKind(Enum):
    """What thermodynamic object a number *is*.

    ELECTRONIC_ENERGY (E), ENTHALPY (H) and GIBBS (G) are mutually
    non-substitutable. The delta variants are what subtraction produces; they
    are tracked separately so a reaction energy can never be mistaken for an
    absolute one.
    """

    ELECTRONIC_ENERGY = "E"
    #: A force-field energy. Deliberately NOT the same kind as an electronic
    #: energy: MMFF and GFN2-xTB numbers are on unrelated scales and adding or
    #: differencing them across kinds is meaningless.
    FORCE_FIELD_ENERGY = "E_ff"
    ENTHALPY = "H"
    GIBBS = "G"
    ZERO_POINT_ENERGY = "ZPE"

    DELTA_ELECTRONIC_ENERGY = "dE"
    DELTA_FORCE_FIELD_ENERGY = "dE_ff"
    DELTA_ENTHALPY = "dH"
    DELTA_GIBBS = "dG"

    #: Homolytic bond dissociation enthalpy. A reaction enthalpy, but given its
    #: own kind so it cannot be silently added to an arbitrary dH.
    BDE = "BDE"

    #: -log10(Ka) for a single deprotonation. Macro vs micro is a property of
    #: the *record*, not of the quantity -- see records.PKaRecord.
    PKA = "pKa"

    #: log10([B]/[A]) for a tautomer pair; sign convention fixed by the record.
    LOG_TAUTOMER_RATIO = "logKt"

    @property
    def dimension(self) -> Dimension:
        return _KIND_DIMENSION[self]

    @property
    def is_delta(self) -> bool:
        return self in _DELTA_KINDS


_KIND_DIMENSION = {
    QuantityKind.ELECTRONIC_ENERGY: Dimension.MOLAR_ENERGY,
    QuantityKind.FORCE_FIELD_ENERGY: Dimension.MOLAR_ENERGY,
    QuantityKind.ENTHALPY: Dimension.MOLAR_ENERGY,
    QuantityKind.GIBBS: Dimension.MOLAR_ENERGY,
    QuantityKind.ZERO_POINT_ENERGY: Dimension.MOLAR_ENERGY,
    QuantityKind.DELTA_ELECTRONIC_ENERGY: Dimension.MOLAR_ENERGY,
    QuantityKind.DELTA_FORCE_FIELD_ENERGY: Dimension.MOLAR_ENERGY,
    QuantityKind.DELTA_ENTHALPY: Dimension.MOLAR_ENERGY,
    QuantityKind.DELTA_GIBBS: Dimension.MOLAR_ENERGY,
    QuantityKind.BDE: Dimension.MOLAR_ENERGY,
    QuantityKind.PKA: Dimension.LOG10,
    QuantityKind.LOG_TAUTOMER_RATIO: Dimension.LOG10,
}

_DELTA_KINDS = frozenset(
    {
        QuantityKind.DELTA_ELECTRONIC_ENERGY,
        QuantityKind.DELTA_FORCE_FIELD_ENERGY,
        QuantityKind.DELTA_ENTHALPY,
        QuantityKind.DELTA_GIBBS,
        QuantityKind.BDE,
    }
)

#: kind -> the kind produced by subtracting two of them
_SUBTRACTION_RESULT = {
    QuantityKind.ELECTRONIC_ENERGY: QuantityKind.DELTA_ELECTRONIC_ENERGY,
    QuantityKind.FORCE_FIELD_ENERGY: QuantityKind.DELTA_FORCE_FIELD_ENERGY,
    QuantityKind.ENTHALPY: QuantityKind.DELTA_ENTHALPY,
    QuantityKind.GIBBS: QuantityKind.DELTA_GIBBS,
    QuantityKind.DELTA_ELECTRONIC_ENERGY: QuantityKind.DELTA_ELECTRONIC_ENERGY,
    QuantityKind.DELTA_FORCE_FIELD_ENERGY: QuantityKind.DELTA_FORCE_FIELD_ENERGY,
    QuantityKind.DELTA_ENTHALPY: QuantityKind.DELTA_ENTHALPY,
    QuantityKind.DELTA_GIBBS: QuantityKind.DELTA_GIBBS,
    QuantityKind.PKA: QuantityKind.PKA,
    QuantityKind.LOG_TAUTOMER_RATIO: QuantityKind.LOG_TAUTOMER_RATIO,
}


@dataclass(frozen=True)
class Phase:
    """Gas phase, or solution with a named solvent and optional conditions.

    Ionic strength and pH are carried because experimental pKa are routinely
    reported at conditions that are not comparable to each other, and pooling
    them across conditions is a silent-error source.
    """

    medium: str  # "gas" | "solution"
    solvent: str | None = None
    ionic_strength_M: float | None = None
    ph: float | None = None

    def __post_init__(self) -> None:
        if self.medium not in ("gas", "solution"):
            raise UnitError(f"medium must be 'gas' or 'solution', got {self.medium!r}")
        if self.medium == "gas" and self.solvent is not None:
            raise UnitError("gas phase cannot carry a solvent")
        if self.medium == "solution" and not self.solvent:
            raise UnitError("solution phase requires an explicit solvent")

    def __str__(self) -> str:
        if self.medium == "gas":
            return "gas"
        bits = [self.solvent or "?"]
        if self.ionic_strength_M is not None:
            bits.append(f"I={self.ionic_strength_M}M")
        if self.ph is not None:
            bits.append(f"pH={self.ph}")
        return "solution(" + ", ".join(bits) + ")"


GAS = Phase("gas")


def water(ionic_strength_M: float | None = None, ph: float | None = None) -> Phase:
    return Phase("solution", "water", ionic_strength_M, ph)


@dataclass(frozen=True)
class Quantity:
    """A number that knows what it is.

    Arithmetic is permitted only between quantities that agree on kind, phase
    and temperature. Mixing with a bare float is always an error -- that is the
    most common way a unit bug enters a pipeline.
    """

    value: float
    unit: Unit
    kind: QuantityKind
    phase: Phase
    temperature_K: float | None = STANDARD_TEMPERATURE_K

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise UnitError(f"value must be a real number, got {type(self.value).__name__}")
        if not math.isfinite(self.value):
            raise UnitError(f"value must be finite, got {self.value}")
        if self.unit.dimension is not self.kind.dimension:
            raise UnitError(
                f"unit {self.unit} has dimension {self.unit.dimension.value} but kind "
                f"{self.kind.value} requires {self.kind.dimension.value}"
            )
        if self.temperature_K is not None and self.temperature_K <= 0:
            raise UnitError(f"temperature must be positive, got {self.temperature_K} K")
        # E is a 0 K quantity; H and G are not meaningful without a temperature.
        if self.kind in (QuantityKind.ENTHALPY, QuantityKind.GIBBS) and self.temperature_K is None:
            raise UnitError(f"{self.kind.value} requires an explicit temperature")

    # -- conversion --------------------------------------------------------

    def to(self, unit: Unit | str) -> "Quantity":
        """Convert to another unit of the same dimension."""
        target = unit_from_symbol(unit) if isinstance(unit, str) else unit
        if target.dimension is not self.unit.dimension:
            raise UnitError(
                f"cannot convert {self.unit} ({self.unit.dimension.value}) to "
                f"{target} ({target.dimension.value})"
            )
        new_value = self.value * self.unit.to_canonical / target.to_canonical
        return replace(self, value=new_value, unit=target)

    @property
    def canonical_value(self) -> float:
        """Value in kcal/mol (energies) or log10 units (pKa, log ratios)."""
        return self.value * self.unit.to_canonical

    # -- compatibility -----------------------------------------------------

    def _require_compatible(self, other: Any, op: str) -> "Quantity":
        if not isinstance(other, Quantity):
            raise UnitError(
                f"cannot {op} {type(other).__name__} and Quantity; wrap the bare "
                f"number in a Quantity with explicit units"
            )
        if self.phase != other.phase:
            raise UnitError(f"cannot {op} across phases: {self.phase} vs {other.phase}")
        if self.temperature_K != other.temperature_K:
            raise UnitError(
                f"cannot {op} at different temperatures: "
                f"{self.temperature_K} K vs {other.temperature_K} K"
            )
        if self.kind.dimension is not other.kind.dimension:
            raise UnitError(
                f"cannot {op} {self.kind.value} and {other.kind.value}: different dimensions"
            )
        return other

    def __add__(self, other: Any) -> "Quantity":
        other = self._require_compatible(other, "add")
        if self.kind is not other.kind:
            raise UnitError(f"cannot add {self.kind.value} and {other.kind.value}")
        return replace(self, value=self.value + other.to(self.unit).value)

    def __sub__(self, other: Any) -> "Quantity":
        other = self._require_compatible(other, "subtract")
        if self.kind is not other.kind:
            raise UnitError(f"cannot subtract {other.kind.value} from {self.kind.value}")
        try:
            result_kind = _SUBTRACTION_RESULT[self.kind]
        except KeyError:
            raise UnitError(f"subtraction is not defined for kind {self.kind.value}") from None
        return replace(self, value=self.value - other.to(self.unit).value, kind=result_kind)

    def __mul__(self, scalar: Any) -> "Quantity":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise UnitError("Quantity may only be multiplied by a dimensionless scalar")
        return replace(self, value=self.value * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> "Quantity":
        return replace(self, value=-self.value)

    def __lt__(self, other: Any) -> bool:
        other = self._require_compatible(other, "compare")
        if self.kind is not other.kind:
            raise UnitError(f"cannot compare {self.kind.value} with {other.kind.value}")
        return self.canonical_value < other.canonical_value

    def __le__(self, other: Any) -> bool:
        return self < other or self.isclose(other)

    def isclose(self, other: "Quantity", *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        other = self._require_compatible(other, "compare")
        if self.kind is not other.kind:
            raise UnitError(f"cannot compare {self.kind.value} with {other.kind.value}")
        return math.isclose(
            self.canonical_value, other.canonical_value, rel_tol=rel_tol, abs_tol=abs_tol
        )

    def __str__(self) -> str:
        t = "" if self.temperature_K is None else f" @ {self.temperature_K} K"
        return f"{self.value:.6g} {self.unit} [{self.kind.value}, {self.phase}{t}]"

    # -- serialisation -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit.symbol,
            "kind": self.kind.value,
            "phase": {
                "medium": self.phase.medium,
                "solvent": self.phase.solvent,
                "ionic_strength_M": self.phase.ionic_strength_M,
                "ph": self.phase.ph,
            },
            "temperature_K": self.temperature_K,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Quantity":
        p = d["phase"]
        return cls(
            value=float(d["value"]),
            unit=unit_from_symbol(d["unit"]),
            kind=QuantityKind(d["kind"]),
            phase=Phase(
                p["medium"], p.get("solvent"), p.get("ionic_strength_M"), p.get("ph")
            ),
            temperature_K=d.get("temperature_K"),
        )


# --------------------------------------------------------------------------
# The one sanctioned bridge between log10 units and free energy.
# --------------------------------------------------------------------------


def log10_to_molar_energy(q: Quantity, *, unit: Unit | str = KCAL_PER_MOL) -> Quantity:
    """Convert a pKa or log tautomer ratio into a Gibbs free energy difference.

        dG = R * T * ln(10) * (log10 value)

    A pKa of 4.0 at 298.15 K is a deprotonation free energy of +5.457 kcal/mol.
    """
    if q.kind.dimension is not Dimension.LOG10:
        raise UnitError(f"expected a log10-dimensioned quantity, got {q.kind.value}")
    if q.temperature_K is None:
        raise UnitError("converting a log10 quantity to energy requires a temperature")
    kcal = q.canonical_value * rt_ln10_kcal_per_mol(q.temperature_K)
    out = Quantity(kcal, KCAL_PER_MOL, QuantityKind.DELTA_GIBBS, q.phase, q.temperature_K)
    return out.to(unit)


def molar_energy_to_log10(q: Quantity, kind: QuantityKind = QuantityKind.PKA) -> Quantity:
    """Inverse of `log10_to_molar_energy`.

    Only a Gibbs free energy difference may be converted; passing an electronic
    energy or an enthalpy raises, because the resulting pKa would be wrong by
    the entropic term.
    """
    if q.kind is not QuantityKind.DELTA_GIBBS:
        raise UnitError(
            f"only DELTA_GIBBS converts to a log10 quantity, got {q.kind.value}; "
            f"a pKa derived from {q.kind.value} omits the entropic term"
        )
    if kind.dimension is not Dimension.LOG10:
        raise UnitError(f"target kind {kind.value} is not log10-dimensioned")
    if q.temperature_K is None:
        raise UnitError("converting an energy to a log10 quantity requires a temperature")
    value = q.canonical_value / rt_ln10_kcal_per_mol(q.temperature_K)
    return Quantity(value, LOG10_UNIT, kind, q.phase, q.temperature_K)
