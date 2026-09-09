"""Calculation specs, results, and resource accounting.

A `CalculationSpec` is the deterministic identity of one QM job: molecule,
charge, spin, fidelity, task, and the *geometry* it starts from. The geometry is
part of the identity because a single point on an xTB geometry and the same
single point on a DFT geometry are different numbers, and the fidelity's
`geometry_source` field only records that fact qualitatively.

Wall time and cost are recorded from the first run, not bolted on later. Whether
the DFT track is feasible at the scale this project needs is a question about
seconds per molecule, and it is cheaper to answer on run one than on run ten
thousand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .fidelity import Fidelity
from .geometry import Conformer
from .records import MoleculeRef
from .units import (
    HARTREE,
    STANDARD_TEMPERATURE_K,
    Phase,
    Quantity,
    QuantityKind,
    UnitError,
)

__all__ = [
    "CalculationError",
    "Task",
    "CalculationSpec",
    "ResourceUsage",
    "CalculationResult",
    "geometry_hash",
]


class CalculationError(RuntimeError):
    """Raised when an engine fails or returns something unusable."""


class Task(Enum):
    """What the engine is being asked to do."""

    SINGLE_POINT = "sp"
    OPTIMIZE = "opt"
    #: optimisation followed by a Hessian -- the only task that yields H and G
    OPT_FREQ = "ohess"
    HESSIAN = "hess"

    @property
    def yields_thermochemistry(self) -> bool:
        return self in (Task.OPT_FREQ, Task.HESSIAN)

    @property
    def changes_geometry(self) -> bool:
        return self in (Task.OPTIMIZE, Task.OPT_FREQ)


def geometry_hash(coords: np.ndarray, elements: tuple[str, ...]) -> str:
    """Stable hash of a geometry, rounded to 1e-6 A so float noise is invisible."""
    arr = np.round(np.asarray(coords, dtype=float), 6) + 0.0  # kill negative zero
    payload = {"elements": list(elements), "coords": arr.tolist()}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CalculationSpec:
    """The deterministic identity of one job."""

    molecule: MoleculeRef
    fidelity: Fidelity
    task: Task
    geometry_hash: str
    charge: int = 0
    multiplicity: int = 1
    temperature_K: float = STANDARD_TEMPERATURE_K

    def __post_init__(self) -> None:
        if self.temperature_K <= 0:
            raise CalculationError(f"temperature must be positive, got {self.temperature_K}")
        if not self.geometry_hash:
            raise CalculationError(
                "a job identity must pin the geometry it starts from; a single "
                "point on an xTB geometry is not the same number as one on a DFT "
                "geometry"
            )

    @property
    def id(self) -> str:
        payload = {
            "smiles": self.molecule.smiles,
            "fidelity": self.fidelity.id,
            "task": self.task.value,
            "geometry": self.geometry_hash,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "temperature_K": round(float(self.temperature_K), 6),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def n_unpaired(self) -> int:
        return self.multiplicity - 1

    @property
    def phase(self) -> Phase:
        f = self.fidelity
        if f.solvation_model:
            return Phase("solution", f.solvation_solvent)
        return Phase("gas")

    def describe(self) -> str:
        spin = "" if self.multiplicity == 1 else f" mult={self.multiplicity}"
        chg = "" if self.charge == 0 else f" charge={self.charge:+d}"
        return (
            f"{self.task.value}/{self.fidelity.label()} on {self.molecule.smiles}"
            f"{chg}{spin} [{self.id}]"
        )


@dataclass(frozen=True)
class ResourceUsage:
    """Wall time and, where rates are known, cost.

    `cost_usd` is None when no rate was supplied rather than zero: an unpriced
    run is not a free run, and reporting zero would make a campaign budget look
    achievable when it has simply not been measured.
    """

    wall_seconds: float
    n_cores: int = 1
    gpu_seconds: float = 0.0
    usd_per_core_hour: float | None = None
    usd_per_gpu_hour: float | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds < 0 or self.gpu_seconds < 0:
            raise CalculationError("times must be non-negative")
        if self.n_cores < 1:
            raise CalculationError("n_cores must be at least 1")

    @property
    def core_hours(self) -> float:
        return self.wall_seconds * self.n_cores / 3600.0

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    @property
    def cost_usd(self) -> float | None:
        if self.usd_per_core_hour is None and self.usd_per_gpu_hour is None:
            return None
        total = 0.0
        if self.usd_per_core_hour is not None:
            total += self.core_hours * self.usd_per_core_hour
        if self.usd_per_gpu_hour is not None:
            total += self.gpu_hours * self.usd_per_gpu_hour
        return total

    def __add__(self, other: "ResourceUsage") -> "ResourceUsage":
        if not isinstance(other, ResourceUsage):
            raise CalculationError("can only add ResourceUsage to ResourceUsage")
        rates = {
            "usd_per_core_hour": self.usd_per_core_hour
            if self.usd_per_core_hour == other.usd_per_core_hour
            else None,
            "usd_per_gpu_hour": self.usd_per_gpu_hour
            if self.usd_per_gpu_hour == other.usd_per_gpu_hour
            else None,
        }
        # Cores are not additive; keep the larger so core_hours stays an
        # over-estimate rather than a silent under-count.
        return ResourceUsage(
            wall_seconds=self.wall_seconds + other.wall_seconds,
            n_cores=max(self.n_cores, other.n_cores),
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            **rates,
        )

    def describe(self) -> str:
        cost = "unpriced" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        return f"{self.wall_seconds:.2f}s wall, {self.core_hours:.4f} core-h, {cost}"


@dataclass
class CalculationResult:
    """What an engine returns for one spec.

    Every energy is a `Quantity`, so a total energy can never be mistaken for a
    free energy downstream. `enthalpy` and `gibbs` are None unless the task
    actually produced a Hessian.
    """

    spec: CalculationSpec
    converged: bool
    usage: ResourceUsage
    engine: str
    engine_version: str
    energy: Quantity | None = None
    enthalpy: Quantity | None = None
    gibbs: Quantity | None = None
    zpe: Quantity | None = None
    geometry: Conformer | None = None
    n_imaginary: int | None = None
    warnings: tuple[str, ...] = ()
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("energy", "enthalpy", "gibbs", "zpe"):
            q = getattr(self, name)
            if q is None:
                continue
            if q.phase != self.spec.phase:
                raise UnitError(
                    f"{name} is reported in {q.phase} but the fidelity implies "
                    f"{self.spec.phase}"
                )
        expected = {
            "energy": QuantityKind.ELECTRONIC_ENERGY,
            "enthalpy": QuantityKind.ENTHALPY,
            "gibbs": QuantityKind.GIBBS,
            "zpe": QuantityKind.ZERO_POINT_ENERGY,
        }
        for name, kind in expected.items():
            q = getattr(self, name)
            if q is not None and q.kind is not kind:
                raise UnitError(f"{name} must be of kind {kind.value}, got {q.kind.value}")
        if not self.spec.task.yields_thermochemistry and (self.enthalpy or self.gibbs):
            raise CalculationError(
                f"task {self.spec.task.value} cannot produce H or G; a Hessian is "
                f"required and none was requested"
            )

    @property
    def is_minimum(self) -> bool | None:
        """True if a Hessian was computed and found no imaginary frequencies."""
        if self.n_imaginary is None:
            return None
        return self.n_imaginary == 0

    def require_usable(self) -> "CalculationResult":
        """Raise unless this result is safe to build a property value from."""
        if not self.converged:
            raise CalculationError(f"{self.spec.describe()} did not converge")
        if self.energy is None:
            raise CalculationError(f"{self.spec.describe()} returned no energy")
        if self.spec.task.yields_thermochemistry and self.n_imaginary:
            raise CalculationError(
                f"{self.spec.describe()} has {self.n_imaginary} imaginary "
                f"frequencies; it is a saddle point, not a minimum, and its "
                f"thermochemistry is not a free energy of a stable species"
            )
        return self

    def describe(self) -> str:
        bits = [self.spec.describe(), self.usage.describe()]
        if self.energy is not None:
            bits.append(f"E={self.energy.to(HARTREE).value:.8f} Eh")
        if self.gibbs is not None:
            bits.append(f"G={self.gibbs.to(HARTREE).value:.8f} Eh")
        if self.n_imaginary is not None:
            bits.append(f"imag={self.n_imaginary}")
        return " | ".join(bits)
