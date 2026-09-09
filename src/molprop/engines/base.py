"""Engine protocol.

An engine is handed a fidelity and must honour it exactly or raise. It may not
substitute a nearby method, silently drop a solvation model, or fall back to a
smaller basis: the fidelity id is a promise about what produced the number, and
an engine that quietly deviates makes every downstream record a lie.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..calc import CalculationResult, Task
from ..fidelity import Fidelity
from ..geometry import ConformerEnsemble

__all__ = ["EngineError", "Engine", "engine_for"]


class EngineError(RuntimeError):
    """Raised when an engine is unavailable or asked for something it cannot do."""


@runtime_checkable
class Engine(Protocol):
    """Minimal contract every QM backend implements."""

    name: str

    def is_available(self) -> bool:
        """True if the executable or library is present and runnable."""
        ...

    def version(self) -> str:
        """Version string, recorded on every result."""
        ...

    def supports(self, fidelity: Fidelity, task: Task) -> bool:
        """True if this engine can run exactly this fidelity and task."""
        ...

    def run(
        self,
        ensemble: ConformerEnsemble,
        fidelity: Fidelity,
        task: Task,
        *,
        temperature_K: float = 298.15,
    ) -> list[CalculationResult]:
        """Run one job per conformer in the ensemble."""
        ...


def engine_for(fidelity: Fidelity, task: Task, engines: Sequence[Engine]) -> Engine:
    """Pick the first available engine that supports this fidelity exactly."""
    tried = []
    for engine in engines:
        tried.append(engine.name)
        if not engine.is_available():
            continue
        if engine.supports(fidelity, task):
            return engine
    raise EngineError(
        f"no available engine supports {fidelity.label()} / {task.value}; "
        f"tried {tried}. Rather than substituting a nearby method, fix the "
        f"fidelity or install the engine."
    )
