"""QM engines.

Every engine takes a `ConformerEnsemble` plus a `CalculationSpec` and returns
`CalculationResult`s. Engines never invent a fidelity: they are handed one and
must either honour it exactly or raise, so that a recorded fidelity id always
describes the calculation that actually ran.
"""

from .base import Engine, EngineError, engine_for
from .xtb import XTBEngine

__all__ = ["Engine", "EngineError", "XTBEngine", "engine_for"]
