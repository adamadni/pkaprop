"""Macroscopic vs microscopic pKa.

Experimental pKa labels are macroscopic. A site-specific model predicts
microscopic values. For any polyprotic molecule these differ, and training a
per-site model on macroscopic labels without an explicit aggregation step is a
silent target mismatch -- which is why this lives in the schema layer rather
than in the modelling code.

Relation used (Ullmann-style microstate formalism). Write each microstate's
free energy in pK units, g = dG / (RT ln 10), relative to any common reference,
and let level n be the set of microstates carrying n dissociable protons:

    Z_n     = sum over microstates m in level n of 10**(-g_m)
    pKa_n   = log10(Z_n) - log10(Z_{n-1})          (deprotonation n -> n-1)

The statistical factor falls out automatically: two identical independent sites
of micro pKa 4.0 give macro pKa1 = 3.699 and pKa2 = 4.301.

The reverse direction is deliberately not implemented. Decomposing macroscopic
constants into microscopic ones is underdetermined without extra measurement,
and a plausible-looking guess is exactly the failure this module exists to
prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .chem import MoleculeError, canonical_smiles, molecular_formula, net_charge
from .units import (
    LOG10_UNIT,
    STANDARD_TEMPERATURE_K,
    Phase,
    Quantity,
    QuantityKind,
    UnitError,
    water,
)

__all__ = [
    "Microstate",
    "MicrostateEnsemble",
    "macro_pka_from_microstates",
    "microstate_populations",
    "micro_pka_from_macro",
]


@dataclass(frozen=True)
class Microstate:
    """One protonation microstate of a molecule.

    `relative_g_pk_units` is a free energy expressed in pK units, relative to a
    reference shared by every microstate in the ensemble. Only differences
    matter, so the reference may be arbitrary -- but it must be common.
    """

    smiles: str
    n_protons: int
    relative_g_pk_units: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "smiles", canonical_smiles(self.smiles))
        if self.n_protons < 0:
            raise ValueError("n_protons must be non-negative")
        if not math.isfinite(self.relative_g_pk_units):
            raise ValueError("relative_g_pk_units must be finite")

    @property
    def charge(self) -> int:
        return net_charge(self.smiles)


@dataclass(frozen=True)
class MicrostateEnsemble:
    """All microstates of one compound across its protonation levels."""

    states: tuple[Microstate, ...]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("a microstate ensemble needs at least one state")
        # Every microstate must be the same compound at a different protonation
        # level: charge must track proton count exactly.
        offsets = {s.charge - s.n_protons for s in self.states}
        if len(offsets) != 1:
            detail = ", ".join(
                f"{s.smiles} (q={s.charge}, nH={s.n_protons})" for s in self.states
            )
            raise MoleculeError(
                "microstate charges are inconsistent with their proton counts; "
                f"charge - n_protons must be constant across the ensemble: {detail}"
            )
        if len({s.smiles for s in self.states}) != len(self.states):
            raise MoleculeError("duplicate microstate SMILES in ensemble")

    @property
    def levels(self) -> list[int]:
        return sorted({s.n_protons for s in self.states})

    def at_level(self, n: int) -> list[Microstate]:
        return [s for s in self.states if s.n_protons == n]


def _log10_sum_exp10(neg_g: list[float]) -> float:
    """log10(sum(10**x)) computed without overflow."""
    hi = max(neg_g)
    return hi + math.log10(sum(10.0 ** (x - hi) for x in neg_g))


def macro_pka_from_microstates(
    ensemble: MicrostateEnsemble,
    *,
    temperature_K: float = STANDARD_TEMPERATURE_K,
    phase: Phase | None = None,
) -> list[tuple[int, Quantity]]:
    """Aggregate microscopic free energies into macroscopic pKa values.

    Returns [(step_index, pKa)] ordered by deprotonation step, where step 1 is
    the loss of the first proton from the most protonated level present.
    """
    phase = phase or water()
    levels = ensemble.levels
    if len(levels) < 2:
        raise ValueError("at least two protonation levels are needed for a macro pKa")
    if levels != list(range(levels[0], levels[-1] + 1)):
        raise ValueError(
            f"protonation levels must be contiguous, got {levels}; a missing level "
            f"means an unenumerated microstate, not a zero-population one"
        )

    log_z = {
        n: _log10_sum_exp10([-s.relative_g_pk_units for s in ensemble.at_level(n)])
        for n in levels
    }

    out: list[tuple[int, Quantity]] = []
    for step, n in enumerate(reversed(levels[1:]), start=1):
        pka = log_z[n] - log_z[n - 1]
        out.append(
            (
                step,
                Quantity(pka, LOG10_UNIT, QuantityKind.PKA, phase, temperature_K),
            )
        )
    return out


def microstate_populations(
    ensemble: MicrostateEnsemble, ph: float
) -> dict[str, float]:
    """Fractional population of each microstate at a given pH.

    Useful for the question drug discovery actually asks -- which species
    dominates at pH 7.4 -- and for checking that an enumeration is complete.
    """
    if not math.isfinite(ph):
        raise ValueError("pH must be finite")
    # Adding a proton costs the solution one unit of proton chemical potential,
    # worth `ph` in pK units.
    weights = {
        s.smiles: -(s.relative_g_pk_units) - s.n_protons * ph for s in ensemble.states
    }
    norm = _log10_sum_exp10(list(weights.values()))
    return {k: 10.0 ** (v - norm) for k, v in weights.items()}


def micro_pka_from_macro(*_args, **_kwargs):  # noqa: D401 - intentional
    """Not implemented, and not an oversight.

    Decomposing macroscopic pKa into microscopic ones is underdetermined: for a
    molecule with two ionisable sites, the two macroscopic constants constrain
    but do not determine the four microscopic ones. Resolving them needs extra
    information (site-specific NMR/UV titration, or a model of the interaction
    term). Pick one explicitly rather than letting a helper guess.
    """
    raise NotImplementedError(
        "macro -> micro decomposition is underdetermined; supply site-resolved "
        "measurements or state the interaction model explicitly"
    )


def check_ensemble_formula(ensemble: MicrostateEnsemble) -> None:
    """Assert every microstate is the same compound bar protonation state.

    Compares heavy-atom composition by stripping hydrogens from the formula.
    """
    def heavy_composition(smiles: str) -> str:
        formula = molecular_formula(smiles)
        # Drop the hydrogen term and any trailing charge annotation.
        out, i = [], 0
        while i < len(formula):
            ch = formula[i]
            if ch == "H" and (i + 1 >= len(formula) or not formula[i + 1].islower()):
                i += 1
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                continue
            if ch in "+-":
                break
            out.append(ch)
            i += 1
        return "".join(out)

    comps = {heavy_composition(s.smiles) for s in ensemble.states}
    if len(comps) != 1:
        raise MoleculeError(
            f"microstates do not share a heavy-atom composition: {sorted(comps)}"
        )


def pka_quantity(
    value: float,
    *,
    temperature_K: float = STANDARD_TEMPERATURE_K,
    phase: Phase | None = None,
) -> Quantity:
    """Convenience constructor so callers never build a pKa with a wrong unit."""
    phase = phase or water()
    if phase.medium != "solution":
        raise UnitError("a pKa is a solution-phase quantity")
    return Quantity(value, LOG10_UNIT, QuantityKind.PKA, phase, temperature_K)
