"""Ensemble thermochemistry.

Boltzmann averaging is where the conformer protocol stops being bookkeeping and
starts changing the answer. Two rules enforced here:

* Averaging is over a *free energy*, not an electronic energy. Weighting by E
  ignores the entropy differences between conformers, which is exactly the term
  that decides which rotamer dominates.
* Every energy in an ensemble must share a kind, a phase and a temperature.
  Mixing an xTB conformer with a DFT one and calling the result an ensemble
  average is a silent error the type system already blocks; this module simply
  refuses to work around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .calc import CalculationResult
from .units import (
    GAS_CONSTANT_KCAL_PER_MOL_K,
    KCAL_PER_MOL,
    STANDARD_TEMPERATURE_K,
    Quantity,
    QuantityKind,
    UnitError,
)

__all__ = [
    "ThermoError",
    "boltzmann_weights",
    "boltzmann_average",
    "ensemble_quantity",
    "monatomic_thermochemistry",
    "monatomic_entropy_kcal_per_mol_k",
    "ATOMIC_MASS_U",
]


class ThermoError(ValueError):
    """Raised on an ill-formed ensemble average."""


def _check_homogeneous(quantities: Sequence[Quantity]) -> Quantity:
    if not quantities:
        raise ThermoError("cannot average an empty set of quantities")
    first = quantities[0]
    for q in quantities[1:]:
        if q.kind is not first.kind:
            raise UnitError(
                f"ensemble mixes kinds {first.kind.value} and {q.kind.value}"
            )
        if q.phase != first.phase:
            raise UnitError(f"ensemble mixes phases {first.phase} and {q.phase}")
        if q.temperature_K != first.temperature_K:
            raise UnitError(
                f"ensemble mixes temperatures {first.temperature_K} and "
                f"{q.temperature_K} K"
            )
    return first


def boltzmann_weights(
    energies: Sequence[Quantity], *, temperature_K: float | None = None
) -> list[float]:
    """Normalised populations from relative energies.

    Computed in log space so a wide energy range cannot overflow.
    """
    reference = _check_homogeneous(energies)
    temp = temperature_K if temperature_K is not None else reference.temperature_K
    if temp is None:
        raise ThermoError(
            "Boltzmann weighting needs a temperature; a 0 K electronic energy "
            "carries none, so pass one explicitly or use free energies"
        )
    if temp <= 0:
        raise ThermoError(f"temperature must be positive, got {temp} K")

    rt = GAS_CONSTANT_KCAL_PER_MOL_K * temp
    values = [q.canonical_value for q in energies]
    lowest = min(values)
    exponents = [-(v - lowest) / rt for v in values]
    weights = [math.exp(x) for x in exponents]
    total = sum(weights)
    if total <= 0:  # pragma: no cover - unreachable for finite inputs
        raise ThermoError("Boltzmann weights summed to zero")
    return [w / total for w in weights]


def boltzmann_average(
    quantities: Sequence[Quantity],
    weighting_energies: Sequence[Quantity] | None = None,
    *,
    temperature_K: float | None = None,
) -> Quantity:
    """Population-weighted average of `quantities`.

    When `weighting_energies` is omitted the quantities weight themselves,
    which is what you want for a free-energy ensemble.
    """
    reference = _check_homogeneous(quantities)
    energies = list(weighting_energies) if weighting_energies is not None else list(quantities)
    if len(energies) != len(quantities):
        raise ThermoError(
            f"{len(energies)} weighting energies for {len(quantities)} quantities"
        )
    if weighting_energies is None and reference.kind not in (
        QuantityKind.GIBBS,
        QuantityKind.DELTA_GIBBS,
    ):
        raise UnitError(
            f"self-weighting an ensemble of {reference.kind.value} ignores the "
            f"entropic differences between conformers; pass explicit free "
            f"energies as weighting_energies, or average Gibbs energies"
        )
    weights = boltzmann_weights(energies, temperature_K=temperature_K)
    total = sum(w * q.canonical_value for w, q in zip(weights, quantities))
    return Quantity(
        total, KCAL_PER_MOL, reference.kind, reference.phase, reference.temperature_K
    ).to(reference.unit)


@dataclass(frozen=True)
class EnsembleThermo:
    """The reduction of a conformer ensemble to single numbers."""

    energy: Quantity | None
    enthalpy: Quantity | None
    gibbs: Quantity | None
    weights: tuple[float, ...]
    treatment: str
    n_conformers: int

    def describe(self) -> str:
        top = max(self.weights) if self.weights else float("nan")
        return (
            f"{self.treatment} over {self.n_conformers} conformers "
            f"(max weight {top:.3f})"
        )


def ensemble_quantity(
    results: Sequence[CalculationResult], *, treatment: str, temperature_K: float | None = None
) -> EnsembleThermo:
    """Reduce per-conformer results to one value using the stated treatment.

    `treatment` must match the ConformerProtocol that produced the ensemble --
    reducing a Boltzmann protocol by taking the lowest conformer would make the
    recorded fidelity id describe a calculation that never happened.
    """
    usable = [r.require_usable() for r in results]
    if not usable:
        raise ThermoError("no results to reduce")

    if treatment not in ("lowest", "boltzmann", "all"):
        raise ThermoError(f"unknown ensemble treatment {treatment!r}")

    gibbs = [r.gibbs for r in usable]
    have_gibbs = all(g is not None for g in gibbs)

    if treatment == "lowest" or len(usable) == 1:
        ranker = gibbs if have_gibbs else [r.energy for r in usable]
        best = min(range(len(usable)), key=lambda i: ranker[i].canonical_value)
        chosen = usable[best]
        weights = tuple(1.0 if i == best else 0.0 for i in range(len(usable)))
        return EnsembleThermo(
            energy=chosen.energy,
            enthalpy=chosen.enthalpy,
            gibbs=chosen.gibbs,
            weights=weights,
            treatment="lowest",
            n_conformers=len(usable),
        )

    if not have_gibbs:
        raise ThermoError(
            "Boltzmann averaging requires a free energy for every conformer; "
            "run a Hessian task rather than weighting electronic energies"
        )
    weights = boltzmann_weights(gibbs, temperature_K=temperature_K)
    enthalpies = [r.enthalpy for r in usable]
    return EnsembleThermo(
        energy=boltzmann_average([r.energy for r in usable], gibbs, temperature_K=temperature_K),
        enthalpy=(
            boltzmann_average(enthalpies, gibbs, temperature_K=temperature_K)
            if all(h is not None for h in enthalpies)
            else None
        ),
        gibbs=boltzmann_average(gibbs, gibbs, temperature_K=temperature_K),
        weights=tuple(weights),
        treatment="boltzmann",
        n_conformers=len(usable),
    )


# --------------------------------------------------------------------------
# Monatomic species
# --------------------------------------------------------------------------

#: Standard atomic weights, u. Needed for the translational partition function.
ATOMIC_MASS_U = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Br": 79.904, "I": 126.90447,
}

_BOLTZMANN_J_PER_K = 1.380649e-23
_PLANCK_J_S = 6.62607015e-34
_AMU_KG = 1.66053906660e-27
_STANDARD_PRESSURE_PA = 101325.0


def monatomic_entropy_kcal_per_mol_k(
    element: str,
    temperature_K: float = STANDARD_TEMPERATURE_K,
    *,
    multiplicity: int = 1,
    pressure_pa: float = _STANDARD_PRESSURE_PA,
) -> float:
    """Sackur-Tetrode translational entropy plus the electronic term.

    For the hydrogen atom at 298.15 K and 1 atm this gives 27.4 cal/(mol K),
    matching the tabulated standard entropy.
    """
    try:
        mass = ATOMIC_MASS_U[element] * _AMU_KG
    except KeyError:
        raise ThermoError(f"no atomic mass for element {element!r}") from None
    if temperature_K <= 0:
        raise ThermoError("temperature must be positive")
    kt = _BOLTZMANN_J_PER_K * temperature_K
    quantum_volume = (2.0 * math.pi * mass * kt / _PLANCK_J_S**2) ** 1.5
    s_over_r = math.log(quantum_volume * kt / pressure_pa) + 2.5
    s_over_r += math.log(multiplicity)  # electronic degeneracy
    return s_over_r * GAS_CONSTANT_KCAL_PER_MOL_K


def monatomic_thermochemistry(
    energy: Quantity,
    element: str,
    *,
    temperature_K: float = STANDARD_TEMPERATURE_K,
    multiplicity: int = 1,
) -> tuple[Quantity, Quantity]:
    """Exact (H, G) for a single atom in the ideal-gas limit.

    An atom has no vibrations, so a numerical Hessian is not merely wasteful --
    most QM programs refuse the job outright. The corrections are analytic:

        H(T) = E + (5/2) RT          (3/2 RT translation + RT for PV)
        G(T) = H(T) - T S            (S from Sackur-Tetrode + spin degeneracy)
    """
    if energy.kind is not QuantityKind.ELECTRONIC_ENERGY:
        raise UnitError(
            f"monatomic thermochemistry starts from an electronic energy, got "
            f"{energy.kind.value}"
        )
    e_kcal = energy.to(KCAL_PER_MOL).value
    rt = GAS_CONSTANT_KCAL_PER_MOL_K * temperature_K
    h_kcal = e_kcal + 2.5 * rt
    entropy = monatomic_entropy_kcal_per_mol_k(
        element, temperature_K, multiplicity=multiplicity
    )
    g_kcal = h_kcal - temperature_K * entropy
    enthalpy = Quantity(
        h_kcal, KCAL_PER_MOL, QuantityKind.ENTHALPY, energy.phase, temperature_K
    )
    gibbs = Quantity(
        g_kcal, KCAL_PER_MOL, QuantityKind.GIBBS, energy.phase, temperature_K
    )
    return enthalpy, gibbs
