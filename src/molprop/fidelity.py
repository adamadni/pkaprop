"""Fidelity specification.

"DFT" is not a fidelity label. wB97X-D/def2-TZVPD/SMD and B3LYP/6-31G*/PCM
differ by more than the delta a model is being asked to learn, so every record
carries a full fidelity *tuple* -- method, basis, dispersion, solvation,
geometry source, and conformer protocol -- hashed to a stable id.

The conformer protocol is part of the fidelity, not an implementation detail.
For pKa and tautomer ratios the answer is a Boltzmann average over the
conformer ensembles of two different species, and the choice between
lowest-conformer and a full CREST/CENSO ensemble moves the answer more than
the choice of functional does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "FidelityError",
    "Tier",
    "ConformerProtocol",
    "Fidelity",
    "FidelityLadder",
    "DeltaSpec",
]


class FidelityError(ValueError):
    """Raised on an incomplete or incoherent fidelity specification."""


class Tier(Enum):
    """Coarse rung of the fidelity ladder.

    EXPERIMENT sits at the top for solution-phase observables (pKa, tautomer
    ratios), where no level of theory is the ground truth -- absolute QM pKa
    always carries an empirical calibration. For gas-phase BDE, DFT is a
    legitimate reference and EXPERIMENT may be absent from the ladder.
    """

    FORCE_FIELD = "force_field"
    SEMIEMPIRICAL = "semiempirical"
    DFT = "dft"
    WFT = "wft"
    EXPERIMENT = "experiment"


_REQUIRES_BASIS = frozenset({Tier.DFT, Tier.WFT})
_REQUIRES_CONFORMERS = frozenset(
    {Tier.FORCE_FIELD, Tier.SEMIEMPIRICAL, Tier.DFT, Tier.WFT}
)


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ConformerProtocol:
    """How the conformer ensemble was produced and reduced to one number.

    `ensemble_treatment` is the field that most often silently differs between
    two datasets that otherwise look identical.
    """

    generator: str  # rdkit_etkdgv3 | crest | provided | none
    #: lowest | boltzmann | all -- how many conformers reach the final value
    ensemble_treatment: str
    n_confs_requested: int | None = None
    rmsd_prune_angstrom: float | None = None
    #: which atoms the pruning RMSD is measured over. "heavy" is the usual
    #: choice, but it cannot see O-H / N-H rotamers -- and those change
    #: intramolecular hydrogen bonding, which is exactly what a pKa or an O-H
    #: BDE depends on. Set "all" for the pKa and tautomer tracks.
    rmsd_atoms: str = "heavy"
    optimizer: str | None = None  # mmff94s | gff | gfn2-xtb | <dft method>
    energy_window_kcal_per_mol: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.rmsd_atoms not in ("heavy", "all"):
            raise FidelityError(
                f"rmsd_atoms must be 'heavy' or 'all', got {self.rmsd_atoms!r}"
            )
        if self.ensemble_treatment not in ("lowest", "boltzmann", "all"):
            raise FidelityError(
                f"ensemble_treatment must be lowest|boltzmann|all, "
                f"got {self.ensemble_treatment!r}"
            )
        if self.generator not in ("rdkit_etkdgv3", "crest", "provided", "none"):
            raise FidelityError(f"unknown conformer generator {self.generator!r}")
        if self.generator != "none" and self.seed is None and self.generator != "provided":
            raise FidelityError(
                "a stochastic conformer generator requires an explicit seed; "
                "conformer determinism is a step-2 validation gate"
            )
        if self.ensemble_treatment == "boltzmann" and self.energy_window_kcal_per_mol is None:
            raise FidelityError(
                "boltzmann averaging requires an explicit energy_window_kcal_per_mol"
            )

    @property
    def id(self) -> str:
        return _hash_payload(asdict(self))

    def describe(self) -> str:
        bits = [self.generator]
        if self.n_confs_requested is not None:
            bits.append(f"n={self.n_confs_requested}")
        if self.optimizer:
            bits.append(f"opt={self.optimizer}")
        if self.rmsd_prune_angstrom is not None:
            bits.append(f"prune={self.rmsd_prune_angstrom}A/{self.rmsd_atoms}")
        bits.append(self.ensemble_treatment)
        if self.energy_window_kcal_per_mol is not None:
            bits.append(f"window={self.energy_window_kcal_per_mol}kcal/mol")
        return "/".join(bits)


#: No conformer search at all -- experimental records, or a single provided 3D
#: structure whose provenance lives in the record metadata.
NO_CONFORMERS = ConformerProtocol(generator="none", ensemble_treatment="lowest")


@dataclass(frozen=True)
class Fidelity:
    """A fully specified level of theory (or 'experiment')."""

    tier: Tier
    method: str  # wB97X-D | M06-2X | GFN2-xTB | experiment
    basis: str | None = None
    dispersion: str | None = None
    solvation_model: str | None = None  # SMD | CPCM | ALPB | GBSA | None
    solvation_solvent: str | None = None
    #: "self" if the geometry was optimised at this level, otherwise the id of
    #: the fidelity the geometry came from (single-point on someone else's
    #: geometry is a different fidelity and must say so).
    geometry_source: str = "self"
    conformers: ConformerProtocol = NO_CONFORMERS
    software: str | None = None
    software_version: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.method:
            raise FidelityError("method is required")
        if self.tier in _REQUIRES_BASIS and not self.basis:
            raise FidelityError(
                f"tier {self.tier.value} requires an explicit basis set; "
                f"'{self.method}' alone is not a fidelity label"
            )
        if self.tier is Tier.EXPERIMENT:
            if self.basis or self.dispersion or self.solvation_model:
                raise FidelityError("experimental fidelity cannot carry a level of theory")
            if self.conformers is not NO_CONFORMERS and self.conformers.generator != "none":
                raise FidelityError("experimental fidelity cannot carry a conformer protocol")
        if self.tier in _REQUIRES_CONFORMERS and self.conformers.generator == "none":
            raise FidelityError(
                f"tier {self.tier.value} requires an explicit conformer protocol; "
                f"use ConformerProtocol(generator='provided', ...) if geometries are given"
            )
        if bool(self.solvation_model) != bool(self.solvation_solvent):
            raise FidelityError(
                "solvation_model and solvation_solvent must be given together"
            )

    @property
    def id(self) -> str:
        """Stable 12-hex-char hash over every field that defines the number."""
        payload = asdict(self)
        payload["tier"] = self.tier.value
        payload.pop("notes", None)  # free text must not change the identity
        return _hash_payload(payload)

    @property
    def is_solution_phase(self) -> bool:
        return self.solvation_model is not None or self.tier is Tier.EXPERIMENT

    def label(self) -> str:
        """Human-readable one-liner, e.g. 'wB97X-D/def2-TZVPD/SMD(water)'."""
        if self.tier is Tier.EXPERIMENT:
            return "experiment"
        bits = self.method
        if self.basis:
            bits += f"/{self.basis}"
        if self.dispersion:
            bits += f"-{self.dispersion}"
        if self.solvation_model:
            bits += f"/{self.solvation_model}({self.solvation_solvent})"
        return bits

    def describe(self) -> str:
        geom = "opt@self" if self.geometry_source == "self" else f"sp@{self.geometry_source}"
        return f"{self.label()} [{geom}; confs={self.conformers.describe()}] ({self.id})"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.value
        d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fidelity":
        d = dict(d)
        d.pop("id", None)
        conf = d.pop("conformers", None)
        conformers = ConformerProtocol(**conf) if conf else NO_CONFORMERS
        return cls(tier=Tier(d.pop("tier")), conformers=conformers, **d)


@dataclass
class FidelityLadder:
    """An explicitly ordered set of fidelities for one property track.

    Fidelity is a model input, which means it needs an ordinal, and the ordinal
    has to be asserted by a human rather than inferred from the tier -- a large
    -basis semiempirical-geometry single point does not slot in where its tier
    suggests.
    """

    name: str
    #: fidelity -> rank; higher rank == closer to the track's reference
    entries: dict[str, tuple[Fidelity, int]] = field(default_factory=dict)

    def add(self, fidelity: Fidelity, rank: int) -> "FidelityLadder":
        if fidelity.id in self.entries:
            raise FidelityError(f"fidelity {fidelity.id} already in ladder {self.name!r}")
        if any(r == rank for _, r in self.entries.values()):
            clash = next(f for f, r in self.entries.values() if r == rank)
            raise FidelityError(
                f"rank {rank} already taken in ladder {self.name!r} by {clash.label()}"
            )
        self.entries[fidelity.id] = (fidelity, rank)
        return self

    def rank(self, fidelity: Fidelity | str) -> int:
        fid = fidelity if isinstance(fidelity, str) else fidelity.id
        try:
            return self.entries[fid][1]
        except KeyError:
            raise FidelityError(
                f"fidelity {fid} is not registered in ladder {self.name!r}; "
                f"register it with an explicit rank before using it as a model input"
            ) from None

    @property
    def reference(self) -> Fidelity:
        """The top rung -- what everything else is a correction towards."""
        if not self.entries:
            raise FidelityError(f"ladder {self.name!r} is empty")
        return max(self.entries.values(), key=lambda kv: kv[1])[0]

    def sorted(self) -> list[tuple[Fidelity, int]]:
        return sorted(self.entries.values(), key=lambda kv: kv[1])

    def describe(self) -> str:
        lines = [f"ladder {self.name!r}:"]
        for f, r in reversed(self.sorted()):
            lines.append(f"  {r:>2}  {f.describe()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class DeltaSpec:
    """A delta-learning target: predict target - base.

    The BDE and pKa tracks are structurally different and this type is where
    that difference is made explicit:

      * BDE  -- base=GFN2-xTB, target=DFT.  DFT is a legitimate ground truth.
      * pKa  -- base=xTB or DFT, target=EXPERIMENT.  A DFT pKa is itself a
                fitted quantity, so it is a fidelity rung, never the reference.
    """

    base: Fidelity
    target: Fidelity
    ladder: FidelityLadder

    def __post_init__(self) -> None:
        base_rank = self.ladder.rank(self.base)
        target_rank = self.ladder.rank(self.target)
        if target_rank <= base_rank:
            raise FidelityError(
                f"delta target must outrank the base: "
                f"{self.target.label()} (rank {target_rank}) does not outrank "
                f"{self.base.label()} (rank {base_rank})"
            )
        if self.base.is_solution_phase != self.target.is_solution_phase:
            raise FidelityError(
                "delta between a gas-phase and a solution-phase fidelity is not a "
                "correction, it is a solvation free energy; make that an explicit step"
            )

    @property
    def id(self) -> str:
        return _hash_payload({"base": self.base.id, "target": self.target.id})

    def describe(self) -> str:
        return f"delta[{self.base.label()} -> {self.target.label()}] ({self.id})"
