"""Result cards.

Project constraint: no model result is reported without a baseline comparison on
the same split. That is enforced here rather than left to discipline -- a
`ResultCard` with no baselines refuses to be constructed, and a baseline whose
split id differs from the model's refuses to be attached.

Metrics carry units for the same reason property values do. An MAE of 0.6 is
meaningless; an MAE of 0.6 kcal/mol is a claim about beating ALFABET, and an MAE
of 0.6 log units is a claim about pKa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .fidelity import DeltaSpec, Fidelity
from .splits import Split
from .units import Unit, UnitError, unit_from_symbol

__all__ = ["CardError", "Metric", "Evaluation", "ResultCard"]


class CardError(ValueError):
    """Raised when a card is missing something that makes it uninterpretable."""


@dataclass(frozen=True)
class Metric:
    """A scalar score with an explicit unit.

    `unit` is a Unit for dimensioned errors (MAE, RMSE in kcal/mol or log
    units) and None only for genuinely dimensionless scores (R2, Spearman,
    accuracy), which must be named in `DIMENSIONLESS_METRICS`.
    """

    name: str
    value: float
    unit: Unit | None = None
    n: int | None = None

    DIMENSIONLESS_METRICS = frozenset(
        {"r2", "pearson", "spearman", "kendall", "accuracy", "auroc", "auprc", "f1"}
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise CardError("a metric needs a name")
        if self.unit is None and self.name.lower() not in self.DIMENSIONLESS_METRICS:
            raise UnitError(
                f"metric {self.name!r} has no unit; dimensioned metrics must state "
                f"one (known dimensionless metrics: "
                f"{sorted(self.DIMENSIONLESS_METRICS)})"
            )
        if self.n is not None and self.n < 1:
            raise CardError("n must be a positive count of evaluated records")

    def __str__(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        n = f" (n={self.n})" if self.n is not None else ""
        return f"{self.name}={self.value:.4g}{unit}{n}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.symbol if self.unit else None,
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Metric":
        return cls(
            name=d["name"],
            value=float(d["value"]),
            unit=unit_from_symbol(d["unit"]) if d.get("unit") else None,
            n=d.get("n"),
        )


@dataclass(frozen=True)
class Evaluation:
    """One system evaluated on one partition of one split."""

    name: str
    split_id: str
    partition: str
    metrics: tuple[Metric, ...]
    #: free-text description of what the system is, e.g. "bond-type mean lookup"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise CardError("an evaluation needs a name")
        if not self.metrics:
            raise CardError(f"evaluation {self.name!r} reports no metrics")
        if self.partition not in ("train", "val", "test", "frozen_test"):
            raise CardError(f"unknown partition {self.partition!r}")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise CardError(f"duplicate metric names in {self.name!r}: {names}")

    def metric(self, name: str) -> Metric:
        for m in self.metrics:
            if m.name == name:
                return m
        raise CardError(f"{self.name!r} has no metric {name!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split_id": self.split_id,
            "partition": self.partition,
            "description": self.description,
            "metrics": [m.as_dict() for m in self.metrics],
        }


@dataclass(frozen=True)
class ResultCard:
    """A reportable result. Refuses to exist without a baseline."""

    model: Evaluation
    baselines: tuple[Evaluation, ...]
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    split: Split
    fidelity: Fidelity | None = None
    delta: DeltaSpec | None = None
    notes: str | None = None
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not self.baselines:
            raise CardError(
                f"no result without a baseline: {self.model.name!r} was evaluated "
                f"but no baseline was run on the same split. A bond-type mean "
                f"lookup or a fingerprint GBM takes minutes and is often close."
            )
        if self.model.split_id != self.split.id:
            raise CardError(
                f"model was evaluated on split {self.model.split_id} but the card "
                f"carries split {self.split.id}"
            )
        for b in self.baselines:
            if b.split_id != self.split.id:
                raise CardError(
                    f"baseline {b.name!r} was evaluated on split {b.split_id}, not "
                    f"the model's split {self.split.id}; the comparison is invalid"
                )
            if b.partition != self.model.partition:
                raise CardError(
                    f"baseline {b.name!r} reports on partition {b.partition!r} but "
                    f"the model reports on {self.model.partition!r}"
                )
        model_metrics = {m.name for m in self.model.metrics}
        for b in self.baselines:
            shared = model_metrics & {m.name for m in b.metrics}
            if not shared:
                raise CardError(
                    f"baseline {b.name!r} shares no metric with the model; "
                    f"there is nothing to compare"
                )
            for name in shared:
                mu, bu = self.model.metric(name).unit, b.metric(name).unit
                if (mu is None) != (bu is None) or (mu and bu and mu.dimension is not bu.dimension):
                    raise UnitError(
                        f"metric {name!r} has incompatible units between model "
                        f"({mu}) and baseline {b.name!r} ({bu})"
                    )
        if self.delta is not None and self.fidelity is not None:
            raise CardError("give either a fidelity or a delta spec, not both")

    def comparison(self, metric_name: str) -> list[tuple[str, float]]:
        """(system, value) for the model and every baseline reporting the metric."""
        out = [(self.model.name, self.model.metric(metric_name).value)]
        for b in self.baselines:
            try:
                out.append((b.name, b.metric(metric_name).value))
            except CardError:
                continue
        return out

    def render_markdown(self) -> str:
        lines: list[str] = []
        target = self.delta.describe() if self.delta else (
            self.fidelity.describe() if self.fidelity else "unspecified"
        )
        lines.append(f"# {self.model.name}")
        lines.append("")
        lines.append(f"- **Dataset**: `{self.dataset_name}` v`{self.dataset_version}`")
        lines.append(f"- **Content hash**: `{self.dataset_sha256[:16]}...`")
        lines.append(f"- **Split**: `{self.split.describe()}`")
        lines.append(f"- **Reported on**: `{self.model.partition}`")
        lines.append(f"- **Target**: {target}")
        lines.append(f"- **Created**: {self.created_utc}")
        if self.notes:
            lines.append(f"- **Notes**: {self.notes}")
        lines.append("")

        metric_names: list[str] = []
        for m in self.model.metrics:
            metric_names.append(m.name)
        header = ["system"] + [
            f"{n} ({self.model.metric(n).unit})" if self.model.metric(n).unit else n
            for n in metric_names
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        def row(ev: Evaluation, label: str) -> str:
            cells = [label]
            for n in metric_names:
                try:
                    cells.append(f"{ev.metric(n).value:.4g}")
                except CardError:
                    cells.append("--")
            return "| " + " | ".join(cells) + " |"

        lines.append(row(self.model, f"**{self.model.name}**"))
        for b in self.baselines:
            lines.append(row(b, b.name))
        lines.append("")
        for b in self.baselines:
            if b.description:
                lines.append(f"- `{b.name}`: {b.description}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.as_dict(),
            "baselines": [b.as_dict() for b in self.baselines],
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version,
                "sha256": self.dataset_sha256,
            },
            "split": {
                "id": self.split.id,
                "spec_id": self.split.spec.id,
                "strategy": self.split.spec.strategy,
                "counts": self.split.counts(),
            },
            "fidelity": self.fidelity.as_dict() if self.fidelity else None,
            "delta": (
                {"base": self.delta.base.as_dict(), "target": self.delta.target.as_dict()}
                if self.delta
                else None
            ),
            "notes": self.notes,
            "created_utc": self.created_utc,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)
