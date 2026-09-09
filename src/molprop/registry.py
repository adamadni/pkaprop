"""Dataset registry.

A dataset is a name, a version, a content hash, and provenance. `load` verifies
the hash and refuses to return records that do not match the manifest, so a
result card can name a dataset version and mean something by it.

The registry also runs a units audit on write. It does not try to fix anything;
it records exactly which units, kinds, phases, temperatures and fidelities are
present, and refuses the write when a single dataset silently mixes conditions
that are not comparable -- pooled aqueous and methanol/water pKa being the
common case.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .fidelity import Fidelity
from .records import MoleculeRef, PKaRecord, PropertyRecord
from .units import Quantity

__all__ = ["RegistryError", "UnitsAudit", "DatasetManifest", "Registry"]


class RegistryError(RuntimeError):
    """Raised on a missing dataset, a hash mismatch, or a failed audit."""


_RECORD_TYPES = {
    "PKaRecord": PKaRecord,
    "PropertyRecord": PropertyRecord,
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compact(value):
    """Drop null fields recursively so a row stores only what it asserts."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _fidelity_table(records: Sequence[PropertyRecord]) -> dict[str, Any]:
    """Every distinct fidelity in the set, keyed by its id."""
    table: dict[str, Any] = {}
    for r in records:
        table.setdefault(r.fidelity.id, _compact(r.fidelity.as_dict()))
    return table


def _serialise(records: Sequence[PropertyRecord]) -> str:
    """Canonical JSONL, sorted so the hash is order-independent.

    The fidelity is stored once per dataset in the manifest and referenced by id
    from each row. A BDE dataset has one fidelity and half a million rows, so
    inlining it would spend most of the file restating the same 500 bytes.
    """
    rows = []
    for r in records:
        row = _compact(r.as_dict())
        row.pop("fidelity", None)
        row["fidelity_id"] = r.fidelity.id
        rows.append(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str))
    rows.sort()
    return "\n".join(rows) + ("\n" if rows else "")


def _record_from_dict(
    d: dict[str, Any], fidelities: dict[str, Fidelity] | None = None
) -> PropertyRecord:
    cls = _RECORD_TYPES.get(d.get("record_type", ""))
    if cls is None:
        raise RegistryError(f"unknown record_type {d.get('record_type')!r}")
    if "fidelity" in d:
        fidelity = Fidelity.from_dict(d["fidelity"])  # inline, v1 layout
    else:
        key = d.get("fidelity_id")
        if not fidelities or key not in fidelities:
            raise RegistryError(
                f"row references fidelity {key!r} but the manifest does not define "
                f"it; the manifest and records file have come apart"
            )
        fidelity = fidelities[key]
    common = dict(
        molecule=MoleculeRef(**d["molecule"]),
        quantity=Quantity.from_dict(d["quantity"]),
        fidelity=fidelity,
        source=d["source"],
        source_ref=d.get("source_ref"),
        uncertainty=Quantity.from_dict(d["uncertainty"]) if d.get("uncertainty") else None,
        date=d.get("date"),
        metadata=d.get("metadata") or {},
    )
    if cls is PKaRecord:
        return PKaRecord(
            resolution=d["resolution"],
            step_index=d.get("step_index"),
            site_atom_idx=d.get("site_atom_idx"),
            protonated_smiles=d.get("protonated_smiles"),
            deprotonated_smiles=d.get("deprotonated_smiles"),
            **common,
        )
    return PropertyRecord(**common)


@dataclass(frozen=True)
class UnitsAudit:
    """What units, kinds and conditions a dataset actually contains."""

    kinds: dict[str, int]
    units: dict[str, int]
    phases: dict[str, int]
    temperatures: dict[str, int]
    fidelities: dict[str, int]
    record_types: dict[str, int]

    @classmethod
    def of(cls, records: Sequence[PropertyRecord]) -> "UnitsAudit":
        return cls(
            kinds=dict(Counter(r.quantity.kind.value for r in records)),
            units=dict(Counter(r.quantity.unit.symbol for r in records)),
            phases=dict(Counter(str(r.quantity.phase) for r in records)),
            temperatures=dict(Counter(str(r.quantity.temperature_K) for r in records)),
            fidelities=dict(Counter(r.fidelity.id for r in records)),
            record_types=dict(Counter(type(r).__name__ for r in records)),
        )

    def problems(self) -> list[str]:
        """Conditions that make a dataset internally incomparable."""
        issues: list[str] = []
        if len(self.kinds) > 1:
            issues.append(f"mixed quantity kinds in one dataset: {sorted(self.kinds)}")
        solvents = {p for p in self.phases if p.startswith("solution")}
        base_solvents = {p.split("(")[1].split(",")[0].rstrip(")") for p in solvents}
        if len(base_solvents) > 1:
            issues.append(
                f"mixed solvents in one dataset: {sorted(base_solvents)}; "
                f"mixed-solvent values are not comparable to aqueous ones"
            )
        if len(self.temperatures) > 1:
            issues.append(f"mixed temperatures: {sorted(self.temperatures)}")
        return issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "kinds": self.kinds,
            "units": self.units,
            "phases": self.phases,
            "temperatures": self.temperatures,
            "fidelities": self.fidelities,
            "record_types": self.record_types,
        }


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    n_records: int
    content_sha256: str
    created_utc: str
    source_url: str | None = None
    doi: str | None = None
    license: str | None = None
    notes: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)
    #: id -> full fidelity spec, so rows reference it rather than repeating it
    fidelities: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "n_records": self.n_records,
            "content_sha256": self.content_sha256,
            "created_utc": self.created_utc,
            "source_url": self.source_url,
            "doi": self.doi,
            "license": self.license,
            "notes": self.notes,
            "audit": self.audit,
            "fidelities": self.fidelities,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetManifest":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class Registry:
    """On-disk dataset store: `<root>/<name>/<version>/{records.jsonl,manifest.json}`."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, name: str, version: str) -> Path:
        if not name or "/" in name or "\\" in name:
            raise RegistryError(f"invalid dataset name {name!r}")
        if not version or "/" in version or "\\" in version:
            raise RegistryError(f"invalid dataset version {version!r}")
        return self.root / name / version

    def write(
        self,
        name: str,
        version: str,
        records: Sequence[PropertyRecord],
        *,
        source_url: str | None = None,
        doi: str | None = None,
        license: str | None = None,
        notes: str | None = None,
        allow_mixed_conditions: bool = False,
        overwrite: bool = False,
    ) -> DatasetManifest:
        if not records:
            raise RegistryError("refusing to register an empty dataset")
        target = self._dir(name, version)
        if target.exists() and not overwrite:
            raise RegistryError(
                f"{name}/{version} already exists; bump the version rather than "
                f"mutating a registered dataset, or pass overwrite=True"
            )

        audit = UnitsAudit.of(records)
        problems = audit.problems()
        if problems and not allow_mixed_conditions:
            raise RegistryError(
                "units audit failed:\n  - "
                + "\n  - ".join(problems)
                + "\nSplit these into separate datasets, or pass "
                "allow_mixed_conditions=True if the mixing is deliberate."
            )

        payload = _serialise(records)
        manifest = DatasetManifest(
            name=name,
            version=version,
            n_records=len(records),
            content_sha256=_sha256_text(payload),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_url=source_url,
            doi=doi,
            license=license,
            notes=notes,
            audit=audit.as_dict(),
            fidelities=_fidelity_table(records),
        )
        target.mkdir(parents=True, exist_ok=True)
        (target / "records.jsonl").write_text(payload, encoding="utf-8")
        (target / "manifest.json").write_text(
            json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    def manifest(self, name: str, version: str) -> DatasetManifest:
        path = self._dir(name, version) / "manifest.json"
        if not path.exists():
            raise RegistryError(f"no manifest for {name}/{version} at {path}")
        return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load(
        self, name: str, version: str, *, verify: bool = True
    ) -> tuple[list[PropertyRecord], DatasetManifest]:
        manifest = self.manifest(name, version)
        path = self._dir(name, version) / "records.jsonl"
        if not path.exists():
            raise RegistryError(f"no records for {name}/{version} at {path}")
        payload = path.read_text(encoding="utf-8")
        if verify:
            digest = _sha256_text(payload)
            if digest != manifest.content_sha256:
                raise RegistryError(
                    f"content hash mismatch for {name}/{version}: manifest says "
                    f"{manifest.content_sha256}, file hashes to {digest}. The data "
                    f"changed underneath the manifest; do not train on it."
                )
        fidelities = {
            key: Fidelity.from_dict(spec)
            for key, spec in (manifest.fidelities or {}).items()
        }
        records = [
            _record_from_dict(json.loads(line), fidelities)
            for line in payload.splitlines()
            if line.strip()
        ]
        if len(records) != manifest.n_records:
            raise RegistryError(
                f"{name}/{version}: manifest claims {manifest.n_records} records, "
                f"file has {len(records)}"
            )
        return records, manifest

    def datasets(self) -> list[tuple[str, str]]:
        if not self.root.exists():
            return []
        out = []
        for name_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for version_dir in sorted(p for p in name_dir.iterdir() if p.is_dir()):
                if (version_dir / "manifest.json").exists():
                    out.append((name_dir.name, version_dir.name))
        return out
