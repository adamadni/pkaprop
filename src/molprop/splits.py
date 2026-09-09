"""Splitting.

Project constraint: no random splits, ever. There is deliberately no
`random_split` in this module, and asking for one by name raises.

Two things this module enforces that a naive scaffold split does not:

1. **Group integrity.** One molecule contributes a record per ionisable site.
   The unit that must not straddle a fold is the record's `group_key`, and
   every assignment is checked against it.

2. **The right holdout unit.** Murcko scaffold is the wrong invariant for a
   local property. pKa is dominated by the environment of the ionisable site,
   so a molecule with a novel scaffold carrying a familiar aniline is trivially
   predicted and a scaffold split will read as generalisation when it is not.
   `reaction_centre_split` groups on the site environment instead.

Every split also reserves a `frozen_test` partition. Active learning shifts the
training distribution, which breaks the assumption that a held-out set stays
representative; the frozen partition is never available for AL selection.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from .chem import atom_environment, murcko_scaffold
from .records import PKaRecord, PropertyRecord

__all__ = [
    "SplitError",
    "LeakageError",
    "SplitSpec",
    "Split",
    "scaffold_split",
    "reaction_centre_split",
    "time_split",
    "assert_no_group_leakage",
    "reaction_centre_key",
]


class SplitError(ValueError):
    """Raised on an ill-formed or impossible split request."""


class LeakageError(AssertionError):
    """Raised when a group straddles a fold boundary."""


_PARTITIONS = ("train", "val", "test", "frozen_test")


@dataclass(frozen=True)
class SplitSpec:
    """Everything needed to reproduce a split, hashed into the split id."""

    strategy: str
    fractions: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [p for p in _PARTITIONS if p not in self.fractions]
        if missing:
            raise SplitError(f"fractions must cover every partition; missing {missing}")
        total = sum(self.fractions[p] for p in _PARTITIONS)
        if abs(total - 1.0) > 1e-9:
            raise SplitError(f"fractions must sum to 1.0, got {total}")
        if any(self.fractions[p] < 0 for p in _PARTITIONS):
            raise SplitError("fractions must be non-negative")

    @property
    def id(self) -> str:
        payload = {
            "strategy": self.strategy,
            "fractions": {k: round(float(v), 12) for k, v in sorted(self.fractions.items())},
            "params": self.params,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Split:
    """Index assignment over a fixed record ordering."""

    spec: SplitSpec
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    frozen_test: tuple[int, ...]
    #: the grouping actually used, index -> group key (kept for auditing)
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allocated = [*self.train, *self.val, *self.test, *self.frozen_test]
        if len(set(allocated)) != len(allocated):
            raise SplitError("a record index appears in more than one partition")

    @property
    def id(self) -> str:
        """Hash over the spec *and* the realised assignment.

        A result card is only comparable to a baseline card if these match, so
        the id has to depend on the actual indices, not just the recipe.
        """
        payload = {
            "spec": self.spec.id,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "frozen_test": list(self.frozen_test),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def counts(self) -> dict[str, int]:
        return {p: len(getattr(self, p)) for p in _PARTITIONS}

    def describe(self) -> str:
        c = self.counts()
        n = sum(c.values())
        parts = ", ".join(f"{p}={c[p]} ({c[p] / n:.1%})" if n else f"{p}=0" for p in _PARTITIONS)
        return f"{self.spec.strategy} [{self.id}] n={n}: {parts}"


# --------------------------------------------------------------------------
# Grouping keys
# --------------------------------------------------------------------------


def reaction_centre_key(record: PropertyRecord, radius: int = 2) -> str:
    """Grouping key describing the local chemistry the property depends on.

    For pKa that is the environment of the ionisable site. Murcko scaffold is
    the wrong invariant here: a molecule can change scaffold entirely while
    keeping the same aniline, and a scaffold split would then read as
    generalisation when the model has simply seen that site before.

    Only microscopic records name a site. A macroscopic value is a property of
    the whole titration, so it has no single centre and this raises rather than
    silently falling back to something coarser.
    """
    if isinstance(record, PKaRecord):
        if record.resolution != "micro":
            raise SplitError(
                "a macroscopic pKa record has no reaction centre; either resolve it "
                "to microstates first, or use scaffold_split / a molecule-level split"
            )
        return atom_environment(record.molecule.smiles, record.site_atom_idx, radius)

    raise SplitError(
        f"no reaction-centre definition for {type(record).__name__}; add one "
        f"rather than falling back to a scaffold split silently"
    )


def _scaffold_key(record: PropertyRecord) -> str:
    scaffold = murcko_scaffold(record.molecule.smiles)
    # Acyclic molecules all share the empty scaffold; keep them apart by
    # molecule so they do not collapse into one giant group.
    return scaffold or f"acyclic:{record.molecule.inchikey}"


# --------------------------------------------------------------------------
# Deterministic group allocation
# --------------------------------------------------------------------------


def _allocate(
    group_of: Sequence[str], fractions: dict[str, float]
) -> dict[str, list[int]]:
    """Assign whole groups to partitions, largest group first, no RNG.

    Each group goes to whichever partition is furthest below its quota, so the
    result is deterministic given the grouping and the fractions. A second pass
    guarantees that no partition with a non-zero fraction ends up empty -- an
    empty frozen_test would quietly remove the AL guard.
    """
    members: dict[str, list[int]] = {}
    for idx, key in enumerate(group_of):
        members.setdefault(key, []).append(idx)

    active = [p for p in _PARTITIONS if fractions[p] > 0]
    if not active:
        raise SplitError("every partition has a zero fraction")
    if len(members) < len(active):
        raise SplitError(
            f"{len(members)} groups cannot populate {len(active)} partitions "
            f"({', '.join(active)}); coarsen the grouping or drop a partition"
        )

    total = len(group_of)
    quota = {p: fractions[p] * total for p in active}
    filled = {p: 0 for p in active}
    assign: dict[str, str] = {}

    # Largest groups first, ties broken by key so the order is stable. Each
    # group goes to the partition with the largest *absolute* shortfall, which
    # drives every partition to its quota together; a relative-shortfall rule
    # lets the small partitions overshoot early on the biggest groups and never
    # recover. Ties resolve in _PARTITIONS order.
    priority = {p: -i for i, p in enumerate(_PARTITIONS)}
    for key, idxs in sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        target = max(active, key=lambda p: (quota[p] - filled[p], priority[p]))
        assign[key] = target
        filled[target] += len(idxs)

    # Guarantee pass: every active partition holds at least one group.
    for partition in active:
        if partition in assign.values():
            continue
        counts = Counter(assign.values())
        donor = max(counts, key=lambda d: (counts[d], d))
        if counts[donor] < 2:
            raise SplitError(
                f"cannot give {partition!r} a group without emptying {donor!r}; "
                f"too few groups for these fractions"
            )
        smallest = min(
            (k for k, a in assign.items() if a == donor),
            key=lambda k: (len(members[k]), k),
        )
        assign[smallest] = partition

    out: dict[str, list[int]] = {p: [] for p in _PARTITIONS}
    for key, partition in assign.items():
        out[partition].extend(members[key])
    for p in _PARTITIONS:
        out[p].sort()
    return out


def _build(
    records: Sequence[PropertyRecord],
    key_fn: Callable[[PropertyRecord], str],
    spec: SplitSpec,
    *,
    enforce_group_key: bool = True,
    max_collapse: float = 0.5,
) -> Split:
    if not records:
        raise SplitError("cannot split an empty record set")
    raw = [key_fn(r) for r in records]
    n_raw = len(set(raw))
    if enforce_group_key:
        # Grouping must never be finer than the record's own group_key, or a
        # molecule's records would scatter across folds. Merge any groups that
        # share a group_key (transitive closure).
        keys = _merge_by_group_key(records, raw)
        n_merged = len(set(keys))
        if n_merged < max(1.0, n_raw * max_collapse):
            raise SplitError(
                f"grouping collapsed from {n_raw} {spec.strategy} groups to "
                f"{n_merged} once molecule-level integrity was enforced. Each "
                f"molecule spans several centres, so holding out whole centre "
                f"classes and keeping every molecule intact are incompatible for "
                f"this data. Choose one deliberately: group_integrity='centre' "
                f"lets a molecule's records span folds (defensible only when the "
                f"property is genuinely site-local, e.g. microscopic pKa), or use "
                f"scaffold_split, which keeps molecules whole."
            )
    else:
        keys = tuple(raw)
    alloc = _allocate(keys, spec.fractions)
    split = Split(
        spec=spec,
        train=tuple(alloc["train"]),
        val=tuple(alloc["val"]),
        test=tuple(alloc["test"]),
        frozen_test=tuple(alloc["frozen_test"]),
        groups=tuple(keys),
    )
    if enforce_group_key:
        assert_no_group_leakage(records, split)
    return split


def _merge_by_group_key(
    records: Sequence[PropertyRecord], keys: Sequence[str]
) -> tuple[str, ...]:
    """Union-find over (chosen key, record group_key) so neither can straddle."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic merge direction.
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for record, key in zip(records, keys):
        union(f"k:{key}", f"g:{record.group_key}")
    return tuple(find(f"k:{k}") for k in keys)


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------

_DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.1, "frozen_test": 0.1}


def scaffold_split(
    records: Sequence[PropertyRecord],
    fractions: dict[str, float] | None = None,
) -> Split:
    """Bemis-Murcko scaffold split.

    Appropriate for whole-molecule properties. For pKa prefer
    `reaction_centre_split` -- see the module docstring.
    """
    spec = SplitSpec("scaffold", fractions or dict(_DEFAULT_FRACTIONS))
    return _build(records, _scaffold_key, spec)


def reaction_centre_split(
    records: Sequence[PropertyRecord],
    radius: int = 2,
    fractions: dict[str, float] | None = None,
    group_integrity: str = "molecule",
) -> Split:
    """Hold out whole classes of reaction-centre environment.

    `group_integrity` resolves a tension that has no universal right answer:

    * ``"molecule"`` (default) keeps every record of a molecule together. For a
      multi-site property like BDE this usually collapses the centre grouping --
      a molecule contains many distinct centres -- and the split then raises
      rather than silently degenerating into a molecule split.
    * ``"centre"`` holds out centre classes and lets a molecule's records span
      folds. Defensible only when the property is genuinely site-local, which is
      the case for microscopic pKa (one record per site) and not for BDE.
    """
    if radius < 1:
        raise SplitError("radius must be at least 1")
    if group_integrity not in ("molecule", "centre"):
        raise SplitError(
            f"group_integrity must be 'molecule' or 'centre', got {group_integrity!r}"
        )
    spec = SplitSpec(
        "reaction_centre",
        fractions or dict(_DEFAULT_FRACTIONS),
        {"radius": radius, "group_integrity": group_integrity},
    )
    return _build(
        records,
        lambda r: reaction_centre_key(r, radius),
        spec,
        enforce_group_key=group_integrity == "molecule",
    )


def time_split(
    records: Sequence[PropertyRecord],
    cutoffs: dict[str, str],
    fractions: dict[str, float] | None = None,
) -> Split:
    """Split by record date.

    `cutoffs` gives the exclusive upper bound (ISO-8601) for train and val;
    everything at or after the val cutoff is split between test and frozen_test
    by group, oldest first.
    """
    missing = [i for i, r in enumerate(records) if not r.date]
    if missing:
        raise SplitError(
            f"time_split needs a date on every record; {len(missing)} records "
            f"have none (first at index {missing[0]})"
        )
    for needed in ("train", "val"):
        if needed not in cutoffs:
            raise SplitError(f"cutoffs must give a bound for {needed!r}")
    if cutoffs["train"] > cutoffs["val"]:
        raise SplitError("train cutoff must not be after the val cutoff")

    spec = SplitSpec(
        "time", fractions or dict(_DEFAULT_FRACTIONS), {"cutoffs": dict(cutoffs)}
    )

    buckets: dict[str, list[int]] = {p: [] for p in _PARTITIONS}
    late: list[int] = []
    for i, r in enumerate(records):
        if r.date < cutoffs["train"]:
            buckets["train"].append(i)
        elif r.date < cutoffs["val"]:
            buckets["val"].append(i)
        else:
            late.append(i)

    # Split the late records between test and frozen_test, whole groups only.
    late_keys = [records[i].group_key for i in late]
    share = spec.fractions["test"] + spec.fractions["frozen_test"]
    if share <= 0:
        raise SplitError("time_split needs a non-zero test share")
    sub = _allocate(
        late_keys,
        {
            "train": 0.0,
            "val": 0.0,
            "test": spec.fractions["test"] / share,
            "frozen_test": spec.fractions["frozen_test"] / share,
        },
    )
    buckets["test"] = sorted(late[j] for j in sub["test"])
    buckets["frozen_test"] = sorted(late[j] for j in sub["frozen_test"])

    split = Split(
        spec=spec,
        train=tuple(buckets["train"]),
        val=tuple(buckets["val"]),
        test=tuple(buckets["test"]),
        frozen_test=tuple(buckets["frozen_test"]),
        groups=tuple(r.group_key for r in records),
    )
    assert_no_group_leakage(records, split)
    return split


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def assert_no_group_leakage(
    records: Sequence[PropertyRecord], split: Split
) -> None:
    """Raise if any record `group_key` appears in more than one partition."""
    seen: dict[str, str] = {}
    for partition in _PARTITIONS:
        for idx in getattr(split, partition):
            key = records[idx].group_key
            prior = seen.get(key)
            if prior is None:
                seen[key] = partition
            elif prior != partition:
                raise LeakageError(
                    f"group {key!r} appears in both {prior!r} and {partition!r}; "
                    f"a molecule's records must stay together"
                )
    allocated = sum(len(getattr(split, p)) for p in _PARTITIONS)
    if allocated != len(records):
        raise SplitError(
            f"split covers {allocated} of {len(records)} records; every record "
            f"must be assigned exactly once"
        )


def coverage_report(records: Sequence[PropertyRecord], split: Split) -> str:
    """Human-readable summary for a result card."""
    lines = [split.describe()]
    for partition in _PARTITIONS:
        idxs = getattr(split, partition)
        groups = {records[i].group_key for i in idxs}
        lines.append(f"  {partition:<12} records={len(idxs):>6}  groups={len(groups):>6}")
    return "\n".join(lines)


def __getattr__(name: str):
    if name in ("random_split", "shuffle_split", "train_test_split"):
        raise AttributeError(
            f"{name} does not exist by design. Random splits overstate performance "
            f"on molecular data; use scaffold_split, reaction_centre_split or "
            f"time_split."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
