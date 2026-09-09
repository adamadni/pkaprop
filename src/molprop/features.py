"""Graph featurisation, including GFN2-xTB electronic descriptors.

The graph is over **heavy atoms only**; hydrogens enter as an atom feature
(`GetTotalNumHs`). Ionisable sites are heavy atoms, so this keeps graphs small
without losing the thing being predicted.

Physics-informed features
-------------------------
Two GFN2-xTB quantities are attached, both from a single point on one
force-field conformer (~0.15 s per molecule):

* **Mulliken partial charge** on each atom. Acidity is largely the question of
  how well the conjugate base stabilises negative charge, so the charge already
  present at a site is a direct prior on that.
* **Wiberg bond orders**, summed per atom into a total bond order. This
  measures delocalisation: a carboxyl oxygen with substantial partial double
  bond character behaves very differently from an isolated alkoxide.

Both are optional. `featurise` works without them and zero-fills those columns,
so the model can be trained and compared with and without the physics -- which
is the only way to know whether the physics is earning its cost.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from .chem import MoleculeError, canonical_smiles

__all__ = [
    "MolGraph",
    "ATOM_FEATURES",
    "BOND_FEATURES",
    "XTBDescriptors",
    "XTBDescriptorCache",
    "featurise",
]

#: Elements the model knows. Anything else lands in the "other" slot rather
#: than silently sharing an embedding with carbon.
ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se")
HYBRIDISATIONS = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)

ATOM_FEATURES = (
    len(ELEMENTS) + 1        # element one-hot + other
    + 6                      # degree 0..5
    + 5                      # formal charge -2..+2
    + 5                      # total H 0..4
    + len(HYBRIDISATIONS) + 1
    + 1                      # aromatic
    + 1                      # in ring
    + 1                      # xTB partial charge
    + 1                      # xTB total Wiberg bond order
)
BOND_FEATURES = (len(BOND_TYPES) + 1) + 3  # type one-hot + other, conjugated, ring, wbo


def _one_hot(value, choices) -> list[float]:
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1.0
    return out


def _clamped_one_hot(value: int, low: int, high: int) -> list[float]:
    out = [0.0] * (high - low + 1)
    out[min(max(value, low), high) - low] = 1.0
    return out


@dataclass(frozen=True)
class XTBDescriptors:
    """Per-atom and per-bond GFN2-xTB quantities for one molecule."""

    charges: tuple[float, ...]
    #: (atom_i, atom_j) -> Wiberg bond order, heavy-atom indices
    bond_orders: dict[tuple[int, int], float]

    def total_bond_order(self, n_atoms: int) -> np.ndarray:
        totals = np.zeros(n_atoms, dtype=np.float32)
        for (i, j), value in self.bond_orders.items():
            if i < n_atoms:
                totals[i] += value
            if j < n_atoms:
                totals[j] += value
        return totals


@dataclass(frozen=True)
class MolGraph:
    """A molecule as tensors, plus the bookkeeping the task needs."""

    smiles: str
    atom_features: np.ndarray      # (n_atoms, ATOM_FEATURES)
    edge_index: np.ndarray         # (2, n_edges), both directions
    edge_features: np.ndarray      # (n_edges, BOND_FEATURES)
    #: heavy-atom indices that can gain or lose a proton
    acid_sites: tuple[int, ...]
    base_sites: tuple[int, ...]

    @property
    def n_atoms(self) -> int:
        return int(self.atom_features.shape[0])


# --------------------------------------------------------------------------
# xTB descriptors
# --------------------------------------------------------------------------


class XTBDescriptorCache:
    """Runs GFN2-xTB single points and caches the results on disk.

    Keyed by canonical SMILES, so a molecule appearing in several records is
    computed once. The cache is plain JSON: it must stay inspectable, because
    a silent change in these numbers would move every downstream result.
    """

    #: One conformer is enough. Mulliken charges and Wiberg bond orders are
    #: dominated by connectivity and are only weakly conformer-dependent, while
    #: generating and force-field-optimising five conformers costs more than the
    #: xTB call itself. This is a deliberate accuracy/throughput trade and it is
    #: part of the cache key, so revisiting it cannot silently reuse old values.
    PROTOCOL = {
        "generator": "rdkit_etkdgv3",
        "ensemble_treatment": "lowest",
        "n_confs_requested": 1,
        "optimizer": "mmff94s",
        "seed": 42,
        "method": "GFN2-xTB",
        "task": "single_point",
    }

    def __init__(self, root: str | Path, executable: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executable = executable or shutil.which("xtb") or "xtb"
        self.protocol_id = hashlib.sha256(
            json.dumps(self.PROTOCOL, sort_keys=True).encode()
        ).hexdigest()[:8]
        self.hits = 0
        self.misses = 0
        self.failures = 0

    def available(self) -> bool:
        return shutil.which(self.executable) is not None or Path(self.executable).exists()

    def _path(self, smiles: str) -> Path:
        """Cache path keyed on BOTH the molecule and the protocol.

        Keying on SMILES alone would let a changed conformer or method setting
        silently reuse descriptors computed under the old one.
        """
        digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:16]
        return self.root / self.protocol_id / f"{digest}.json"

    def get(self, smiles: str) -> XTBDescriptors | None:
        canonical = canonical_smiles(smiles)
        path = self._path(canonical)
        if path.exists():
            self.hits += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload is None:
                return None
            return XTBDescriptors(
                charges=tuple(payload["charges"]),
                bond_orders={
                    (int(a), int(b)): float(v) for a, b, v in payload["bond_orders"]
                },
            )
        self.misses += 1
        result = self._compute(canonical)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                None
                if result is None
                else {
                    "charges": list(result.charges),
                    "bond_orders": [[a, b, v] for (a, b), v in result.bond_orders.items()],
                }
            ),
            encoding="utf-8",
        )
        if result is None:
            self.failures += 1
        return result

    def _compute(self, canonical: str) -> XTBDescriptors | None:
        from .conformers import generate_conformers
        from .fidelity import ConformerProtocol

        protocol = ConformerProtocol(
            generator=self.PROTOCOL["generator"],
            ensemble_treatment=self.PROTOCOL["ensemble_treatment"],
            n_confs_requested=self.PROTOCOL["n_confs_requested"],
            optimizer=self.PROTOCOL["optimizer"],
            seed=self.PROTOCOL["seed"],
        )
        try:
            ensemble = generate_conformers(canonical, protocol)
        except Exception:
            return None

        mol = Chem.MolFromSmiles(canonical)
        n_heavy = mol.GetNumAtoms()
        charge = Chem.GetFormalCharge(mol)
        with tempfile.TemporaryDirectory(prefix="xtbdesc_") as tmp:
            work = Path(tmp)
            (work / "in.xyz").write_text(ensemble.to_xyz(0), encoding="utf-8")
            try:
                subprocess.run(
                    [self.executable, "in.xyz", "--gfn", "2", "--chrg", str(charge),
                     "--uhf", "0", "--sp", "--parallel", "1"],
                    cwd=work, capture_output=True, text=True, timeout=120,
                    encoding="utf-8", errors="replace",
                )
            except Exception:
                return None
            charges_file, wbo_file = work / "charges", work / "wbo"
            if not charges_file.exists():
                return None
            values = [float(x) for x in charges_file.read_text().split()]
            # The xyz is the explicit-H mol; heavy atoms come first because
            # AddHs appends hydrogens after the heavy block.
            if len(values) < n_heavy:
                return None
            bond_orders: dict[tuple[int, int], float] = {}
            if wbo_file.exists():
                for line in wbo_file.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                    i, j, v = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                    if i < n_heavy and j < n_heavy:
                        bond_orders[(min(i, j), max(i, j))] = v
            return XTBDescriptors(tuple(values[:n_heavy]), bond_orders)


# --------------------------------------------------------------------------
# Featurisation
# --------------------------------------------------------------------------


def featurise(smiles: str, descriptors: XTBDescriptors | None = None) -> MolGraph:
    """Build the graph tensors for one molecule."""
    from .enumeration import find_protonation_sites

    canonical = canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise MoleculeError(f"could not parse {smiles!r}")
    n = mol.GetNumAtoms()

    charges = np.zeros(n, dtype=np.float32)
    totals = np.zeros(n, dtype=np.float32)
    if descriptors is not None:
        supplied = np.asarray(descriptors.charges, dtype=np.float32)
        charges[: min(n, len(supplied))] = supplied[:n]
        totals = descriptors.total_bond_order(n)

    rows = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        rows.append(
            _one_hot(atom.GetSymbol(), ELEMENTS)
            + _clamped_one_hot(atom.GetDegree(), 0, 5)
            + _clamped_one_hot(atom.GetFormalCharge(), -2, 2)
            + _clamped_one_hot(atom.GetTotalNumHs(), 0, 4)
            + _one_hot(atom.GetHybridization(), HYBRIDISATIONS)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
            + [float(charges[i]), float(totals[i])]
        )
    atom_matrix = np.asarray(rows, dtype=np.float32)

    src, dst, edge_rows = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        wbo = 0.0
        if descriptors is not None:
            wbo = descriptors.bond_orders.get((min(i, j), max(i, j)), 0.0)
        feature = (
            _one_hot(bond.GetBondType(), BOND_TYPES)
            + [float(bond.GetIsConjugated()), float(bond.IsInRing()), float(wbo)]
        )
        # Undirected graph stored as two directed edges.
        src += [i, j]
        dst += [j, i]
        edge_rows += [feature, feature]

    edge_index = np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)
    edge_matrix = (
        np.asarray(edge_rows, dtype=np.float32)
        if edge_rows
        else np.zeros((0, BOND_FEATURES), np.float32)
    )

    sites = find_protonation_sites(canonical)
    return MolGraph(
        smiles=canonical,
        atom_features=atom_matrix,
        edge_index=edge_index,
        edge_features=edge_matrix,
        acid_sites=tuple(sorted({s.atom_idx for s in sites if s.kind == "acid"})),
        base_sites=tuple(sorted({s.atom_idx for s in sites if s.kind == "base"})),
    )
