"""Conformer generation.

Stage one of the QM pipeline. The protocol that produced an ensemble is part of
the fidelity, so this module takes a `ConformerProtocol` and refuses to deviate
from it silently: an unseeded stochastic generator is already rejected at
protocol construction, and everything here is a deterministic function of
(canonical SMILES, protocol).

Determinism is a step-2 validation gate, not an aspiration -- see
`assert_deterministic`.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from .chem import indexed_mol
from .fidelity import ConformerProtocol, FidelityError
from .geometry import Conformer, ConformerEnsemble, GeometryError
from .records import MoleculeRef
from .units import GAS, KCAL_PER_MOL, Quantity, QuantityKind

__all__ = [
    "ConformerGenerationError",
    "generate_conformers",
    "assert_deterministic",
    "heavy_atom_rmsd",
    "conformer_rmsd",
]

#: Force fields this module knows how to run, mapped to their RDKit entry point.
_FORCE_FIELDS = {"mmff94", "mmff94s", "uff"}


class ConformerGenerationError(RuntimeError):
    """Raised when embedding or optimisation fails for a molecule."""


def _embed(mol: Chem.Mol, protocol: ConformerProtocol) -> Chem.Mol:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(protocol.seed)
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    # Deliberately no embedding-time pruning: `rmsd_prune_angstrom` describes
    # the FINAL ensemble, and distinct embeddings routinely relax to distinct
    # minima. Pruning here would discard them before the force field has had a
    # chance to separate them, and would make the protocol field mean two
    # different things at two different stages.
    params.pruneRmsThresh = -1.0
    # Single-threaded on purpose: numThreads > 1 makes the result depend on
    # thread scheduling, which would break the determinism gate.
    params.numThreads = 1
    n = int(protocol.n_confs_requested or 1)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    if not len(ids):
        # Retry once with random coordinates, which rescues most hard cases.
        params.useRandomCoords = True
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params)
    if not len(ids):
        raise ConformerGenerationError(
            f"ETKDGv3 could not embed any conformer (seed={protocol.seed})"
        )
    return mol


def _optimise(mol: Chem.Mol, force_field: str) -> tuple[list[float], list[bool]]:
    """Optimise every conformer in place; return (energies kcal/mol, converged)."""
    ff = force_field.lower()
    if ff not in _FORCE_FIELDS:
        raise ConformerGenerationError(
            f"unknown force field {force_field!r}; known: {sorted(_FORCE_FIELDS)}"
        )
    if ff == "uff":
        results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=2000)
    else:
        variant = "MMFF94s" if ff == "mmff94s" else "MMFF94"
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if props is None:
            raise ConformerGenerationError(
                f"{variant} has no parameters for this molecule; fall back to UFF "
                f"explicitly rather than letting the protocol drift"
            )
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, maxIters=2000, mmffVariant=variant
        )
    converged = [code == 0 for code, _ in results]
    energies = [float(energy) for _, energy in results]  # RDKit reports kcal/mol
    return energies, converged


def conformer_rmsd(
    mol: Chem.Mol, conf_i: int, conf_j: int, *, atoms: str = "heavy"
) -> float:
    """Symmetry-aware RMSD between two conformers of one mol.

    `atoms="heavy"` is the conventional choice but is blind to O-H and N-H
    rotamers; `atoms="all"` keeps them apart.
    """
    if atoms == "heavy":
        target = Chem.RemoveHs(Chem.Mol(mol))
    elif atoms == "all":
        target = Chem.Mol(mol)
    else:
        raise ConformerGenerationError(f"atoms must be 'heavy' or 'all', got {atoms!r}")
    return float(rdMolAlign.GetBestRMS(target, target, prbId=conf_i, refId=conf_j))


def heavy_atom_rmsd(mol: Chem.Mol, conf_i: int, conf_j: int) -> float:
    """Backwards-compatible alias for the heavy-atom case."""
    return conformer_rmsd(mol, conf_i, conf_j, atoms="heavy")


def _prune(
    mol: Chem.Mol, order: list[int], threshold: float, atoms: str
) -> list[int]:
    """Greedy RMSD pruning over conformer ids, lowest energy first."""
    kept: list[int] = []
    for cid in order:
        if all(
            conformer_rmsd(mol, cid, other, atoms=atoms) > threshold for other in kept
        ):
            kept.append(cid)
    return kept


def generate_conformers(
    molecule: MoleculeRef | str,
    protocol: ConformerProtocol,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> ConformerEnsemble:
    """Generate an ensemble following `protocol` exactly.

    The returned ensemble carries force-field energies of kind
    FORCE_FIELD_ENERGY, which by construction cannot be mixed with the
    semiempirical or DFT energies that later stages attach.
    """
    if isinstance(molecule, str):
        molecule = MoleculeRef.from_smiles(molecule)
    if protocol.generator != "rdkit_etkdgv3":
        raise FidelityError(
            f"generate_conformers implements 'rdkit_etkdgv3', not "
            f"{protocol.generator!r}; CREST ensembles come from the remote engine"
        )
    if protocol.seed is None:
        raise FidelityError("a reproducible ensemble needs an explicit seed")

    mol = indexed_mol(molecule.smiles)
    elements = tuple(a.GetSymbol() for a in mol.GetAtoms())

    if mol.GetNumAtoms() == 1:
        # A lone atom has no conformers and no force-field parameters. This is
        # not a degenerate case to guard against -- it is the hydrogen atom
        # produced by every X-H homolysis, so it has to work.
        return ConformerEnsemble(
            molecule=molecule,
            elements=elements,
            conformers=[
                Conformer(
                    coords=np.zeros((1, 3)),
                    energy=Quantity(
                        0.0, KCAL_PER_MOL, QuantityKind.FORCE_FIELD_ENERGY, GAS,
                        temperature_K=None,
                    ),
                    converged=True,
                    label="atom",
                )
            ],
            protocol=protocol,
            charge=charge,
            multiplicity=multiplicity,
            metadata={"n_embedded": 1, "n_after_prune": 1, "monatomic": True},
        )

    _embed(mol, protocol)

    if protocol.optimizer:
        energies, converged = _optimise(mol, protocol.optimizer)
    else:
        energies = [0.0] * mol.GetNumConformers()
        converged = [True] * mol.GetNumConformers()

    ids = [c.GetId() for c in mol.GetConformers()]
    index_of = {cid: i for i, cid in enumerate(ids)}
    order = sorted(ids, key=lambda cid: (energies[index_of[cid]], cid))

    if protocol.rmsd_prune_angstrom is not None and len(order) > 1:
        order = _prune(
            mol, order, float(protocol.rmsd_prune_angstrom), protocol.rmsd_atoms
        )

    conformers = []
    for cid in order:
        idx = index_of[cid]
        conformers.append(
            Conformer(
                coords=np.array(mol.GetConformer(cid).GetPositions(), dtype=float),
                energy=Quantity(
                    energies[idx],
                    KCAL_PER_MOL,
                    QuantityKind.FORCE_FIELD_ENERGY,
                    GAS,
                    temperature_K=None,
                ),
                converged=converged[idx],
                label=f"etkdgv3-{cid}",
            )
        )

    ensemble = ConformerEnsemble(
        molecule=molecule,
        elements=elements,
        conformers=conformers,
        protocol=protocol,
        charge=charge,
        multiplicity=multiplicity,
        metadata={
            "n_embedded": len(ids),
            "n_after_prune": len(order),
            "force_field": protocol.optimizer,
            "rmsd_atoms": protocol.rmsd_atoms,
        },
    )

    if protocol.energy_window_kcal_per_mol is not None:
        ensemble = ensemble.within_window(float(protocol.energy_window_kcal_per_mol))

    if protocol.ensemble_treatment == "lowest" and len(ensemble) > 1:
        lowest = ensemble.lowest()
        ensemble = ConformerEnsemble(
            molecule=ensemble.molecule,
            elements=ensemble.elements,
            conformers=[lowest],
            protocol=protocol,
            charge=charge,
            multiplicity=multiplicity,
            metadata=dict(ensemble.metadata),
        )
    return ensemble


def assert_deterministic(
    molecule: MoleculeRef | str,
    protocol: ConformerProtocol,
    *,
    repeats: int = 3,
    tolerance_angstrom: float = 1e-6,
) -> None:
    """Validation gate: the same input must give byte-identical geometries.

    Run before any large campaign. A failure here means the fidelity id is
    lying -- two runs with the same recorded protocol produced different
    structures, so the energies attached to them are not comparable.
    """
    if repeats < 2:
        raise ValueError("determinism needs at least two runs to compare")
    reference = generate_conformers(molecule, protocol)
    for attempt in range(1, repeats):
        other = generate_conformers(molecule, protocol)
        if len(other) != len(reference):
            raise GeometryError(
                f"run {attempt} produced {len(other)} conformers, reference "
                f"produced {len(reference)}; the protocol is not deterministic"
            )
        for i, (a, b) in enumerate(zip(reference, other)):
            delta = float(np.abs(a.coords - b.coords).max())
            if delta > tolerance_angstrom:
                raise GeometryError(
                    f"run {attempt}, conformer {i}: coordinates differ by "
                    f"{delta:.3e} A (tolerance {tolerance_angstrom:.1e}); "
                    f"the protocol is not deterministic"
                )
