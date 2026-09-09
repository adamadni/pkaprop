"""Homolytic bond dissociation.

    A-B  ->  A.  +  B.        BDE = H(A.) + H(B.) - H(A-B)

All three terms are gas-phase enthalpies at the same temperature, which is the
convention ALFABET and the experimental compilations use. The units layer
enforces that: mixing a solution-phase fragment into this sum raises.

The recurring way this calculation goes quietly wrong is spin. Both fragments
are open shell, and an engine handed multiplicity 1 for a radical will converge
happily to a closed-shell solution that is tens of kcal/mol too low. Fragment
multiplicity is therefore derived from the radical electron count here and
checked against the electron parity in `geometry`, never defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from rdkit import Chem

from .calc import Task
from .chem import BondRef, MoleculeError, indexed_mol
from .fidelity import ConformerProtocol, Fidelity
from .records import BDERecord, MoleculeRef
from .units import GAS, KCAL_PER_MOL, Quantity, QuantityKind, UnitError

__all__ = [
    "BDEError",
    "Fragment",
    "homolysis_fragments",
    "bde_from_enthalpies",
    "compute_bde",
]


class BDEError(ValueError):
    """Raised when a bond cannot be homolysed or a BDE cannot be assembled."""


@dataclass(frozen=True)
class Fragment:
    """One homolysis product, with the spin state its radical count implies."""

    molecule: MoleculeRef
    multiplicity: int
    charge: int = 0
    n_radical_electrons: int = 1

    @property
    def is_radical(self) -> bool:
        return self.n_radical_electrons > 0

    def describe(self) -> str:
        return f"{self.molecule.smiles} (mult={self.multiplicity}, q={self.charge})"


def homolysis_fragments(
    molecule: MoleculeRef | str, bond: BondRef
) -> tuple[Fragment, Fragment]:
    """Split a bond homolytically into two radical fragments.

    Atom indices refer to the indexed (explicit-H) mol, matching every other
    atom index in this package.
    """
    if isinstance(molecule, str):
        molecule = MoleculeRef.from_smiles(molecule)
    mol = indexed_mol(molecule.smiles)
    i, j = bond.as_tuple()
    target = mol.GetBondBetweenAtoms(i, j)
    if target is None:
        raise MoleculeError(f"no bond between atoms {i} and {j} in {molecule.smiles}")
    if target.IsInRing():
        raise BDEError(
            f"bond ({i}, {j}) in {molecule.smiles} is in a ring; breaking it gives "
            f"one diradical, not two fragments, so it has no bond dissociation "
            f"enthalpy in the usual sense"
        )
    if Chem.GetFormalCharge(mol) != 0:
        raise BDEError(
            "homolysis of a charged species is not a plain BDE; the fragments' "
            "charges are not determined by the bond alone"
        )

    editable = Chem.RWMol(mol)
    editable.RemoveBond(i, j)
    for idx in (i, j):
        atom = editable.GetAtomWithIdx(idx)
        atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
        atom.SetNoImplicit(True)

    broken = editable.GetMol()
    try:
        Chem.SanitizeMol(broken)
    except Exception as exc:  # pragma: no cover - RDKit valence edge cases
        raise BDEError(f"could not sanitise the homolysed structure: {exc}") from None

    pieces = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    if len(pieces) != 2:
        raise BDEError(
            f"breaking bond ({i}, {j}) gave {len(pieces)} fragment(s), not 2; a "
            f"ring bond does not dissociate into two species and needs a "
            f"different treatment"
        )

    fragments = []
    for piece in pieces:
        radicals = sum(a.GetNumRadicalElectrons() for a in piece.GetAtoms())
        if radicals == 0:
            raise BDEError("a homolysis fragment carries no radical electron")
        smiles = Chem.MolToSmiles(piece, canonical=True)
        fragments.append(
            Fragment(
                molecule=MoleculeRef.from_smiles(smiles),
                # One unpaired electron per radical centre; a doublet for the
                # usual single-bond case.
                multiplicity=radicals + 1,
                charge=Chem.GetFormalCharge(piece),
                n_radical_electrons=radicals,
            )
        )
    return fragments[0], fragments[1]


def bde_from_enthalpies(
    parent: Quantity, fragment_a: Quantity, fragment_b: Quantity
) -> Quantity:
    """Assemble a BDE from three gas-phase enthalpies.

    Returns a quantity of kind BDE, which the record layer requires and which
    cannot be silently added to an arbitrary reaction enthalpy.
    """
    for name, q in (("parent", parent), ("fragment_a", fragment_a), ("fragment_b", fragment_b)):
        if q.kind is not QuantityKind.ENTHALPY:
            raise UnitError(
                f"{name} must be an enthalpy for a BDE, got {q.kind.value}; an "
                f"electronic energy omits ZPE and the thermal correction, which "
                f"are worth several kcal/mol for an X-H bond"
            )
        if q.phase.medium != "gas":
            raise UnitError(f"{name} is {q.phase}; BDE is a gas-phase quantity")
    # Subtraction already enforces matching phase and temperature.
    total = (fragment_a + fragment_b) - parent
    return Quantity(
        total.to(KCAL_PER_MOL).value,
        KCAL_PER_MOL,
        QuantityKind.BDE,
        GAS,
        parent.temperature_K,
    )


def compute_bde(
    molecule: MoleculeRef | str,
    bond: BondRef,
    fidelity: Fidelity,
    engine,
    *,
    protocol: ConformerProtocol | None = None,
    temperature_K: float = 298.15,
    task: Task = Task.OPT_FREQ,
) -> dict:
    """Run the full parent + two fragments calculation and assemble the BDE.

    Returns a dict carrying the BDE, the three results, and the summed resource
    usage, so cost per BDE is visible from the first molecule rather than
    discovered at scale.
    """
    from .conformers import generate_conformers
    from .thermo import ensemble_quantity

    if isinstance(molecule, str):
        molecule = MoleculeRef.from_smiles(molecule)
    if not task.yields_thermochemistry:
        raise BDEError(
            f"task {task.value} produces no enthalpy; a BDE needs a Hessian"
        )
    protocol = protocol or fidelity.conformers
    treatment = protocol.ensemble_treatment

    frag_a, frag_b = homolysis_fragments(molecule, bond)

    def evaluate(ref: MoleculeRef, multiplicity: int, charge: int):
        ensemble = generate_conformers(
            ref, protocol, charge=charge, multiplicity=multiplicity
        )
        results = engine.run(ensemble, fidelity, task, temperature_K=temperature_K)
        return results, ensemble_quantity(results, treatment=treatment, temperature_K=temperature_K)

    parent_results, parent_thermo = evaluate(molecule, 1, 0)
    a_results, a_thermo = evaluate(frag_a.molecule, frag_a.multiplicity, frag_a.charge)
    b_results, b_thermo = evaluate(frag_b.molecule, frag_b.multiplicity, frag_b.charge)

    for label, thermo in (
        ("parent", parent_thermo),
        (frag_a.molecule.smiles, a_thermo),
        (frag_b.molecule.smiles, b_thermo),
    ):
        if thermo.enthalpy is None:
            raise BDEError(f"{label} produced no enthalpy; cannot assemble a BDE")

    value = bde_from_enthalpies(
        parent_thermo.enthalpy, a_thermo.enthalpy, b_thermo.enthalpy
    )

    usage = parent_results[0].usage
    for r in [*parent_results[1:], *a_results, *b_results]:
        usage = usage + r.usage

    return {
        "bde": value,
        "fragments": (frag_a, frag_b),
        "parent": parent_thermo,
        "fragment_a": a_thermo,
        "fragment_b": b_thermo,
        "usage": usage,
        "n_jobs": len(parent_results) + len(a_results) + len(b_results),
    }


def bde_record(
    molecule: MoleculeRef,
    bond: BondRef,
    value: Quantity,
    fidelity: Fidelity,
    *,
    source: str,
    **kwargs,
) -> BDERecord:
    """Wrap a computed BDE in a registrable record."""
    return BDERecord(
        molecule=molecule,
        quantity=value,
        fidelity=fidelity,
        source=source,
        bond=bond,
        **kwargs,
    )
