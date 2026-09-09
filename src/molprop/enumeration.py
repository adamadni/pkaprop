"""Protonation microstate enumeration.

Every pKa target -- monoprotic site labels, latent-site training, or plain
macroscopic regression -- needs to know which sites a molecule has. The SMARTS
screen in `curation` deliberately over-reports and is only a curation aid; this
module is the real thing, and its performance is measured against SAMPL6's
published enumeration rather than asserted.

**Measured against SAMPL6** (24 molecules, `physical_properties/pKa/microstates`):

===========================================  ==============  ==================
comparison                                   exact SMILES    species (InChI)
===========================================  ==============  ==================
microstates in an experimental pKa transition   13/20 (65%)     19/20 (95%)
SAMPL6's full published enumeration            129/351 (37%)    92/144 (64%)
===========================================  ==============  ==================

The two columns differ because SAMPL6 enumerates Lewis structures, and its 351
microstates are only **144 distinct chemical species** -- the rest are tautomer
and resonance forms of the same protonation level. Comparing by exact SMILES
therefore measures agreement about notation as much as about chemistry. See
`species_key`.

What this enumerator does and does not do:

* It enumerates **protonation states**: which heavy atoms bear how many
  protons, within a charge window.
* It does **not** enumerate tautomers. Every experimentally-relevant structure
  it fails to reproduce by exact SMILES is a tautomer or resonance form of a
  state it does generate -- verified, not assumed. RDKit's TautomerEnumerator
  does not bridge the important cases here (it finds 1-2 tautomers for a
  4-aminoquinazoline cation and never reaches the imino form), so composing
  with it does not help; `species_key` is the right tool instead.

Two failure modes, with very different costs:

* A **missed** site removes a real equilibrium from the model's hypothesis
  space. Nothing downstream can recover it.
* A **spurious** site adds a microstate that gets a low predicted population.
  Costly in compute, not in correctness.

The defaults therefore lean towards over-generation.

**Known limitation.** 69 of the 5,994 czodrowskilab compounds (1.2%) have a
measured pKa but no detected site. The remainder are mostly carbon acids --
pyrazolones, alpha-nitroso ketones, HCN -- whose acidity comes from
delocalisation into an adjacent activating group rather than from a
heteroatom-H bond. Detecting those needs an electronic argument, not a SMARTS
pattern, and they are excluded rather than guessed at.

Counting caution: this module answers "which atoms can gain or lose a proton",
not "how many equilibria titrate between pH 2 and 12". It is deliberately
permissive, so it is the wrong instrument for counting titratable centres --
it calls 222 of the 280 Novartis compounds polyprotic where the source curated
them as monoprotic. `curation.ionisable_sites` narrows it a little by reporting
one label per atom, but that count remains an upper bound.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

from .chem import MoleculeError, canonical_smiles

__all__ = [
    "ProtonationSite",
    "EnumeratedState",
    "largest_fragment",
    "find_protonation_sites",
    "find_protonation_sites_on_mol",
    "enumerate_microstates",
    "species_key",
    "to_ensemble",
]


@dataclass(frozen=True)
class ProtonationSite:
    """One atom that can gain or lose a proton."""

    atom_idx: int
    kind: str  # "acid" (can lose H+) | "base" (can gain H+)
    label: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.label}@{self.atom_idx}({self.kind})"


@dataclass(frozen=True)
class EnumeratedState:
    """A generated protonation state.

    Deliberately NOT called `Microstate`: `microstates.Microstate` already means
    something different -- it carries an *absolute* proton count for partition
    -function grouping, whereas `proton_delta` here is signed and relative to
    the input structure. Feeding one straight into the other is a real bug, so
    the two types have different names and `to_ensemble` does the conversion.
    """

    smiles: str
    charge: int
    #: protons gained (+) or lost (-) relative to the input structure
    proton_delta: int
    #: (atom_idx, delta) pairs applied to the reference structure
    changes: tuple[tuple[int, int], ...] = ()


# --------------------------------------------------------------------------
# Site definitions
#
# Each entry maps a SMARTS to the index (within the match) of the atom that
# actually gains or loses the proton. Ordered from strong to weak so that a
# label collision resolves to the more specific group.
# --------------------------------------------------------------------------

_ACID_PATTERNS: tuple[tuple[str, str, int], ...] = (
    ("carboxylic_acid", "[OX2H1][CX3]=[OX1]", 0),
    ("sulfonic_acid", "[OX2H1][SX4](=[OX1])(=[OX1])", 0),
    ("phosphonic_acid", "[OX2H1][PX4]=[OX1]", 0),
    ("tetrazole", "[nX3H1]1nnnc1", 0),
    ("sulfonamide_NH", "[NX3;H1,H2][SX4](=[OX1])=[OX1]", 0),
    ("acyl_sulfonamide", "[NX3;H1][CX3]=[OX1]", 0),
    ("azole_NH", "[nX3H1]", 0),
    ("thiol", "[SX2H1]", 0),
    ("phenol", "[OX2H1]c", 0),
    ("enol", "[OX2H1][CX3]=[CX3]", 0),
    # O-H bonded to nitrogen: hydroxamic acids and N-hydroxy compounds titrate
    # around pKa 9 and were invisible to the [OX2H1][CX4] alcohol pattern.
    ("hydroxamic_or_hydroxylamine_OH", "[OX2H1][NX3]", 0),
    ("peroxy_acid", "[OX2H1][OX2][CX3]=[OX1]", 0),
    ("sulfinic_acid", "[OX2H1][SX3]=[OX1]", 0),
    ("alcohol", "[OX2H1][CX4]", 0),
    ("aniline_NH", "[NX3;H1,H2]c", 0),
    ("amine_NH", "[NX3;H1,H2;!$(N[!#6]);!$(N=*)]", 0),
    # Already-cationic species can lose a proton back to neutral.
    ("ammonium", "[NX4+;H1,H2,H3,H4]", 0),
    ("protonated_aromatic_n", "[nX3+;H1]", 0),
    ("oxonium", "[OX3+;H1,H2,H3]", 0),
)

_BASE_PATTERNS: tuple[tuple[str, str, int], ...] = (
    ("amidine_guanidine", "[NX3;!$(N[CX3]=[OX1])][CX3]=[NX2]", 2),
    ("aliphatic_amine", "[NX3;!$(N[!#6;!#1]);!$(N=*);!$(N#*);"
                        "!$([NX3][CX3]=[OX1,SX1]);!$([NX3][SX4]);!$([NX3]a)]", 0),
    ("aromatic_n", "[nX2]", 0),
    # Allow the imine carbon to be aromatic: 2-imino heterocycles are common
    # bases (pKa ~10) and [CX3] alone never matches an aromatic ring carbon.
    ("imine", "[NX2;!$(N[!#6;!#1])]=[CX3,c]", 0),
    # Hydrazines and hydroxylamines: the aliphatic_amine pattern excludes any N
    # bonded to a heteroatom, which silently discarded both classes.
    ("hydrazine", "[NX3;!$(N[CX3]=[OX1]);!$(N[SX4]);!$(N=*);!$(N[OX1])]"
                  "[NX3;!$(N=*);!$(N[CX3]=[OX1])]", 0),
    # O need not carry an H: cyclic N-O-C systems are bases too.
    ("hydroxylamine", "[NX3;!$(N[CX3]=[OX1]);!$(N=*);!$(N[OX1])][OX2]", 0),
    # Any nitrogen on an aromatic ring carbon, including TERTIARY anilines.
    # Requiring an N-H here silently dropped every N,N-dialkylaniline, which
    # are ordinary bases (N,N-dimethylaniline titrates near pKa 5).
    ("aniline_N", "[NX3;!$(N[CX3]=[OX1]);!$(N[SX4]);!$(N=*);!$(N[OX1])]c", 0),
    # Anions can be re-protonated.
    ("carboxylate", "[OX1-][CX3]=[OX1]", 0),
    ("alkoxide_phenoxide", "[OX1-;$([O-][#6])]", 0),
    ("azolate", "[nX2-]", 0),
    ("amide_anion", "[NX2-]", 0),
    ("thiolate", "[SX1-]", 0),
)


def _compile(patterns) -> tuple[tuple[str, Chem.Mol, int], ...]:
    out = []
    for label, smarts, index in patterns:
        query = Chem.MolFromSmarts(smarts)
        if query is not None:
            out.append((label, query, index))
    return tuple(out)


_ACIDS = _compile(_ACID_PATTERNS)
_BASES = _compile(_BASE_PATTERNS)


def species_key(smiles: str) -> str:
    """Identity of a microstate as a *chemical species*, not a Lewis structure.

    Returns the full standard InChIKey. Its first block applies mobile-hydrogen
    perception, so tautomers and resonance forms of one protonation level
    collapse together, while the final character carries the /p protonation
    layer, so different protonation levels stay apart. Acetic acid is
    ``QTBSBXVTEAMEQO-UHFFFAOYSA-N`` and acetate ``...-M``; the three
    4-aminoquinazoline cations that SAMPL6 lists separately all share
    ``FVWANTDQRFSCAL-UHFFFAOYSA-O``.

    Using only the first block would be wrong: it merges an acid with its
    conjugate base, which is precisely the distinction a microstate encodes.

    This matters more than it sounds. SAMPL6 publishes 351 microstate Lewis
    structures across its 24 molecules, but those collapse to 144 distinct
    species under this key -- the rest are alternative ways of drawing the same
    protonation level. For a delocalised amidinium the choice of which nitrogen
    carries the proton is a drawing convention, so comparing enumerators by
    exact SMILES measures agreement about notation as much as about chemistry.

    Use exact SMILES when the Lewis structure genuinely matters (feeding a
    specific geometry to a QM engine); use this when asking whether two
    enumerations found the same chemistry.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeError(f"could not parse {smiles!r}")
    key = Chem.MolToInchiKey(mol)
    if not key:
        raise MoleculeError(f"could not generate InChIKey for {smiles!r}")
    return key


def largest_fragment(smiles: str) -> str:
    """Strip salts and solvates -- several sources ship HCl salts."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeError(f"could not parse {smiles!r}")
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        raise MoleculeError(f"no fragments in {smiles!r}")
    best = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    return Chem.MolToSmiles(best, isomericSmiles=True, canonical=True)


def find_protonation_sites_on_mol(mol: Chem.Mol) -> list[ProtonationSite]:
    """Sites on an already-built mol, using ITS atom ordering.

    Needed when indices must line up with an external file (an SDF atom block,
    say) rather than with this package's canonical ordering.
    """
    seen: dict[tuple[int, str], ProtonationSite] = {}
    for kind, table in (("acid", _ACIDS), ("base", _BASES)):
        for label, query, index in table:
            for match in mol.GetSubstructMatches(query):
                if index >= len(match):
                    continue
                atom_idx = match[index]
                key = (atom_idx, kind)
                if key in seen:
                    continue
                atom = mol.GetAtomWithIdx(atom_idx)
                if kind == "acid" and atom.GetTotalNumHs() < 1:
                    continue
                if kind == "base" and atom.GetFormalCharge() > 0:
                    continue
                seen[key] = ProtonationSite(atom_idx, kind, label)
    return sorted(seen.values(), key=lambda s: (s.atom_idx, s.kind))


def find_protonation_sites(smiles: str) -> list[ProtonationSite]:
    """Atoms that can gain or lose a proton, on the heavy-atom mol.

    Indices refer to `Chem.MolFromSmiles(canonical_smiles(smiles))`, i.e. the
    heavy-atom ordering. Hydrogens are implicit throughout because protonation
    changes their count.
    """
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    if mol is None:
        raise MoleculeError(f"could not parse {smiles!r}")
    return find_protonation_sites_on_mol(mol)


def _apply(mol: Chem.Mol, changes: Iterable[tuple[int, int]]) -> str | None:
    """Apply proton deltas and return canonical SMILES, or None if invalid."""
    editable = Chem.RWMol(mol)
    for atom_idx, delta in changes:
        atom = editable.GetAtomWithIdx(atom_idx)
        total_h = atom.GetTotalNumHs()
        if delta < 0 and total_h < 1:
            return None
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(max(total_h + delta, 0))
        atom.SetFormalCharge(atom.GetFormalCharge() + delta)
    candidate = editable.GetMol()
    try:
        Chem.SanitizeMol(candidate)
    except Exception:
        return None
    return Chem.MolToSmiles(candidate, isomericSmiles=True, canonical=True)


def enumerate_microstates(
    smiles: str,
    *,
    min_charge: int = -2,
    max_charge: int = 2,
    max_changes: int = 3,
    max_states: int = 512,
    strip_salts: bool = True,
) -> list[EnumeratedState]:
    """All protonation states reachable within the given charge window.

    `max_changes` bounds how many protons may be added or removed relative to
    the input structure, which keeps the combinatorics finite on polyprotic
    molecules without silently dropping the states that matter.
    """
    if min_charge > max_charge:
        raise ValueError("min_charge must not exceed max_charge")
    reference = largest_fragment(smiles) if strip_salts else canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(reference)
    sites = find_protonation_sites(reference)
    acids = [s.atom_idx for s in sites if s.kind == "acid"]
    bases = [s.atom_idx for s in sites if s.kind == "base"]

    base_charge = Chem.GetFormalCharge(mol)
    found: dict[str, EnumeratedState] = {}
    # The reference structure itself is always a microstate.
    found[reference] = EnumeratedState(reference, base_charge, 0, ())

    for n_deprot in range(0, min(len(acids), max_changes) + 1):
        for n_prot in range(0, min(len(bases), max_changes) + 1):
            if n_deprot + n_prot == 0 or n_deprot + n_prot > max_changes:
                continue
            charge = base_charge - n_deprot + n_prot
            if not (min_charge <= charge <= max_charge):
                continue
            for lose in itertools.combinations(acids, n_deprot):
                for gain in itertools.combinations(bases, n_prot):
                    if set(lose) & set(gain):
                        continue
                    changes = tuple(
                        [(i, -1) for i in lose] + [(i, +1) for i in gain]
                    )
                    result = _apply(mol, changes)
                    if result is None or result in found:
                        continue
                    found[result] = EnumeratedState(
                        result, charge, n_prot - n_deprot, changes
                    )
                    if len(found) >= max_states:
                        return sorted(found.values(), key=lambda m: (m.charge, m.smiles))
    return sorted(found.values(), key=lambda m: (m.charge, m.smiles))


def to_ensemble(
    states: Iterable[EnumeratedState],
    relative_energies_pk_units: dict[str, float] | None = None,
):
    """Convert enumerated states into a `microstates.MicrostateEnsemble`.

    This is the bridge that latent-site pKa training needs, and it exists
    because the two proton counts mean different things. `EnumeratedState`
    carries a signed `proton_delta` relative to whatever structure was handed
    in; `microstates.Microstate` needs an absolute, non-negative count so that
    protonation levels can be grouped for the partition-function sum. The shift
    is by the most-deprotonated state present, which makes that level zero.

    `relative_energies_pk_units` maps SMILES to a free energy in pK units; any
    state left out is given 0.0, which is only meaningful if you intend every
    state to be equally weighted.
    """
    from .microstates import Microstate as _Microstate
    from .microstates import MicrostateEnsemble

    states = list(states)
    if not states:
        raise ValueError("cannot build an ensemble from no states")
    floor = min(s.proton_delta for s in states)
    energies = relative_energies_pk_units or {}
    return MicrostateEnsemble(
        states=tuple(
            _Microstate(
                smiles=s.smiles,
                n_protons=s.proton_delta - floor,
                relative_g_pk_units=float(energies.get(s.smiles, 0.0)),
            )
            for s in states
        )
    )
