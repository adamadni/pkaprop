"""molprop -- pKa prediction for small molecules.

Predicts a pKa for every ionisable site in a molecule and reports the most
acidic one. The modelling is physics-informed: GFN2-xTB electronic descriptors
are computed per atom and fed to a graph neural network alongside the usual
topological features.

The package is organised in three layers.

**Correctness layer** -- makes silent errors impossible rather than unlikely.
`units` gives every value an explicit unit, kind, phase and temperature and
refuses arithmetic across a mismatch. `records` types a pKa measurement and
insists it declare whether it is macroscopic or microscopic. `splits` provides
scaffold and reaction-centre splits and has no random split at all. `cards`
refuses to report a result without a baseline on the same split.

**Chemistry layer** -- `chem` fixes a deterministic atom ordering, `enumeration`
finds protonation sites and generates microstates (validated against SAMPL6),
`microstates` aggregates site-level pKa into macroscopic constants.

**Modelling layer** -- `features` builds atom and bond tensors including the
xTB descriptors, `model` is the message-passing network, `train` fits it.
"""

from .cards import Evaluation, Metric, ResultCard
from .chem import MoleculeError, canonical_smiles, indexed_mol
from .curation import CurationIssue, audit_pka_records, ionisable_sites, summarise
from .enumeration import (
    EnumeratedState,
    ProtonationSite,
    enumerate_microstates,
    find_protonation_sites,
    species_key,
    to_ensemble,
)
from .microstates import (
    Microstate,
    MicrostateEnsemble,
    macro_pka_from_microstates,
    microstate_populations,
)
from .records import MoleculeRef, PKaRecord, PropertyRecord
from .registry import DatasetManifest, Registry, UnitsAudit
from .splits import (
    Split,
    SplitSpec,
    assert_no_group_leakage,
    reaction_centre_split,
    scaffold_split,
    time_split,
)
from .units import (
    LOG10_UNIT,
    Phase,
    Quantity,
    QuantityKind,
    UnitError,
    log10_to_molar_energy,
    molar_energy_to_log10,
    rt_ln10_kcal_per_mol,
    water,
)

__version__ = "0.2.0"

__all__ = [
    "Evaluation", "Metric", "ResultCard",
    "MoleculeError", "canonical_smiles", "indexed_mol",
    "CurationIssue", "audit_pka_records", "ionisable_sites", "summarise",
    "EnumeratedState", "ProtonationSite", "enumerate_microstates",
    "find_protonation_sites", "species_key", "to_ensemble",
    "Microstate", "MicrostateEnsemble", "macro_pka_from_microstates",
    "microstate_populations",
    "MoleculeRef", "PKaRecord", "PropertyRecord",
    "DatasetManifest", "Registry", "UnitsAudit",
    "Split", "SplitSpec", "assert_no_group_leakage", "reaction_centre_split",
    "scaffold_split", "time_split",
    "LOG10_UNIT", "Phase", "Quantity", "QuantityKind", "UnitError",
    "log10_to_molar_energy", "molar_energy_to_log10", "rt_ln10_kcal_per_mol",
    "water",
    "__version__",
]
