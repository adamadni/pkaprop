"""molprop -- drug-oriented QM property prediction infrastructure.

Step 1 (infrastructure) and step 1b (curation schema) of the project plan.
Nothing here computes a property; this layer exists to make the things that
compute properties impossible to get silently wrong.

Four invariants are enforced by construction rather than by convention:

* every value carries units, kind, phase and temperature   -- `units`
* "DFT" is not a fidelity label                            -- `fidelity`
* macro and micro pKa are different targets                -- `microstates`
* no random splits, no group leakage, no unbaselined result -- `splits`, `cards`
"""

from .bde import Fragment, bde_from_enthalpies, compute_bde, homolysis_fragments
from .calc import CalculationResult, CalculationSpec, ResourceUsage, Task
from .cards import Evaluation, Metric, ResultCard
from .conformers import assert_deterministic, generate_conformers
from .geometry import Conformer, ConformerEnsemble, check_multiplicity, electron_count
from .thermo import (
    boltzmann_average,
    boltzmann_weights,
    ensemble_quantity,
    monatomic_thermochemistry,
)
from .chem import BondRef, MoleculeError
from .curation import (
    KNOWN_PKA_SOURCES,
    KNOWN_TAUTOMER_SOURCES,
    CurationIssue,
    SourceSpec,
    audit_pka_records,
    audit_tautomer_records,
    ionisable_sites,
)
from .fidelity import ConformerProtocol, DeltaSpec, Fidelity, FidelityLadder, Tier
from .microstates import (
    Microstate,
    MicrostateEnsemble,
    macro_pka_from_microstates,
    microstate_populations,
)
from .records import BDERecord, MoleculeRef, PKaRecord, PropertyRecord, TautomerRecord
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
    GAS,
    HARTREE,
    KCAL_PER_MOL,
    KJ_PER_MOL,
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

__version__ = "0.1.0"

__all__ = [
    "Fragment",
    "bde_from_enthalpies",
    "compute_bde",
    "homolysis_fragments",
    "CalculationResult",
    "CalculationSpec",
    "ResourceUsage",
    "Task",
    "assert_deterministic",
    "generate_conformers",
    "Conformer",
    "ConformerEnsemble",
    "check_multiplicity",
    "electron_count",
    "boltzmann_average",
    "boltzmann_weights",
    "ensemble_quantity",
    "monatomic_thermochemistry",
    "Evaluation",
    "Metric",
    "ResultCard",
    "BondRef",
    "MoleculeError",
    "KNOWN_PKA_SOURCES",
    "KNOWN_TAUTOMER_SOURCES",
    "CurationIssue",
    "SourceSpec",
    "audit_pka_records",
    "audit_tautomer_records",
    "ionisable_sites",
    "ConformerProtocol",
    "DeltaSpec",
    "Fidelity",
    "FidelityLadder",
    "Tier",
    "Microstate",
    "MicrostateEnsemble",
    "macro_pka_from_microstates",
    "microstate_populations",
    "BDERecord",
    "MoleculeRef",
    "PKaRecord",
    "PropertyRecord",
    "TautomerRecord",
    "DatasetManifest",
    "Registry",
    "UnitsAudit",
    "Split",
    "SplitSpec",
    "assert_no_group_leakage",
    "reaction_centre_split",
    "scaffold_split",
    "time_split",
    "GAS",
    "HARTREE",
    "KCAL_PER_MOL",
    "KJ_PER_MOL",
    "LOG10_UNIT",
    "Phase",
    "Quantity",
    "QuantityKind",
    "UnitError",
    "log10_to_molar_energy",
    "molar_energy_to_log10",
    "rt_ln10_kcal_per_mol",
    "water",
    "__version__",
]
