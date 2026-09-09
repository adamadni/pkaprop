# molprop

Drug-oriented QM property prediction: bond dissociation energies, pKa, tautomer
ratios. Delta-learning from a cheap method to a reference, with fidelity as a
model input and uncertainty-driven selection of what to compute next.

**Status: steps 1, 1b and 2 complete.** The infrastructure layer makes silent
errors impossible; the pipeline layer computes real numbers through it.

`xtb` 6.7.1 runs locally (conda-forge ships a win-64 build), so the whole
semiempirical stage — conformers, optimisation, Hessians, BDE assembly — needs
no remote compute. A full GFN2 BDE (parent + two radical fragments, each
optimised with a Hessian) takes ~0.3 s.

## Install

```bash
mamba env create -f environment.yml
conda activate molprop
pip install -e . --no-deps
pytest
```

## The four invariants

Each is enforced by construction, not by convention.

### 1. Every value carries units, kind, phase and temperature (`units.py`)

`Quantity` refuses arithmetic across a mismatch. Three distinct error classes
are blocked: unit confusion (kcal/mol vs kJ/mol vs hartree), *kind* confusion
(electronic energy used where enthalpy or Gibbs was meant — numerically
plausible as each other, and the most expensive silent bug in this domain), and
ensemble confusion (gas-phase mixed with solution-phase, or different
temperatures).

```python
>>> g(10.0, GAS) + h(4.0, GAS)
UnitError: cannot add G and H
>>> g(1.0, phase=GAS) + g(1.0, phase=water())
UnitError: cannot add across phases: gas vs solution(water)
>>> g(1.0) + 2.0
UnitError: cannot add float and Quantity; wrap the bare number in a Quantity
```

`R*T*ln(10) = 1.3642 kcal/mol at 298.15 K` is defined in exactly one place.
`log10_to_molar_energy` / `molar_energy_to_log10` are the only bridge between
log units and free energy, and the reverse direction accepts **only**
`DELTA_GIBBS` — deriving a pKa from an enthalpy omits the entropic term, so it
raises rather than returning a wrong number.

### 2. "DFT" is not a fidelity label (`fidelity.py`)

wB97X-D/def2-TZVPD/SMD and B3LYP/6-31G\*/PCM differ by more than the delta being
learned. A `Fidelity` carries method, basis, dispersion, solvation model and
solvent, geometry source, software and version — and a `ConformerProtocol`,
because for pKa and tautomer ratios the choice between lowest-conformer and a
Boltzmann ensemble moves the answer more than the functional does. Constructing
a DFT fidelity without a basis raises.

The two property tracks are structurally different, and `DeltaSpec` is where
that is made explicit:

| track | base | target | why |
|---|---|---|---|
| BDE | GFN2-xTB | DFT | DFT is a legitimate gas-phase ground truth |
| pKa | xTB or DFT | **experiment** | a DFT pKa is itself an empirically calibrated quantity, so it is a rung, never the reference |

### 3. Macro and micro pKa are different targets (`microstates.py`)

Experimental labels are macroscopic; a site model predicts microscopic. For any
polyprotic molecule these differ. `macro_pka_from_microstates` aggregates with
the standard partition-function relation, so the statistical factor falls out:
two independent sites at micro pKa 4.0 give macro pKa₁ = 3.699 and pKa₂ = 4.301.

The reverse (`micro_pka_from_macro`) raises `NotImplementedError` and says why:
the decomposition is underdetermined without site-resolved measurement.

### 4. No random splits, no leakage, no unbaselined result (`splits.py`, `cards.py`)

There is no `random_split`; asking for one by name raises with an explanation.
Three strategies exist: `scaffold_split`, `reaction_centre_split`, `time_split`.
Allocation is deterministic (no RNG), and every split reserves a **`frozen_test`**
partition that active learning may never draw from.

Two subtleties the module surfaces rather than hides:

- **Group integrity.** A molecule contributes many BDE records; scattering them
  across folds leaks. Every assignment is checked against the record's
  `group_key`.
- **Scaffold is the wrong invariant for a local property.** pKa is dominated by
  the ionisable site's environment, so a novel scaffold carrying a familiar
  aniline is trivially predicted. `reaction_centre_split` groups on the site
  environment — but for a multi-site property like BDE, centre holdout and
  molecule integrity are *incompatible*, and the split raises rather than
  silently degenerating into a molecule split. Choose `group_integrity="centre"`
  deliberately, or use `scaffold_split`.

A `ResultCard` cannot be constructed without at least one baseline evaluated on
the same split id and the same partition, with commensurable metric units.
Metrics without units are rejected unless the name is a known dimensionless
score.

## Curation contract (`curation.py`)

`audit_pka_records` and `audit_tautomer_records` check a candidate batch before
registration. Codes emitted:

| code | severity | what it catches |
|---|---|---|
| `mixed-resolution` | error | macro and micro pKa pooled as one target |
| `mixed-solvent` | error | methanol/water values pooled with aqueous |
| `missing-temperature` | error | a pKa that cannot become a free energy |
| `non-contiguous-steps` | error | a titration with an unrecorded equilibrium |
| `site-has-no-proton` | error | a site index against the wrong atom ordering |
| `conflicting-duplicate` | error | one compound, two irreconcilable values |
| `incomplete-titration` | warning | polyprotic compound with a single macro pKa |
| `implausible-value` | warning | extrapolations and unit errors |
| `saturated-ratio` | warning | "only the keto form is seen" encoded as a number |
| `not-a-tautomer-pair` | warning | structures RDKit cannot connect by tautomerism |
| `missing-ionic-strength` | warning | fine for one source, not for a pooled set |
| `missing-date` | warning | blocks a time split |

`KNOWN_PKA_SOURCES` and `KNOWN_TAUTOMER_SOURCES` list candidate sources with
their caveats. Their `doi`, `url` and `n_records` are deliberately unset: those
get filled from the artefact actually downloaded in step 4, so a manifest never
carries provenance nobody verified.

## Registry (`registry.py`)

`<root>/<name>/<version>/{records.jsonl, manifest.json}`. Content is hashed
(order-independent) and `load` refuses to return records that do not match the
manifest. `write` runs a units audit and refuses a dataset that silently mixes
kinds, solvents or temperatures unless `allow_mixed_conditions=True`.
Registered versions are immutable — bump the version rather than mutating.

Fidelities are stored **once** in the manifest and referenced by id from each
row; null fields are dropped. A dataset has few fidelities and many rows, so
inlining spent most of the file restating the same 500 bytes — ALFABET went
from 463 MB to 206 MB. Rows carrying an inline `fidelity` (the earlier layout)
still load, and a row referencing a fidelity the manifest does not define is a
hard error rather than a silent substitution.

## Registered datasets

| name | version | records | provenance |
|---|---|---|---|
| `alfabet-bde-db` | v2 | 484,907 | figshare `10.6084/m9.figshare.10248932`, MIT, md5-verified |
| `pka-czodrowskilab-train` | v1 | 5,994 | czodrowskilab/Machine-learning-meets-pKa, MIT |
| `pka-novartis` | v1 | 280 | same repo, held-out test set |
| `pka-avlilumove` | v1 | 123 | same repo, held-out test set |
| `pka-sampl6` | v1 | 31 | samplchallenges/SAMPL6, MIT, 25 C / 0.15 M KCl |

Each pKa manifest embeds its curation audit at import time. Three decisions are
baked into `datasets/pka.py` because the alternative is a silent error:
molecule identity is the **neutral parent** (sources disagree about protonation
state, and 3,363 of 5,994 training rows needed normalising); Marvin's
`marvin_atom` is a **prediction** kept in metadata and never used as
`site_atom_idx` on a macroscopic value; and temperature is left **unset** where
the source does not distribute it, so the audit raises rather than a plausible
298.15 being invented.

ALFABET's `bond_index` indexes into `AddHs(MolFromSmiles(molecule))` using its
SMILES *as written*, and ~0.9% of those are not RDKit-canonical — so reusing the
integer directly picks the wrong bond. `datasets.alfabet.canonical_bond`
remaps through canonical atom ranks, and every row is gated on its element pair
matching the published `bond_type` column.

**The published 0.58 kcal/mol MAE is on a random molecule-level holdout.** This
project uses scaffold splits only, so no result here is comparable to that
number, and the result-card notes must say so rather than implying otherwise.

## Atom indexing

Record atom indices (BDE bonds, microscopic pKa sites) are meaningless without a
fixed atom ordering. The convention throughout:

```
identity    = canonical isomeric SMILES (RDKit, no explicit hydrogens)
indexed mol = Chem.AddHs(Chem.MolFromSmiles(identity))
```

`AddHs` appends hydrogens after the heavy atoms in heavy-atom order, so indices
are a deterministic function of the identity string alone.

## The pipeline (step 2)

```
SMILES -> conformers.generate_conformers   ETKDGv3 + MMFF94s, seeded, RMSD-pruned
       -> engines.XTBEngine.run            GFN2-xTB opt + Hessian
       -> thermo.ensemble_quantity         lowest or Boltzmann reduction
       -> bde.compute_bde                  H(A.) + H(B.) - H(A-B)
```

Five things the pipeline handles that a first draft would not:

* **Spin.** xtb's `--uhf` takes unpaired electrons, not multiplicity. Fragment
  multiplicity is derived from the radical count and checked against electron
  parity, so a radical can never be computed as a closed shell.
* **The hydrogen atom.** Every X-H homolysis produces one. It has no MMFF
  parameters and no vibrations, so xtb refuses `--ohess`. It gets a single point
  plus analytic ideal-gas thermochemistry (Sackur-Tetrode), which reproduces the
  tabulated 27.4 cal/(mol K) standard entropy.
* **Saddle points.** A structure with imaginary frequencies is not a minimum and
  its "free energy" is not a free energy. `require_usable()` refuses it. With
  `imaginary_retries`, the engine displaces along the imaginary mode and
  re-optimises, which recovers both torsional saddles and Jahn-Teller cases.
* **Encoding.** xtb writes UTF-8; Windows would decode as cp1252 and crash on
  the first Greek letter. `normal termination of xtb` goes to *stderr*.
* **Cost.** Wall time and core-hours are recorded per job and summed per BDE.
  An unpriced run reports `None`, never zero.

### Determinism gate

`conformers.assert_deterministic` runs a protocol several times and requires
byte-identical geometries. A failure means the fidelity id is lying. Embedding
is single-threaded for that reason, and RMSD pruning happens only *after*
force-field optimisation, so `rmsd_prune_angstrom` means one thing rather than
two.

`rmsd_atoms="heavy"` cannot see O-H / N-H rotamers, which change intramolecular
hydrogen bonding. The pKa and tautomer tracks should set `"all"`.

## Next

Reference data. The validation gates that matter -- reproduce ALFABET BDEs at
their level of theory, reproduce SAMPL6 pKa -- need those datasets registered,
and by this project's own curation rule their provenance must come from the
downloaded artefact rather than from memory.

Then a DFT engine. PySCF is Linux-only and Psi4 ships only py310 builds for
win-64, so that stage is the one that goes to Runpod.
