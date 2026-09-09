# molprop — physics-informed pKa prediction

Predicts a pKa for **every ionisable site** in a small molecule and reports the
**most acidic** one. A graph neural network does the learning; GFN2-xTB
electronic descriptors supply the physics prior.

```python
from molprop.training import most_acidic_site

most_acidic_site(model, "NCCc1ccc(O)c(O)c1")   # dopamine
# {'smiles': 'NCCc1ccc(O)c(O)c1',
#  'most_acidic': {'atom_index': 6, 'kind': 'acid', 'pka': 9.1},
#  'sites': [...]}                              # every site, sorted
```

---

## Results

Scaffold split at molecule level, 5,994 compounds. **There is no random split
anywhere in this project** — see [Splitting](#splitting).

| system | MAE | RMSE | R² | within 1 unit | within 2 |
|---|---|---|---|---|---|
| global median | 1.95 | 2.29 | −0.00 | 28% | 52% |
| acid/base median | 1.83 | 2.22 | 0.06 | 30% | 61% |
| site-class median | 1.34 | 1.75 | 0.42 | 49% | 75% |
| **pKa GNN** | **0.84** | **1.21** | **0.72** | **71%** | **90%** |

Held-out sets curated by the data source, never seen in training or validation:

| set | n | MAE | RMSE | within 1 unit |
|---|---|---|---|---|
| Novartis | 280 | 1.03 | 1.39 | 58% |
| AvLiLuMoVe | 123 | 0.57 | 0.80 | 85% |

The baseline that matters is **site-class median** — the pKa you would guess
from knowing only the functional group (carboxylic acid, aliphatic amine,
phenol…). Beating it by 0.50 log units is the claim that the model has learned
substituent effects rather than group identity. `ResultCard` refuses to be
constructed without a baseline on the same split, so this comparison cannot be
quietly omitted from a report.

---

## The target, and why it is defined this way

Every ionisable proton has a pKa. A carboxylic acid donates one directly; an
amine's proton belongs to its *conjugate acid*. Both are deprotonations, so the
sign convention is uniform.

**"Most acidic site" = argmin over every site's predicted pKa** — acid sites
scored by the acidic head, basic sites by the basic head.

That definition was not the obvious one, and getting it wrong is easy. The
natural first framing — "predict the pKa of the most acidic proton" — is wrong
for this corpus, because **60% of the compounds are bases** (3,596 basic vs
2,398 acidic). For a base, the recorded value is the pKa of the conjugate acid
at the most *basic* centre, which in the protonated species is the *least*
acidic proton. Framing the target around acidic protons alone mislabels the
majority of the data. Scoring both site types and taking the global minimum is
what makes the question well posed.

---

## Architecture

### Why a GNN at all

pKa is a **local** property with **non-local modifiers**. Which atom loses the
proton is decided by its immediate environment; how easily depends on
substituents several bonds away donating or withdrawing electron density. That
is a graph problem: a fingerprint discards the site, and a flat descriptor
vector discards the topology through which the substituent effect travels.

### Per-atom output, not per-molecule

The prediction is attached to an **atom**. A whole-molecule regressor could emit
a number but could never point at a site, and pooling to a molecule vector
discards exactly the locality the answer depends on.

### Edge-conditioned message passing

Messages are built from `[h_source, h_target, edge_features]`.

Acidity is set by how well the conjugate base delocalises charge, and that
propagates through bonds — inductive effects through σ bonds, resonance through
π systems. A convolution whose messages depend on the **bond** as well as the
neighbouring atom can distinguish those. One that sees only neighbour identities
cannot tell a conjugated path from a saturated one, which is roughly the
difference between a phenol (pKa 10) and a cyclohexanol (pKa 16).

### Depth 4

A Hammett substituent effect reaches roughly 4–6 bonds through a conjugated
system before decaying into noise. Four rounds give each atom a 4-bond receptive
field, covering *para* substitution on a ring bearing the ionisable group — the
canonical long-range case. Deeper mostly buys over-smoothing, where every atom
in a fused ring converges to the same embedding.

### Residual updates with layer norm

At depth 4 on ~20-atom graphs, plain stacking washes out the atom's own
identity. A residual update keeps the site's local character (an oxygen stays
recognisably an oxygen) while the message stream carries the environment.

### Two heads, one for acids and one for bases

The same nitrogen has an "acidic pKa" near 35 and a "basic pKa" near 10. These
are different physical quantities that happen to share an atom. A single head
would have to fit a strongly bimodal target conditioned on an unstated variable.

**Only the head matching a record's type is supervised.** An amine has no
measured acidic pKa, so inventing a target for it would teach the acidic head
chemistry that was never observed. The unmatched head receives no gradient from
that example.

### Sum-pooled global context

A small molecule-level vector is concatenated into the per-atom head. Overall
charge and molecular size shift pKa systematically, and a purely local readout
cannot see them.

### Huber loss

The corpus contains a few extreme or mis-transcribed values. Squared error would
let them dominate the gradient; Huber keeps them informative without letting
them steer.

---

## The physics: GFN2-xTB descriptors

Two quantities per molecule, from a single point on one force-field conformer:

- **Mulliken partial charges** (per atom). Acidity is largely the question of
  how well the conjugate base stabilises negative charge, so the charge already
  present at a site is a direct prior on that.
- **Wiberg bond orders** (per bond, summed per atom). This measures
  delocalisation: a carboxyl oxygen with substantial partial double-bond
  character behaves very differently from an isolated alkoxide.

**They are optional, and that is the point.** `featurise` zero-fills those
columns when descriptors are absent, so the identical architecture trains with
and without them. `--no-xtb` is a real ablation, not a fallback — the only way
to know whether the physics earns its compute is to measure it.

One conformer is used deliberately: these quantities are dominated by
connectivity and only weakly conformer-dependent, while generating and
optimising five conformers costs more than the xTB call itself. That trade-off
is part of the cache key, so revisiting it cannot silently reuse stale
descriptors.

---

## Data

| dataset | n | role | source |
|---|---|---|---|
| czodrowskilab combined | 5,994 | train / val / test | [Machine-learning-meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa) (MIT) |
| Novartis | 280 | held out | same repo, source-curated monoprotic |
| AvLiLuMoVe | 123 | held out | same repo, source-curated monoprotic |
| SAMPL6 | 31 | reference | [SAMPL6](https://github.com/samplchallenges/SAMPL6) (MIT), 25 °C / 0.15 M KCl |

`python scripts/fetch_data.py` downloads everything with checksums.

### Three curation decisions, each a silent error if reversed

**1. The site label is a prediction, and this is stated.** The pKa *value* is
experimental. The *site* comes from Marvin, distributed with the source, and
nobody measured which atom titrates for these 5,994 compounds. The model learns
"the experimental pKa, at the site Marvin believes is responsible". It is kept
in `metadata`, never in `PKaRecord.site_atom_idx`, so a macroscopic value can
never masquerade as a microscopic measurement.

**2. Molecule identity is the neutral parent.** Sources disagree about
protonation state — **3,363 of 5,994** training rows and 114/280 Novartis rows
were stored as charged species. Keying on the published form would give one
compound two identities and let it straddle a train/test boundary.

**3. Atom indices are remapped, not reused.** `marvin_atom` indexes the original
SDF atom block; records key on the canonical neutral parent — a different
molecule with a different atom order. Reusing the integer points at the wrong
atom. `map_atom_to_canonical` remaps through canonical ranks and lands on a
heteroatom for **5,928/5,994 (98.9%)**.

### Known data limitations

- **Temperature and ionic strength are not distributed** with the czodrowskilab
  corpus. `audit_pka_records` raises `missing-temperature` as an **error**
  rather than filling in a plausible 298.15 K.
- **Macroscopic labels on polyprotic compounds.** Many compounds have more than
  one ionisable centre but carry a single recorded value, so which equilibrium
  it refers to is an assumption. The audit reports the count on every import.
- **A leak in the obvious validation setup.** SAMPL6 **SM11** appears in the
  training corpus with an identical value (3.89, via DataWarrior). Exclude it
  before using SAMPL6 as a blind set.
- SAMPL6 ships no Marvin site annotations, so it cannot yet be scored
  site-wise; its 10 NMR-determined microstates are the right future check, and
  the only genuinely measured site assignments available anywhere here.

---

## Splitting

**There is no random split, and `molprop.splits.random_split` does not exist** —
asking for it by name raises with an explanation. Three strategies are
provided: `scaffold_split`, `reaction_centre_split`, `time_split`. Allocation is
deterministic (no RNG).

- **Group integrity.** One molecule contributes a record per titration step.
  Every assignment is checked so a molecule cannot straddle folds.
- **A `frozen_test` partition** is always reserved and is never available for
  model selection.
- **Scaffold is the wrong invariant for a local property**, so
  `reaction_centre_split` groups on the ionisable site's environment instead —
  a molecule can change scaffold entirely while keeping the same aniline.

---

## Correctness layer

The parts that exist to make silent errors impossible rather than unlikely.

**`units`** — every value carries a unit, a *kind* (pKa vs Gibbs vs enthalpy), a
phase and a temperature, and arithmetic across a mismatch raises. The expensive
bug in this domain is not kcal/kJ confusion; it is using an electronic energy
where a free energy was meant, which is numerically plausible and silently
wrong. `RT·ln(10) = 1.3642 kcal/mol at 298.15 K` is defined in exactly one
place, and only `DELTA_GIBBS` may be converted to a pKa.

**`enumeration`** — finds protonation sites and generates microstates, validated
against SAMPL6's published enumeration:

| comparison | exact SMILES | as species |
|---|---|---|
| microstates in an experimental pKa transition | 13/20 (65%) | **19/20 (95%)** |
| SAMPL6's full published enumeration | 129/351 (37%) | 92/144 (64%) |

The columns differ because SAMPL6 enumerates *Lewis structures*: its 351
microstates are only **144 distinct chemical species**. `species_key` uses the
full InChIKey, whose mobile-H layer merges tautomers while the `/p` layer keeps
protonation levels apart.

Its site patterns were not guessed. Auditing the 231 corpus compounds that had a
measured pKa but no detected site produced five genuinely missing classes —
hydrazines, hydroxamic acids, aromatic amidines, tertiary anilines, peroxy and
sulfinic acids — taking undetected compounds from **231 → 69 (1.2%)** with no
loss of SAMPL6 recall. The remainder are carbon acids, whose acidity comes from
delocalisation rather than a heteroatom–H bond; they are excluded and documented
rather than guessed at.

**`microstates`** — aggregates site-level pKa into macroscopic constants via the
partition-function relation, so the statistical factor falls out correctly (two
independent sites at micro pKa 4.0 give macro 3.699 and 4.301). The reverse
decomposition raises `NotImplementedError` because it is underdetermined.

**`cards`** — a `ResultCard` cannot be built without a baseline on the same
split id and partition, with commensurable metric units.

---

## Install and run

```bash
mamba env create -f environment.yml
conda activate molprop
pip install -e ".[dev]"

python scripts/fetch_data.py              # download sources with checksums
python scripts/precompute_xtb.py          # warm the descriptor cache
python scripts/train_pka.py --xtb         # train with physics
python scripts/train_pka.py --no-xtb      # ablation without it

pytest
```

`precompute_xtb.py` takes `--shard i --num-shards n` for parallel warming.

---

## Layout

```
src/molprop/
  units.py        typed quantities; mismatches raise
  chem.py         deterministic atom ordering
  records.py      PKaRecord: macro vs micro must be declared
  enumeration.py  protonation sites and microstates (SAMPL6-validated)
  microstates.py  micro -> macro aggregation
  curation.py     import-time audits
  splits.py       scaffold / reaction-centre / time; no random split
  registry.py     content-hashed datasets with provenance
  cards.py        results, baseline-enforced
  features.py     graph tensors + xTB descriptors
  model.py        the GNN
  training.py     dataset assembly, fitting, inference
  fidelity.py     level-of-theory specification
  conformers.py   ETKDGv3 + MMFF, deterministic
  engines/xtb.py  GFN2-xTB driver
  geometry.py calc.py thermo.py    QM plumbing
scripts/
  fetch_data.py  precompute_xtb.py  train_pka.py
```

---

## Project history

This codebase began as a broader QM property predictor covering bond
dissociation energies, pKa and tautomer ratios, and was rescoped to pKa alone.
The BDE track — ALFABET import, xTB campaigns, a direct-vs-delta ablation — is
preserved in commit `c0fa78b` rather than deleted outright.

Two findings from that work shaped this one. GFN2-xTB explains ~74% of C–H
bond-strength variance but only ~10% for O–H (r = 0.51, n = 1,528), so cheap
semiempirical thermochemistry is weakest exactly where acidic chemistry lives —
which is why xTB enters here as a *descriptor* feeding a learned model rather
than as a physical baseline to correct. And on the BDE benchmark an aggregate
MAE of 2.33 hid a model that was 6× **worse** than a trivial lookup on rare bond
types, which is why results here are reported against a class-wise baseline
rather than a single headline number.

## Licence

MIT. Both upstream datasets are MIT licensed.
