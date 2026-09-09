"""No random splits, no group leakage, and the right holdout unit."""

from __future__ import annotations

import pytest
from conftest import find_atom

import molprop.splits as splits_module
from molprop.chem import BondRef, indexed_mol
from molprop.records import BDERecord, MoleculeRef, PKaRecord
from molprop.splits import (
    LeakageError,
    Split,
    SplitError,
    SplitSpec,
    assert_no_group_leakage,
    coverage_report,
    reaction_centre_key,
    reaction_centre_split,
    scaffold_split,
    time_split,
)
from molprop.units import GAS, KCAL_PER_MOL, LOG10_UNIT, Quantity, QuantityKind, water

MOLECULES = [
    "Oc1ccccc1",
    "Oc1ccc(C)cc1",
    "Oc1ccc(Cl)cc1",
    "CCO",
    "CCCO",
    "c1ccc2ccccc2c1",
    "CC(=O)Nc1ccccc1",
    "Cc1ccccc1N",
    "OC(=O)c1ccccc1",
    "CCN(CC)CC",
]


def bde_q(value=90.0):
    return Quantity(value, KCAL_PER_MOL, QuantityKind.BDE, GAS, 298.15)


def all_xh_bonds(smiles: str) -> list[BondRef]:
    mol = indexed_mol(smiles)
    out = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        syms = {mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()}
        if "H" in syms:
            out.append(BondRef(i, j))
    return out


def bde_records(dft_gas) -> list[BDERecord]:
    """Every X-H bond of every molecule: many records per molecule."""
    records = []
    for m, smiles in enumerate(MOLECULES):
        ref = MoleculeRef.from_smiles(smiles)
        # Date is a property of the molecule, not the bond, so a time split
        # cannot scatter one molecule's records across folds.
        date = f"{2010 + m}-01-01"
        for k, bond in enumerate(all_xh_bonds(ref.smiles)):
            records.append(
                BDERecord(
                    molecule=ref,
                    quantity=bde_q(90.0 + k),
                    fidelity=dft_gas,
                    source="synthetic",
                    bond=bond,
                    date=date,
                )
            )
    return records


# -- the constraint itself -------------------------------------------------


def test_random_split_does_not_exist():
    with pytest.raises(AttributeError, match="by design"):
        splits_module.random_split


def test_train_test_split_does_not_exist():
    with pytest.raises(AttributeError, match="by design"):
        splits_module.train_test_split


def test_fractions_must_sum_to_one():
    with pytest.raises(SplitError, match="sum to 1"):
        SplitSpec("scaffold", {"train": 0.8, "val": 0.1, "test": 0.1, "frozen_test": 0.1})


def test_fractions_must_cover_every_partition():
    with pytest.raises(SplitError, match="missing"):
        SplitSpec("scaffold", {"train": 0.8, "val": 0.1, "test": 0.1})


# -- group integrity -------------------------------------------------------


def test_scaffold_split_never_scatters_a_molecule(dft_gas):
    records = bde_records(dft_gas)
    assert len(records) > len(MOLECULES), "each molecule must contribute many bonds"
    split = scaffold_split(records)
    assert_no_group_leakage(records, split)


def test_every_record_is_assigned_exactly_once(dft_gas):
    records = bde_records(dft_gas)
    split = scaffold_split(records)
    total = sum(split.counts().values())
    assert total == len(records)


def test_leakage_is_detected(dft_gas):
    records = bde_records(dft_gas)
    # Deliberately break it: put the first two records of one molecule apart.
    same_molecule = [
        i for i, r in enumerate(records) if r.group_key == records[0].group_key
    ]
    assert len(same_molecule) >= 2
    rest = [i for i in range(len(records)) if i not in same_molecule[:2]]
    broken = Split(
        spec=SplitSpec("manual", {"train": 0.5, "val": 0.0, "test": 0.5, "frozen_test": 0.0}),
        train=(same_molecule[0], *rest),
        val=(),
        test=(same_molecule[1],),
        frozen_test=(),
    )
    with pytest.raises(LeakageError, match="must stay together"):
        assert_no_group_leakage(records, broken)


def test_a_record_cannot_be_in_two_partitions():
    with pytest.raises(SplitError, match="more than one partition"):
        Split(
            spec=SplitSpec("manual", {"train": 1.0, "val": 0.0, "test": 0.0, "frozen_test": 0.0}),
            train=(0, 1),
            val=(1,),
            test=(),
            frozen_test=(),
        )


# -- determinism -----------------------------------------------------------


def test_scaffold_split_is_deterministic(dft_gas):
    records = bde_records(dft_gas)
    a, b = scaffold_split(records), scaffold_split(records)
    assert a.id == b.id
    assert a.train == b.train and a.test == b.test


def test_split_id_depends_on_the_assignment(dft_gas):
    records = bde_records(dft_gas)
    a = scaffold_split(records)
    b = scaffold_split(records, {"train": 0.5, "val": 0.2, "test": 0.2, "frozen_test": 0.1})
    assert a.id != b.id


# -- frozen test set -------------------------------------------------------


def test_a_frozen_test_partition_is_reserved(dft_gas):
    records = bde_records(dft_gas)
    split = scaffold_split(records)
    assert len(split.frozen_test) > 0, (
        "active learning shifts the training distribution; the frozen partition "
        "must exist from the first split"
    )
    assert not (set(split.frozen_test) & set(split.train))


# -- reaction centre -------------------------------------------------------


def test_reaction_centre_key_differs_between_oh_and_ch(dft_gas):
    ref = MoleculeRef.from_smiles("Oc1ccccc1")
    mol = ref.indexed()
    oh = next(
        BondRef(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if {mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
            mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} == {"O", "H"}
    )
    ch = next(
        BondRef(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if {mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
            mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol()} == {"C", "H"}
    )
    make = lambda bond: BDERecord(
        molecule=ref, quantity=bde_q(), fidelity=dft_gas, source="s", bond=bond
    )
    assert reaction_centre_key(make(oh)) != reaction_centre_key(make(ch))


def test_centre_grouping_collapses_for_a_multi_site_property(dft_gas):
    """BDE cannot have both centre holdout and molecule integrity.

    Every molecule spans several distinct centres, so enforcing molecule
    integrity merges the centre classes back into a molecule split. That must
    be an error, not a silent degeneration.
    """
    records = bde_records(dft_gas)
    with pytest.raises(SplitError, match="collapsed from"):
        reaction_centre_split(records, radius=2)


def test_centre_integrity_is_available_as_a_deliberate_choice(dft_gas):
    records = bde_records(dft_gas)
    split = reaction_centre_split(records, radius=2, group_integrity="centre")
    assert sum(split.counts().values()) == len(records)
    # ...and the molecule-level guard is genuinely relaxed, not merely unchecked
    partition_of = {}
    for p in ("train", "val", "test", "frozen_test"):
        for i in getattr(split, p):
            partition_of[i] = p
    spanning = {
        r.group_key
        for r in records
        if len({partition_of[i] for i, x in enumerate(records) if x.group_key == r.group_key}) > 1
    }
    assert spanning, "centre mode should let at least one molecule span folds"


def test_group_integrity_argument_is_validated(dft_gas):
    with pytest.raises(SplitError, match="group_integrity"):
        reaction_centre_split(bde_records(dft_gas), group_integrity="whatever")


def test_every_partition_gets_at_least_one_group(dft_gas):
    records = bde_records(dft_gas)
    split = scaffold_split(records)
    for partition in ("train", "val", "test", "frozen_test"):
        assert len(getattr(split, partition)) > 0, f"{partition} is empty"


def test_too_few_groups_is_an_error(dft_gas):
    records = [r for r in bde_records(dft_gas) if r.molecule.smiles == "CCO"]
    with pytest.raises(SplitError, match="cannot populate"):
        scaffold_split(records)


def test_macro_pka_has_no_reaction_centre(experiment):
    rec = PKaRecord(
        molecule=MoleculeRef.from_smiles("CC(=O)O"),
        quantity=Quantity(4.76, LOG10_UNIT, QuantityKind.PKA, water(), 298.15),
        fidelity=experiment,
        source="iupac",
        resolution="macro",
        step_index=1,
    )
    with pytest.raises(SplitError, match="no reaction centre"):
        reaction_centre_key(rec)


def test_micro_pka_reaction_centre_is_the_site(experiment):
    smiles = "CC(=O)O"
    rec = PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=Quantity(4.76, LOG10_UNIT, QuantityKind.PKA, water(), 298.15),
        fidelity=experiment,
        source="sampl6",
        resolution="micro",
        site_atom_idx=find_atom(smiles, "O", with_h=True),
        protonated_smiles="CC(=O)O",
        deprotonated_smiles="CC(=O)[O-]",
    )
    assert reaction_centre_key(rec)


# -- time split ------------------------------------------------------------


def test_time_split_requires_dates(dft_gas):
    records = [
        BDERecord(
            molecule=MoleculeRef.from_smiles("CCO"),
            quantity=bde_q(),
            fidelity=dft_gas,
            source="s",
            bond=all_xh_bonds("CCO")[0],
        )
    ]
    with pytest.raises(SplitError, match="date on every record"):
        time_split(records, {"train": "2015-01-01", "val": "2018-01-01"})


def test_time_split_respects_the_cutoffs(dft_gas):
    records = bde_records(dft_gas)
    split = time_split(records, {"train": "2014-01-01", "val": "2016-01-01"})
    assert all(records[i].date < "2014-01-01" for i in split.train)
    assert all("2014-01-01" <= records[i].date < "2016-01-01" for i in split.val)
    assert all(records[i].date >= "2016-01-01" for i in (*split.test, *split.frozen_test))
    assert sum(split.counts().values()) == len(records)


def test_coverage_report_mentions_every_partition(dft_gas):
    records = bde_records(dft_gas)
    report = coverage_report(records, scaffold_split(records))
    for partition in ("train", "val", "test", "frozen_test"):
        assert partition in report


def test_realised_fractions_track_the_targets(dft_gas):
    """With enough comparable groups the allocation lands near the targets.

    Uses acyclic molecules, which each form their own group, so the test
    exercises the allocator rather than the scaffold skew of a ring series.
    """
    smiles = []
    for n in range(1, 16):
        chain = "C" * n
        smiles += [chain + "O", chain + "N", chain + "S", chain + "C(=O)O"]

    records = []
    for s in smiles:
        ref = MoleculeRef.from_smiles(s)
        for bond in all_xh_bonds(ref.smiles)[:4]:
            records.append(
                BDERecord(
                    molecule=ref, quantity=bde_q(), fidelity=dft_gas,
                    source="synthetic", bond=bond,
                )
            )

    targets = {"train": 0.7, "val": 0.1, "test": 0.1, "frozen_test": 0.1}
    split = scaffold_split(records, targets)
    n = len(records)
    for partition, want in targets.items():
        got = len(getattr(split, partition)) / n
        assert abs(got - want) < 0.05, f"{partition}: wanted {want:.0%}, got {got:.1%}"


def test_skewed_groups_cannot_hit_the_targets_but_still_split(dft_gas):
    """A handful of huge scaffold families is a data property, not a bug.

    All the substituted benzenes share one Murcko scaffold, so train cannot
    reach 70% without breaking group integrity. The split must still be valid.
    """
    from rdkit import Chem

    smiles = set()
    for core in ("c1ccccc1{}", "c1ccncc1{}", "C1CCCCC1{}", "C1CCNCC1{}"):
        for sub in ("C", "CC", "Cl", "F", "O", "N", "OC", "C#N", "Br", "CCC"):
            mol = Chem.MolFromSmiles(core.format(sub))
            if mol is not None:
                smiles.add(Chem.MolToSmiles(mol))

    records = []
    for s in sorted(smiles):
        ref = MoleculeRef.from_smiles(s)
        for bond in all_xh_bonds(ref.smiles)[:4]:
            records.append(
                BDERecord(
                    molecule=ref, quantity=bde_q(), fidelity=dft_gas,
                    source="synthetic", bond=bond,
                )
            )

    split = scaffold_split(records)
    assert_no_group_leakage(records, split)
    assert sum(split.counts().values()) == len(records)
    for partition in ("train", "val", "test", "frozen_test"):
        assert len(getattr(split, partition)) > 0
