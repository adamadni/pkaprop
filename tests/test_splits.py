"""No random splits, no group leakage, and the right holdout unit."""

from __future__ import annotations

import pytest
from conftest import find_atom, macro_record

import molprop.splits as splits_module
from molprop.records import MoleculeRef, PKaRecord
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
from molprop.units import LOG10_UNIT, Quantity, QuantityKind, water

MOLECULES = [
    "NCCC(=O)O", "NCCCC(=O)O", "OC(=O)c1ccccc1", "Nc1ccc(C(=O)O)cc1",
    "OC(=O)CCC(=O)O", "Oc1ccc2ccccc2c1", "CC(=O)Nc1ccc(O)cc1",
    "Cc1ccccc1N", "OC(=O)c1ccncc1", "CCN(CC)CCO",
]


def pka_records(experiment, steps: int = 2) -> list[PKaRecord]:
    """Each molecule contributes several titration steps."""
    records = []
    for m, smiles in enumerate(MOLECULES):
        for step in range(1, steps + 1):
            records.append(
                macro_record(
                    smiles, 3.0 + step, experiment, step=step,
                    # Date belongs to the molecule, not the step, so a time
                    # split cannot scatter one molecule across folds.
                    date=f"{2010 + m}-01-01",
                )
            )
    return records


def deprotonate(smiles: str, element: str = "O") -> tuple[int, str]:
    """Remove a proton from the first `element` that carries one."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != element or atom.GetTotalNumHs() < 1:
            continue
        rw = Chem.RWMol(mol)
        a = rw.GetAtomWithIdx(atom.GetIdx())
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
        a.SetFormalCharge(a.GetFormalCharge() - 1)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return atom.GetIdx(), Chem.MolToSmiles(out)
    raise AssertionError(f"no protonated {element} in {smiles}")


def micro(smiles: str, experiment, element: str = "O") -> PKaRecord:
    _, anion = deprotonate(smiles, element)
    return PKaRecord(
        molecule=MoleculeRef.from_smiles(smiles),
        quantity=Quantity(5.0, LOG10_UNIT, QuantityKind.PKA, water(), 298.15),
        fidelity=experiment,
        source="test",
        resolution="micro",
        site_atom_idx=find_atom(smiles, element, with_h=True),
        protonated_smiles=smiles,
        deprotonated_smiles=anion,
    )


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


def test_scaffold_split_never_scatters_a_molecule(experiment):
    records = pka_records(experiment)
    assert len(records) > len(MOLECULES)
    assert_no_group_leakage(records, scaffold_split(records))


def test_every_record_is_assigned_exactly_once(experiment):
    records = pka_records(experiment)
    assert sum(scaffold_split(records).counts().values()) == len(records)


def test_leakage_is_detected(experiment):
    records = pka_records(experiment)
    same = [i for i, r in enumerate(records) if r.group_key == records[0].group_key]
    assert len(same) >= 2
    rest = [i for i in range(len(records)) if i not in same[:2]]
    broken = Split(
        spec=SplitSpec("manual", {"train": 0.5, "val": 0.0, "test": 0.5, "frozen_test": 0.0}),
        train=(same[0], *rest), val=(), test=(same[1],), frozen_test=(),
    )
    with pytest.raises(LeakageError, match="must stay together"):
        assert_no_group_leakage(records, broken)


def test_a_record_cannot_be_in_two_partitions():
    with pytest.raises(SplitError, match="more than one partition"):
        Split(
            spec=SplitSpec("manual", {"train": 1.0, "val": 0.0, "test": 0.0, "frozen_test": 0.0}),
            train=(0, 1), val=(1,), test=(), frozen_test=(),
        )


# -- determinism and balance -----------------------------------------------


def test_scaffold_split_is_deterministic(experiment):
    records = pka_records(experiment)
    a, b = scaffold_split(records), scaffold_split(records)
    assert a.id == b.id and a.train == b.train and a.test == b.test


def test_split_id_depends_on_the_assignment(experiment):
    records = pka_records(experiment)
    a = scaffold_split(records)
    b = scaffold_split(records, {"train": 0.5, "val": 0.2, "test": 0.2, "frozen_test": 0.1})
    assert a.id != b.id


def test_every_partition_gets_at_least_one_group(experiment):
    split = scaffold_split(pka_records(experiment))
    for partition in ("train", "val", "test", "frozen_test"):
        assert len(getattr(split, partition)) > 0, f"{partition} is empty"


def test_a_frozen_test_partition_is_reserved(experiment):
    split = scaffold_split(pka_records(experiment))
    assert len(split.frozen_test) > 0
    assert not (set(split.frozen_test) & set(split.train))


def test_too_few_groups_is_an_error(experiment):
    with pytest.raises(SplitError, match="cannot populate"):
        scaffold_split([macro_record("CC(=O)O", 4.76, experiment)])


def test_realised_fractions_track_the_targets(experiment):
    records = []
    for n in range(1, 21):
        for step in (1, 2):
            records.append(
                macro_record("C" * n + "C(=O)O", 3.0 + step, experiment, step=step)
            )
    targets = {"train": 0.7, "val": 0.1, "test": 0.1, "frozen_test": 0.1}
    split = scaffold_split(records, targets)
    for partition, want in targets.items():
        got = len(getattr(split, partition)) / len(records)
        assert abs(got - want) < 0.06, f"{partition}: wanted {want:.0%}, got {got:.1%}"


# -- reaction centre -------------------------------------------------------


def test_macro_pka_has_no_reaction_centre(experiment):
    """A macroscopic value describes a whole titration, not one site."""
    with pytest.raises(SplitError, match="no reaction centre"):
        reaction_centre_key(macro_record("CC(=O)O", 4.76, experiment))


def test_micro_pka_reaction_centre_is_the_site(experiment):
    assert reaction_centre_key(micro("CC(=O)O", experiment))


def test_similar_sites_share_a_group(experiment):
    """Homologous carboxylic acids must not be split across folds."""
    keys = {reaction_centre_key(micro(s, experiment))
            for s in ("CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O")}
    assert len(keys) == 1


def test_different_site_chemistry_gives_different_groups(experiment):
    acid = reaction_centre_key(micro("CCC(=O)O", experiment))
    phenol = reaction_centre_key(micro("Oc1ccccc1", experiment))
    assert acid != phenol


def test_group_integrity_argument_is_validated(experiment):
    with pytest.raises(SplitError, match="group_integrity"):
        reaction_centre_split(pka_records(experiment), group_integrity="whatever")


# -- time split ------------------------------------------------------------


def test_time_split_requires_dates(experiment):
    with pytest.raises(SplitError, match="date on every record"):
        time_split([macro_record("CC(=O)O", 4.76, experiment)],
                   {"train": "2015-01-01", "val": "2018-01-01"})


def test_time_split_respects_the_cutoffs(experiment):
    records = pka_records(experiment)
    split = time_split(records, {"train": "2014-01-01", "val": "2016-01-01"})
    assert all(records[i].date < "2014-01-01" for i in split.train)
    assert all("2014-01-01" <= records[i].date < "2016-01-01" for i in split.val)
    assert all(records[i].date >= "2016-01-01" for i in (*split.test, *split.frozen_test))
    assert sum(split.counts().values()) == len(records)


def test_coverage_report_mentions_every_partition(experiment):
    records = pka_records(experiment)
    report = coverage_report(records, scaffold_split(records))
    for partition in ("train", "val", "test", "frozen_test"):
        assert partition in report
