"""Registry provenance and the no-result-without-a-baseline rule."""

from __future__ import annotations

import pytest

from molprop.cards import CardError, Evaluation, Metric, ResultCard
from molprop.chem import BondRef, indexed_mol
from molprop.records import BDERecord, MoleculeRef, PKaRecord
from molprop.registry import Registry, RegistryError, UnitsAudit
from molprop.splits import scaffold_split
from molprop.units import (
    GAS,
    KCAL_PER_MOL,
    LOG10_UNIT,
    Phase,
    Quantity,
    QuantityKind,
    UnitError,
    water,
)

# Deliberately scaffold-diverse: a scaffold split needs at least as many
# distinct scaffolds as it has partitions.
MOLECULES = [
    "Oc1ccccc1",
    "CCO",
    "c1ccc2ccccc2c1",
    "OC1CCCCC1",
    "c1ccncc1",
    "C1CCNCC1",
]


def xh_bonds(smiles: str) -> list[BondRef]:
    mol = indexed_mol(smiles)
    return [
        BondRef(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if "H" in {
            mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol(),
            mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol(),
        }
    ]


def bde_records(dft_gas):
    out = []
    for smiles in MOLECULES:
        ref = MoleculeRef.from_smiles(smiles)
        for k, bond in enumerate(xh_bonds(ref.smiles)):
            out.append(
                BDERecord(
                    molecule=ref,
                    quantity=Quantity(
                        90.0 + k, KCAL_PER_MOL, QuantityKind.BDE, GAS, 298.15
                    ),
                    fidelity=dft_gas,
                    source="synthetic",
                    bond=bond,
                )
            )
    return out


# -- registry --------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path, dft_gas):
    reg = Registry(tmp_path)
    records = bde_records(dft_gas)
    manifest = reg.write("synthetic-bde", "v1", records, license="CC-BY-4.0")
    loaded, loaded_manifest = reg.load("synthetic-bde", "v1")
    assert len(loaded) == len(records)
    assert loaded_manifest.content_sha256 == manifest.content_sha256
    assert {r.record_id for r in loaded} == {r.record_id for r in records}
    assert loaded[0].quantity.unit.symbol == "kcal/mol"


def test_content_hash_mismatch_is_fatal(tmp_path, dft_gas):
    reg = Registry(tmp_path)
    reg.write("synthetic-bde", "v1", bde_records(dft_gas))
    path = tmp_path / "synthetic-bde" / "v1" / "records.jsonl"
    tampered = path.read_text(encoding="utf-8").replace("90.0", "99.0", 1)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(RegistryError, match="content hash mismatch"):
        reg.load("synthetic-bde", "v1")


def test_registered_versions_are_immutable(tmp_path, dft_gas):
    reg = Registry(tmp_path)
    records = bde_records(dft_gas)
    reg.write("synthetic-bde", "v1", records)
    with pytest.raises(RegistryError, match="bump the version"):
        reg.write("synthetic-bde", "v1", records)


def test_empty_datasets_are_refused(tmp_path):
    with pytest.raises(RegistryError, match="empty dataset"):
        Registry(tmp_path).write("nothing", "v1", [])


def test_mixed_solvents_fail_the_units_audit(tmp_path, experiment):
    aqueous = PKaRecord(
        molecule=MoleculeRef.from_smiles("CC(=O)O"),
        quantity=Quantity(4.76, LOG10_UNIT, QuantityKind.PKA, water(), 298.15),
        fidelity=experiment,
        source="iupac",
        resolution="macro",
        step_index=1,
    )
    methanolic = PKaRecord(
        molecule=MoleculeRef.from_smiles("OC(=O)c1ccccc1"),
        quantity=Quantity(
            9.4, LOG10_UNIT, QuantityKind.PKA, Phase("solution", "methanol"), 298.15
        ),
        fidelity=experiment,
        source="literature",
        resolution="macro",
        step_index=1,
    )
    reg = Registry(tmp_path)
    with pytest.raises(RegistryError, match="mixed solvents"):
        reg.write("pka-mixed", "v1", [aqueous, methanolic])
    # deliberate mixing is allowed, but must be stated
    reg.write("pka-mixed", "v1", [aqueous, methanolic], allow_mixed_conditions=True)


def test_audit_records_what_is_present(dft_gas):
    audit = UnitsAudit.of(bde_records(dft_gas))
    assert audit.kinds == {"BDE": len(bde_records(dft_gas))}
    assert audit.units == {"kcal/mol": len(bde_records(dft_gas))}
    assert list(audit.phases) == ["gas"]
    assert audit.problems() == []


def test_datasets_are_listed(tmp_path, dft_gas):
    reg = Registry(tmp_path)
    reg.write("a", "v1", bde_records(dft_gas))
    reg.write("a", "v2", bde_records(dft_gas))
    assert reg.datasets() == [("a", "v1"), ("a", "v2")]


# -- result cards ----------------------------------------------------------


def make_split(dft_gas):
    records = bde_records(dft_gas)
    return records, scaffold_split(records)


def evaluation(name, split_id, mae, *, partition="test", description=None):
    return Evaluation(
        name=name,
        split_id=split_id,
        partition=partition,
        description=description,
        metrics=(
            Metric("mae", mae, KCAL_PER_MOL, n=100),
            Metric("r2", 0.9),
        ),
    )


def test_a_card_requires_a_baseline(dft_gas):
    _, split = make_split(dft_gas)
    with pytest.raises(CardError, match="no result without a baseline"):
        ResultCard(
            model=evaluation("delta-gnn", split.id, 0.55),
            baselines=(),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_baseline_must_share_the_split(dft_gas):
    _, split = make_split(dft_gas)
    with pytest.raises(CardError, match="not the model's split"):
        ResultCard(
            model=evaluation("delta-gnn", split.id, 0.55),
            baselines=(evaluation("bond-type mean", "otherhash123", 3.2),),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_baseline_must_share_the_partition(dft_gas):
    _, split = make_split(dft_gas)
    with pytest.raises(CardError, match="partition"):
        ResultCard(
            model=evaluation("delta-gnn", split.id, 0.55),
            baselines=(evaluation("bond-type mean", split.id, 3.2, partition="val"),),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_card_split_must_match_the_model(dft_gas):
    _, split = make_split(dft_gas)
    with pytest.raises(CardError, match="carries split"):
        ResultCard(
            model=evaluation("delta-gnn", "wronghash", 0.55),
            baselines=(evaluation("bond-type mean", "wronghash", 3.2),),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_dimensioned_metric_needs_a_unit():
    with pytest.raises(UnitError, match="has no unit"):
        Metric("mae", 0.55)


def test_dimensionless_metric_may_omit_a_unit():
    assert Metric("spearman", 0.91).unit is None


def test_incompatible_metric_units_are_caught(dft_gas):
    _, split = make_split(dft_gas)
    model = Evaluation(
        name="delta-gnn",
        split_id=split.id,
        partition="test",
        metrics=(Metric("mae", 0.55, KCAL_PER_MOL, n=100),),
    )
    baseline = Evaluation(
        name="lookup",
        split_id=split.id,
        partition="test",
        metrics=(Metric("mae", 1.2, LOG10_UNIT, n=100),),
    )
    with pytest.raises(UnitError, match="incompatible units"):
        ResultCard(
            model=model,
            baselines=(baseline,),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_a_complete_card_renders(dft_gas, xtb_gas, bde_ladder):
    from molprop.fidelity import DeltaSpec

    _, split = make_split(dft_gas)
    card = ResultCard(
        model=evaluation("delta-gnn", split.id, 0.55),
        baselines=(
            evaluation(
                "bond-type mean",
                split.id,
                3.21,
                description="mean BDE per element-pair label, fit on train",
            ),
            evaluation("raw xTB", split.id, 8.4, description="uncorrected GFN2-xTB"),
        ),
        dataset_name="synthetic-bde",
        dataset_version="v1",
        dataset_sha256="deadbeefcafebabe0000",
        split=split,
        delta=DeltaSpec(base=xtb_gas, target=dft_gas, ladder=bde_ladder),
        notes="direct-vs-delta ablation pending",
    )
    md = card.render_markdown()
    assert "delta-gnn" in md and "bond-type mean" in md and "raw xTB" in md
    assert "kcal/mol" in md
    assert card.comparison("mae") == [
        ("delta-gnn", 0.55),
        ("bond-type mean", 3.21),
        ("raw xTB", 8.4),
    ]
    assert "delta-gnn" in card.to_json()


# -- serialisation layout --------------------------------------------------


def test_fidelity_is_stored_once_in_the_manifest(tmp_path, dft_gas):
    """A dataset has few fidelities and many rows; inlining wastes most of the file."""
    import json

    reg = Registry(tmp_path)
    records = bde_records(dft_gas)
    manifest = reg.write("compact", "v1", records)

    assert list(manifest.fidelities) == [dft_gas.id]
    assert manifest.fidelities[dft_gas.id]["method"] == "M06-2X"

    lines = (tmp_path / "compact" / "v1" / "records.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    row = json.loads(lines[0])
    assert row["fidelity_id"] == dft_gas.id
    assert "fidelity" not in row, "the fidelity must not be repeated in every row"


def test_null_fields_are_not_stored(tmp_path, dft_gas):
    import json

    reg = Registry(tmp_path)
    reg.write("compact", "v1", bde_records(dft_gas))
    row = json.loads(
        (tmp_path / "compact" / "v1" / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "date" not in row and "uncertainty" not in row
    assert "solvent" not in row["quantity"]["phase"]


def test_referenced_fidelity_round_trips(tmp_path, dft_gas):
    reg = Registry(tmp_path)
    written = bde_records(dft_gas)
    reg.write("compact", "v1", written)
    loaded, _ = reg.load("compact", "v1")
    assert {r.fidelity.id for r in loaded} == {dft_gas.id}
    assert {r.record_id for r in loaded} == {r.record_id for r in written}
    assert loaded[0].fidelity.conformers.ensemble_treatment == (
        dft_gas.conformers.ensemble_treatment
    )


def test_inline_fidelity_rows_still_load(tmp_path, dft_gas):
    """The earlier layout inlined the fidelity; those files must still read."""
    import json

    from molprop.registry import _record_from_dict

    record = bde_records(dft_gas)[0]
    inline = record.as_dict()  # as_dict still emits the full fidelity
    assert "fidelity" in inline
    restored = _record_from_dict(inline)
    assert restored.fidelity.id == dft_gas.id
    assert restored.record_id == record.record_id


def test_a_row_referencing_an_unknown_fidelity_is_fatal(tmp_path, dft_gas):
    """Manifest and records file must not be able to drift apart silently."""
    from molprop.registry import _record_from_dict

    row = bde_records(dft_gas)[0].as_dict()
    row.pop("fidelity")
    row["fidelity_id"] = "deadbeef1234"
    with pytest.raises(RegistryError, match="come apart"):
        _record_from_dict(row, {})


def test_compaction_does_not_change_record_identity(tmp_path, dft_gas):
    reg = Registry(tmp_path)
    written = bde_records(dft_gas)
    reg.write("compact", "v1", written)
    loaded, _ = reg.load("compact", "v1")
    by_id = {r.record_id: r for r in loaded}
    for r in written:
        other = by_id[r.record_id]
        assert other.bond.as_tuple() == r.bond.as_tuple()
        assert other.quantity.canonical_value == pytest.approx(r.quantity.canonical_value)
        assert other.group_key == r.group_key
