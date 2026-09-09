"""Registry provenance and the no-result-without-a-baseline rule."""

from __future__ import annotations

import pytest
from conftest import macro_record

from molprop.cards import CardError, Evaluation, Metric, ResultCard
from molprop.registry import Registry, RegistryError, UnitsAudit
from molprop.splits import scaffold_split
from molprop.units import LOG10_UNIT, KCAL_PER_MOL, Phase, Quantity, QuantityKind, UnitError, water

MOLECULES = [
    "CC(=O)O", "CCC(=O)O", "OC(=O)c1ccccc1", "Oc1ccccc1",
    "Oc1ccc(C)cc1", "NCCC(=O)O", "CCN(CC)CC", "OC(=O)c1ccncc1",
]


def pka_records(experiment):
    """One record per titration step; molecules contribute several."""
    out = []
    for m, smiles in enumerate(MOLECULES):
        for step in (1, 2):
            out.append(macro_record(smiles, 3.0 + step + 0.1 * m, experiment, step=step))
    return out


# -- registry --------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path, experiment):
    reg = Registry(tmp_path)
    records = pka_records(experiment)
    manifest = reg.write("synthetic-bde", "v1", records, license="CC-BY-4.0")
    loaded, loaded_manifest = reg.load("synthetic-bde", "v1")
    assert len(loaded) == len(records)
    assert loaded_manifest.content_sha256 == manifest.content_sha256
    assert {r.record_id for r in loaded} == {r.record_id for r in records}
    assert loaded[0].quantity.unit.symbol == "log10"


def test_content_hash_mismatch_is_fatal(tmp_path, experiment):
    reg = Registry(tmp_path)
    reg.write("synthetic-bde", "v1", pka_records(experiment))
    path = tmp_path / "synthetic-bde" / "v1" / "records.jsonl"
    tampered = path.read_text(encoding="utf-8").replace("4.0", "9.9", 1)
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(RegistryError, match="content hash mismatch"):
        reg.load("synthetic-bde", "v1")


def test_registered_versions_are_immutable(tmp_path, experiment):
    reg = Registry(tmp_path)
    records = pka_records(experiment)
    reg.write("synthetic-bde", "v1", records)
    with pytest.raises(RegistryError, match="bump the version"):
        reg.write("synthetic-bde", "v1", records)


def test_empty_datasets_are_refused(tmp_path):
    with pytest.raises(RegistryError, match="empty dataset"):
        Registry(tmp_path).write("nothing", "v1", [])


def test_mixed_solvents_fail_the_units_audit(tmp_path, experiment):
    from molprop.records import MoleculeRef, PKaRecord

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


def test_audit_records_what_is_present(experiment):
    audit = UnitsAudit.of(pka_records(experiment))
    n = len(pka_records(experiment))
    assert audit.kinds == {"pKa": n}
    assert audit.units == {"log10": n}
    assert all(p.startswith("solution") for p in audit.phases)
    assert audit.problems() == []


def test_datasets_are_listed(tmp_path, experiment):
    reg = Registry(tmp_path)
    reg.write("a", "v1", pka_records(experiment))
    reg.write("a", "v2", pka_records(experiment))
    assert reg.datasets() == [("a", "v1"), ("a", "v2")]


# -- result cards ----------------------------------------------------------


def make_split(experiment):
    records = pka_records(experiment)
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


def test_a_card_requires_a_baseline(experiment):
    _, split = make_split(experiment)
    with pytest.raises(CardError, match="no result without a baseline"):
        ResultCard(
            model=evaluation("pKa GNN", split.id, 0.55),
            baselines=(),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_baseline_must_share_the_split(experiment):
    _, split = make_split(experiment)
    with pytest.raises(CardError, match="not the model's split"):
        ResultCard(
            model=evaluation("pKa GNN", split.id, 0.55),
            baselines=(evaluation("site-class median", "otherhash123", 3.2),),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_baseline_must_share_the_partition(experiment):
    _, split = make_split(experiment)
    with pytest.raises(CardError, match="partition"):
        ResultCard(
            model=evaluation("pKa GNN", split.id, 0.55),
            baselines=(evaluation("site-class median", split.id, 3.2, partition="val"),),
            dataset_name="synthetic-bde",
            dataset_version="v1",
            dataset_sha256="deadbeef",
            split=split,
        )


def test_card_split_must_match_the_model(experiment):
    _, split = make_split(experiment)
    with pytest.raises(CardError, match="carries split"):
        ResultCard(
            model=evaluation("pKa GNN", "wronghash", 0.55),
            baselines=(evaluation("site-class median", "wronghash", 3.2),),
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


def test_incompatible_metric_units_are_caught(experiment):
    _, split = make_split(experiment)
    model = Evaluation(
        name="pKa GNN",
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


def test_a_complete_card_renders(experiment):
    _, split = make_split(experiment)
    card = ResultCard(
        model=evaluation("pKa GNN", split.id, 0.55),
        baselines=(
            evaluation(
                "site-class median",
                split.id,
                3.21,
                description="median training pKa per ionisable-site class",
            ),
            evaluation("global median", split.id, 8.4, description="median training pKa"),
        ),
        dataset_name="synthetic-bde",
        dataset_version="v1",
        dataset_sha256="deadbeefcafebabe0000",
        split=split,
        notes="scaffold split at molecule level",
    )
    md = card.render_markdown()
    assert "pKa GNN" in md and "site-class median" in md and "global median" in md
    assert "kcal/mol" in md
    assert card.comparison("mae") == [
        ("pKa GNN", 0.55),
        ("site-class median", 3.21),
        ("global median", 8.4),
    ]
    assert "pKa GNN" in card.to_json()


# -- serialisation layout --------------------------------------------------


def test_fidelity_is_stored_once_in_the_manifest(tmp_path, experiment):
    """A dataset has few fidelities and many rows; inlining wastes most of the file."""
    import json

    reg = Registry(tmp_path)
    records = pka_records(experiment)
    manifest = reg.write("compact", "v1", records)

    assert list(manifest.fidelities) == [experiment.id]
    assert manifest.fidelities[experiment.id]["method"] == "experiment"

    lines = (tmp_path / "compact" / "v1" / "records.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    row = json.loads(lines[0])
    assert row["fidelity_id"] == experiment.id
    assert "fidelity" not in row, "the fidelity must not be repeated in every row"


def test_null_fields_are_not_stored(tmp_path, experiment):
    import json

    reg = Registry(tmp_path)
    reg.write("compact", "v1", pka_records(experiment))
    row = json.loads(
        (tmp_path / "compact" / "v1" / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "date" not in row and "uncertainty" not in row
    # ph was never set, so it must not be stored; solvent WAS set and must be.
    assert "ph" not in row["quantity"]["phase"]
    assert row["quantity"]["phase"]["solvent"] == "water"


def test_referenced_fidelity_round_trips(tmp_path, experiment):
    reg = Registry(tmp_path)
    written = pka_records(experiment)
    reg.write("compact", "v1", written)
    loaded, _ = reg.load("compact", "v1")
    assert {r.fidelity.id for r in loaded} == {experiment.id}
    assert {r.record_id for r in loaded} == {r.record_id for r in written}
    assert loaded[0].fidelity.conformers.ensemble_treatment == (
        experiment.conformers.ensemble_treatment
    )


def test_inline_fidelity_rows_still_load(tmp_path, experiment):
    """The earlier layout inlined the fidelity; those files must still read."""
    import json

    from molprop.registry import _record_from_dict

    record = pka_records(experiment)[0]
    inline = record.as_dict()  # as_dict still emits the full fidelity
    assert "fidelity" in inline
    restored = _record_from_dict(inline)
    assert restored.fidelity.id == experiment.id
    assert restored.record_id == record.record_id


def test_a_row_referencing_an_unknown_fidelity_is_fatal(tmp_path, experiment):
    """Manifest and records file must not be able to drift apart silently."""
    from molprop.registry import _record_from_dict

    row = pka_records(experiment)[0].as_dict()
    row.pop("fidelity")
    row["fidelity_id"] = "deadbeef1234"
    with pytest.raises(RegistryError, match="come apart"):
        _record_from_dict(row, {})


def test_compaction_does_not_change_record_identity(tmp_path, experiment):
    reg = Registry(tmp_path)
    written = pka_records(experiment)
    reg.write("compact", "v1", written)
    loaded, _ = reg.load("compact", "v1")
    by_id = {r.record_id: r for r in loaded}
    for r in written:
        other = by_id[r.record_id]
        assert other.step_index == r.step_index
        assert other.quantity.canonical_value == pytest.approx(r.quantity.canonical_value)
        assert other.group_key == r.group_key
