"""ALFABET import: the bond-index remapping is the part that must not be wrong."""

from __future__ import annotations

import textwrap

import pytest
from rdkit import Chem

from molprop.chem import bond_label, canonical_smiles, indexed_mol
from molprop.datasets.alfabet import (
    ALFABET_FIDELITY,
    ALFABET_PROVENANCE,
    canonical_bond,
    load_alfabet,
)
from molprop.fidelity import Tier

# Real rows from rdf_data_190531.csv, including a molecule whose SMILES is not
# RDKit-canonical (the case where naive index reuse silently picks a wrong bond).
CSV = textwrap.dedent(
    """\
    rid,molecule,bond_index,fragment1,fragment2,bde,bond_type
    1,NCCCC(=O)O,0,[CH2]CCC(=O)O,[NH2],87.59900138,C-N
    2,NCCCC(=O)O,1,[CH2]CC(=O)O,[CH2]N,82.34424102,C-C
    3,NCCCC(=O)O,2,[CH2]C(=O)O,[CH2]CN,86.31386295,C-C
    4,CO,0,[CH3],[OH],92.0,C-O
    5,CO,1,[CH2]O,[H],96.5,C-H
    6,CO,4,C[O],[H],104.9,H-O
    """
)


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "bde.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


# -- fidelity --------------------------------------------------------------


def test_fidelity_is_fully_specified():
    assert ALFABET_FIDELITY.tier is Tier.DFT
    assert ALFABET_FIDELITY.basis == "def2-TZVP"
    assert ALFABET_FIDELITY.label() == "M06-2X/def2-TZVP"
    assert not ALFABET_FIDELITY.is_solution_phase


def test_conformer_protocol_does_not_claim_reproducibility():
    """No seed was published, so the protocol must not pretend to be rerunnable."""
    assert ALFABET_FIDELITY.conformers.generator == "provided"
    assert ALFABET_FIDELITY.conformers.seed is None


def test_provenance_carries_a_checksum():
    assert ALFABET_PROVENANCE["md5"] == "2e876544fda84a4b7d44d9f792ad50bd"
    assert ALFABET_PROVENANCE["license"] == "MIT"
    assert ALFABET_PROVENANCE["doi"].startswith("10.6084/")


# -- bond index remapping --------------------------------------------------


def test_bond_index_maps_to_the_right_element_pair():
    # CO: AddHs gives C(0) O(1) then H H H on C, H on O.
    assert bond_label(indexed_mol("CO"), *canonical_bond("CO", 0).as_tuple()) == "C-O"
    assert bond_label(indexed_mol("CO"), *canonical_bond("CO", 1).as_tuple()) == "C-H"
    assert bond_label(indexed_mol("CO"), *canonical_bond("CO", 4).as_tuple()) == "H-O"


def test_remapping_survives_a_non_canonical_input_smiles():
    """The mapping must follow the atoms, not the integer."""
    original = "C[C@@H]1CC(C)(C)O[C@H](C)O1"
    assert canonical_smiles(original) != original, "test needs a non-canonical SMILES"

    original_mol = Chem.AddHs(Chem.MolFromSmiles(original))
    for index in range(original_mol.GetNumBonds()):
        bond = original_mol.GetBondWithIdx(index)
        expected = "-".join(
            sorted(
                (
                    bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol(),
                )
            )
        )
        mapped = canonical_bond(original, index)
        observed = bond_label(
            indexed_mol(canonical_smiles(original)), *mapped.as_tuple()
        )
        assert observed == expected, f"bond {index} remapped to the wrong element pair"


def test_out_of_range_bond_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        canonical_bond("CO", 99)


# -- loading ---------------------------------------------------------------


def test_load_accepts_every_row(csv_file):
    records, report = load_alfabet(csv_file, verify_fragments_every=1)
    assert report.total_rows == 6
    assert report.accepted == 6
    assert report.rejected_bond_type == 0
    assert report.fragment_mismatch == 0, report.describe()
    assert len(records) == 6


def test_bond_types_match_the_published_column(csv_file):
    records, _ = load_alfabet(csv_file)
    assert [r.bond_type for r in records] == [
        "C-N", "C-C", "C-C", "C-O", "C-H", "H-O",
    ]


def test_values_are_gas_phase_bde_enthalpies(csv_file):
    records, _ = load_alfabet(csv_file)
    q = records[0].quantity
    assert q.kind.value == "BDE"
    assert q.unit.symbol == "kcal/mol"
    assert q.phase.medium == "gas"
    assert q.temperature_K == pytest.approx(298.15)


def test_acidic_xh_subset_is_identifiable(csv_file):
    records, _ = load_alfabet(csv_file)
    acidic = [r for r in records if r.is_acidic_xh]
    assert len(acidic) == 1 and acidic[0].bond_type == "H-O"


def test_records_of_one_molecule_share_a_group_key(csv_file):
    records, _ = load_alfabet(csv_file)
    keys = {r.group_key for r in records[:3]}
    assert len(keys) == 1, "a molecule's bonds must not scatter across folds"


def test_source_ref_preserves_the_upstream_row_id(csv_file):
    records, _ = load_alfabet(csv_file)
    assert records[0].source_ref == "rid:1"
    assert records[0].metadata["alfabet_bond_index"] == 0


def test_limit_is_respected(csv_file):
    records, report = load_alfabet(csv_file, limit=2)
    assert report.total_rows == 2 and len(records) == 2


def test_a_wrong_bond_type_is_rejected_not_imported(tmp_path):
    """The bond-type column is the cheap gate that catches an index error."""
    path = tmp_path / "bad.csv"
    path.write_text(
        "rid,molecule,bond_index,fragment1,fragment2,bde,bond_type\n"
        "1,CO,0,[CH3],[OH],92.0,C-H\n",  # bond 0 is C-O, not C-H
        encoding="utf-8",
    )
    records, report = load_alfabet(path)
    assert records == []
    assert report.rejected_bond_type == 1
    assert "ALFABET says C-H" in report.examples[0]
