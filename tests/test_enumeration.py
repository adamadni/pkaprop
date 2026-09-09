"""Protonation microstate enumeration, including the SAMPL6 regression cases."""

from __future__ import annotations

import pytest
from rdkit import Chem

from molprop.chem import MoleculeError
from molprop.enumeration import (
    enumerate_microstates,
    find_protonation_sites,
    largest_fragment,
    species_key,
)


def labels(smiles: str) -> set[str]:
    return {s.label for s in find_protonation_sites(smiles)}


def states(smiles: str, **kw) -> dict[str, int]:
    return {m.smiles: m.charge for m in enumerate_microstates(smiles, **kw)}


# -- site detection --------------------------------------------------------


def test_carboxylic_acid_is_an_acid_site():
    assert "carboxylic_acid" in labels("CC(=O)O")


def test_aliphatic_amine_is_a_base_site():
    assert "aliphatic_amine" in labels("CCN(C)C")


def test_pyridine_nitrogen_is_a_base_site():
    assert "aromatic_n" in labels("c1ccncc1")


def test_phenol_is_an_acid_site():
    assert "phenol" in labels("Oc1ccccc1")


def test_amide_nitrogen_is_not_treated_as_a_base():
    """Amide N is a very poor base; treating it as one floods the enumeration."""
    base_labels = {s.label for s in find_protonation_sites("CC(=O)NC") if s.kind == "base"}
    assert "aliphatic_amine" not in base_labels


def test_alkane_has_no_sites():
    assert find_protonation_sites("CCCC") == []


def test_amino_acid_has_both_an_acid_and_a_base():
    kinds = {s.kind for s in find_protonation_sites("NCC(=O)O")}
    assert kinds == {"acid", "base"}


def test_site_indices_address_the_heavy_atom_mol():
    smiles = "NCC(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    for site in find_protonation_sites(smiles):
        assert 0 <= site.atom_idx < mol.GetNumAtoms()


# -- salts -----------------------------------------------------------------


def test_hcl_salt_is_stripped():
    """SM09 and SM12 in SAMPL6 ship as hydrochlorides."""
    assert "Cl" not in largest_fragment("COc1cccc(c1)Nc2c3ccccc3ncn2.Cl")


def test_stripping_keeps_the_organic_fragment():
    assert largest_fragment("CC(=O)O.[Na+]").startswith("CC(=O)O")


# -- enumeration -----------------------------------------------------------


def test_acetic_acid_gives_acid_and_conjugate_base():
    found = states("CC(=O)O")
    assert "CC(=O)O" in found
    assert "CC(=O)[O-]" in found


def test_methylamine_gives_neutral_and_cation():
    found = states("CN")
    assert found.get("CN") == 0
    assert any(c == 1 for c in found.values())


def test_glycine_reaches_the_zwitterion_charge_levels():
    charges = set(states("NCC(=O)O").values())
    assert {-1, 0, 1} <= charges


def test_charge_window_is_respected():
    for charge in states("NCCN", min_charge=0, max_charge=1).values():
        assert 0 <= charge <= 1


def test_max_changes_bounds_the_enumeration():
    few = states("NCCNCCN", max_changes=1)
    many = states("NCCNCCN", max_changes=3)
    assert len(few) < len(many)


def test_enumeration_is_deterministic():
    assert states("NCC(=O)O") == states("NCC(=O)O")


def test_the_input_structure_is_always_included():
    assert "Oc1ccccc1" in states("Oc1ccccc1")


def test_unparseable_input_raises():
    with pytest.raises(MoleculeError):
        enumerate_microstates("this is not a molecule[[[")


# -- species identity ------------------------------------------------------


def test_amidinium_resonance_forms_share_a_species_key():
    """Which nitrogen carries the proton is a drawing convention here."""
    a = "c1ccc(C[NH2+]c2ncnc3ccccc23)cc1"
    b = "c1ccc(C/[NH+]=c2/nc[nH]c3ccccc23)cc1"
    assert Chem.CanonSmiles(a) != Chem.CanonSmiles(b)
    assert species_key(a) == species_key(b)


def test_different_protonation_levels_do_not_share_a_species_key():
    """The InChIKey's FIRST BLOCK alone merges these -- the full key must not."""
    acid, base = species_key("CC(=O)O"), species_key("CC(=O)[O-]")
    assert acid != base
    assert acid.split("-")[0] == base.split("-")[0], (
        "first blocks are expected to collide; that is why the full key is used"
    )


def test_species_key_rejects_junk():
    with pytest.raises(MoleculeError):
        species_key("nonsense[[[")


# -- SAMPL6 regression -----------------------------------------------------
#
# Structures taken from SAMPL6 physical_properties/pKa/microstates. These are
# the cases the enumerator must not regress on.


@pytest.mark.parametrize(
    "parent,wanted",
    [
        # SM14_pKa1: benzimidazole aniline, neutral <-> cation
        ("Nc1ccc2c(c1)ncn2-c1ccccc1", "Nc1ccc2c(c1)[nH+]cn2-c1ccccc1"),
        # SM15: phenol deprotonation on a benzimidazole scaffold
        ("Oc1ccc(-n2cnc3ccccc32)cc1", "[O-]c1ccc(-n2cnc3ccccc32)cc1"),
        # SM07: 4-aminoquinazoline ring-nitrogen protonation
        ("c1ccc(CNc2ncnc3ccccc23)cc1", "c1ccc(CNc2[nH+]cnc3ccccc23)cc1"),
    ],
)
def test_sampl6_microstates_are_generated(parent, wanted):
    found = states(parent)
    assert Chem.CanonSmiles(wanted) in found


def test_sampl6_amidinium_reached_at_species_level():
    """SM07_micro006 is an imino tautomer; matched as a species, not a string."""
    generated = {species_key(s) for s in states("c1ccc(CNc2ncnc3ccccc23)cc1")}
    assert species_key("c1ccc(C/[NH+]=c2/nc[nH]c3ccccc23)cc1") in generated


# -- site classes found by auditing real data ------------------------------
#
# Each of these was a silent recall failure: a compound in the czodrowskilab
# corpus with a MEASURED pKa that the enumerator reported zero sites for.
# Auditing those 231 compounds is what produced this list.


@pytest.mark.parametrize(
    "smiles,expected,why",
    [
        ("C#CCNN", "hydrazine", "N-N was excluded by the amine pattern"),
        ("C1CCN(N2CCCC2)C1", "hydrazine", "cyclic hydrazine, pKa ~7"),
        ("CC(=O)N(C)O", "hydroxamic_or_hydroxylamine_OH",
         "O-H on nitrogen, invisible to the [OX2H1][CX4] alcohol pattern"),
        ("CC(=O)OO", "peroxy_acid", "peroxy acid, pKa ~8.2"),
        ("CC(C)S(=O)O", "sulfinic_acid", "sulfinic acid, pKa ~2"),
        ("CN(C)c1ccccc1", "aniline_N",
         "tertiary aniline: requiring an N-H dropped every N,N-dialkylaniline"),
        ("C/N=c1/sc2c(n1C)CCCC2", "imine",
         "imine carbon is aromatic; [CX3] never matches a ring carbon"),
    ],
)
def test_site_classes_recovered_from_the_pka_corpus(smiles, expected, why):
    assert expected in labels(smiles), why


def test_tertiary_aniline_is_a_base_not_an_acid():
    kinds = {s.kind for s in find_protonation_sites("CN(C)c1ccccc1")}
    assert "base" in kinds


def test_amide_is_still_not_treated_as_an_aniline_base():
    """Relaxing the aniline pattern must not swallow anilides."""
    base_labels = {s.label for s in find_protonation_sites("CC(=O)Nc1ccccc1")
                   if s.kind == "base"}
    assert "aniline_N" not in base_labels


def test_enumeration_stays_bounded_on_a_polyprotic_molecule():
    """New patterns must not blow up the state count."""
    assert len(enumerate_microstates("NCCNCCNCCO")) < 200
