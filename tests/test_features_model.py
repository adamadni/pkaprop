"""Featurisation and model shapes.

The declared feature-count constants and the tensors actually produced must
agree; a mismatch is a silent shape bug that only surfaces inside a matmul.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molprop.features import ATOM_FEATURES, BOND_FEATURES, featurise
from molprop.model import PKaGNN, PKaGNNConfig
from molprop.training import SiteExample, collate


def test_atom_feature_constant_matches_the_tensor():
    graph = featurise("CC(=O)Nc1ccccc1O")
    assert graph.atom_features.shape[1] == ATOM_FEATURES


def test_bond_feature_constant_matches_the_tensor():
    graph = featurise("CC(=O)Nc1ccccc1O")
    assert graph.edge_features.shape[1] == BOND_FEATURES


def test_graph_is_undirected():
    graph = featurise("CCO")
    assert graph.edge_index.shape[1] % 2 == 0
    src, dst = graph.edge_index
    assert set(zip(src.tolist(), dst.tolist())) == set(zip(dst.tolist(), src.tolist()))


def test_sites_are_found_and_in_range():
    graph = featurise("NCCC(=O)O")
    assert graph.acid_sites or graph.base_sites
    for idx in (*graph.acid_sites, *graph.base_sites):
        assert 0 <= idx < graph.n_atoms


def test_featurisation_works_without_xtb():
    """The physics features are optional; zero-filled when absent."""
    graph = featurise("Oc1ccccc1")
    assert np.allclose(graph.atom_features[:, -2:], 0.0)


def test_a_single_atom_molecule_does_not_crash():
    graph = featurise("O")
    assert graph.n_atoms == 1
    assert graph.edge_index.shape == (2, 0)


def _example(smiles: str) -> SiteExample:
    return SiteExample(featurise(smiles), 0, 4.5, True, "g", "test")


def test_forward_pass_returns_one_value_per_atom():
    batch = collate([_example("CC(=O)O"), _example("Oc1ccccc1")])
    model = PKaGNN(PKaGNNConfig(atom_features=ATOM_FEATURES, bond_features=BOND_FEATURES))
    acid, base = model(
        batch.atom_features, batch.edge_index, batch.edge_features,
        batch.batch_index, batch.n_graphs,
    )
    n_atoms = batch.atom_features.shape[0]
    assert acid.shape == (n_atoms,) and base.shape == (n_atoms,)


def test_batching_is_equivalent_to_running_singly():
    """Disjoint-graph batching must not leak information between molecules."""
    torch.manual_seed(0)
    model = PKaGNN(PKaGNNConfig(atom_features=ATOM_FEATURES, bond_features=BOND_FEATURES))
    model.eval()
    a, b = _example("CC(=O)O"), _example("Oc1ccccc1")
    with torch.no_grad():
        together = model(*_unpack(collate([a, b])))[0]
        alone_a = model(*_unpack(collate([a])))[0]
    n_a = a.graph.n_atoms
    assert torch.allclose(together[:n_a], alone_a, atol=1e-5)


def _unpack(batch):
    return (batch.atom_features, batch.edge_index, batch.edge_features,
            batch.batch_index, batch.n_graphs)


def test_site_rows_point_at_the_right_atoms():
    a, b = _example("CC(=O)O"), _example("Oc1ccccc1")
    batch = collate([a, b])
    assert batch.site_rows.tolist() == [0, a.graph.n_atoms]
