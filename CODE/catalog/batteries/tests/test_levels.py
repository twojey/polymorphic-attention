"""Tests pour les batteries pré-composées par niveau."""

from __future__ import annotations

import torch

from catalog.batteries import (
    level_extended,
    level_full,
    level_minimal,
    level_principal,
)
from catalog.oracles import SyntheticOracle


def test_level_minimal_only_cost_1_properties() -> None:
    bat = level_minimal()
    assert all(p.cost_class <= 1 for p in bat.properties)


def test_level_principal_includes_cost_1_and_2() -> None:
    bat = level_principal()
    assert all(p.cost_class <= 2 for p in bat.properties)
    cost_classes = {p.cost_class for p in bat.properties}
    assert 2 in cost_classes  # au moins une cost_class=2 attendue (A1, B1, B2 actuels)


def test_level_extended_supersets_principal() -> None:
    p_names = {p.name for p in level_principal().properties}
    e_names = {p.name for p in level_extended().properties}
    assert p_names.issubset(e_names)


def test_level_full_supersets_extended() -> None:
    e_names = {p.name for p in level_extended().properties}
    f_names = {p.name for p in level_full().properties}
    assert e_names.issubset(f_names)


def test_level_principal_runs_on_synthetic_oracle() -> None:
    """End-to-end : niveau principal sur SyntheticOracle, retourne résultats."""
    oracle = SyntheticOracle(structure="random", n_layers=2, seq_len=16, seed=0)
    bat = level_principal(device="cpu", dtype=torch.float64)
    results = bat.run(oracle, n_examples_per_regime=4)
    assert len(results.per_regime) == 3
    # Sample regime contient bien les Properties exécutées
    sample = next(iter(results.per_regime.values()))
    for p in bat.properties:
        assert p.name in sample, f"{p.name} absent des résultats"
