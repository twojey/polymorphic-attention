"""Tests cross_oracle harness."""

from __future__ import annotations

import torch

from catalog.cross_oracle import build_oracle_from_spec, compare_signatures
from catalog.oracles import SyntheticOracle


def test_build_synthetic_from_spec() -> None:
    spec = {"kind": "synthetic", "seq_len": 8, "n_layers": 2, "seed": 1}
    oracle = build_oracle_from_spec(spec)
    assert oracle.domain == "synthetic"
    assert oracle.n_layers == 2


def test_build_rejects_unknown_kind() -> None:
    import pytest

    with pytest.raises(ValueError, match="kind inconnu"):
        build_oracle_from_spec({"kind": "alien"})


def test_compare_signatures_empty() -> None:
    out = compare_signatures({})
    assert out == {}


def test_compare_signatures_basic() -> None:
    """Lance Battery sur 2 synthetic oracles puis compare."""
    from catalog.batteries import level_minimal

    oracle_1 = SyntheticOracle(seq_len=8, n_layers=1, seed=0)
    oracle_2 = SyntheticOracle(seq_len=8, n_layers=1, seed=1)
    battery = level_minimal()
    r1 = battery.run(oracle_1, n_examples_per_regime=4)
    r2 = battery.run(oracle_2, n_examples_per_regime=4)

    comparison = compare_signatures({"o1": r1, "o2": r2})
    assert comparison["n_oracles"] == 2
    assert "matrix" in comparison
    assert "o1" in comparison["matrix"]
    assert "o2" in comparison["matrix"]
