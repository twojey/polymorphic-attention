"""Tests Battery end-to-end avec SyntheticOracle."""

from __future__ import annotations

import torch

from catalog.batteries import Battery
from catalog.oracles import SyntheticOracle, RegimeSpec
from catalog.properties.family_a_spectral.a1_r_eff import A1ReffTheta99
from catalog.properties.family_b_structural.b1_toeplitz_distance import B1ToeplitzDistance
from catalog.properties.family_b_structural.b_identity_distance import B0IdentityDistance


def test_battery_basic_run_per_regime() -> None:
    """Battery exécute A1 + B1 sur 3 régimes synthétiques, produit résultats valides."""
    oracle = SyntheticOracle(structure="random", n_layers=2, n_heads=4, seq_len=16, seed=42)
    battery = Battery(
        properties=[A1ReffTheta99(theta=0.95), B1ToeplitzDistance()],
        name="test_battery",
        device="cpu", dtype=torch.float64,
    )
    results = battery.run(oracle, n_examples_per_regime=4)

    # 3 régimes par défaut dans SyntheticOracle.regime_grid()
    assert len(results.per_regime) == 3
    # Chaque régime a les 2 Properties
    for regime_key, regime_out in results.per_regime.items():
        assert "A1_r_eff_theta099" in regime_out
        assert "B1_toeplitz_distance" in regime_out


def test_battery_layered_output_keys() -> None:
    """Les clés des outputs sont préfixées par layer."""
    oracle = SyntheticOracle(structure="random", n_layers=3, seq_len=8, seed=0)
    battery = Battery(
        properties=[B0IdentityDistance()],
        name="test", device="cpu", dtype=torch.float64,
    )
    results = battery.run(oracle, n_examples_per_regime=2)
    sample = next(iter(results.per_regime.values()))
    keys = list(sample["B0_identity_distance"].keys())
    # On a layer0_*, layer1_*, layer2_* pour les n_layers=3 layers
    assert any("layer0_" in k for k in keys)
    assert any("layer2_" in k for k in keys)


def test_battery_metadata_populated() -> None:
    oracle = SyntheticOracle(structure="random", seq_len=8, seed=0, oracle_id="test_oracle")
    battery = Battery(
        properties=[B0IdentityDistance()],
        name="my_battery", device="cpu", dtype=torch.float64,
    )
    results = battery.run(oracle, n_examples_per_regime=2)
    assert results.metadata["battery_name"] == "my_battery"
    assert results.metadata["oracle_id"] == "test_oracle"
    assert results.metadata["domain"] == "synthetic"
    assert results.metadata["n_regimes"] == 3
    assert "B0_identity_distance" in results.metadata["properties"]


def test_battery_property_failure_does_not_kill_run() -> None:
    """Si une Property crash sur un layer, la Battery continue et log stderr."""
    class FailingProp(B0IdentityDistance):
        name = "failing_test_prop"
        def compute(self, A, ctx):
            if ctx.regime.get("layer") == 1:
                raise RuntimeError("simulated failure on layer 1")
            return super().compute(A, ctx)

    oracle = SyntheticOracle(structure="random", n_layers=3, seq_len=8, seed=0)
    battery = Battery(
        properties=[FailingProp()],
        name="test_resilient", device="cpu", dtype=torch.float64,
    )
    results = battery.run(oracle, n_examples_per_regime=2)
    # On a quand même des résultats (layer 0 et 2)
    assert len(results.per_regime) == 3
    sample = next(iter(results.per_regime.values()))
    keys = list(sample.get("failing_test_prop", {}).keys())
    assert any("layer0_" in k for k in keys)
    assert any("layer2_" in k for k in keys)
    # layer 1 absent
    assert not any("layer1_" in k for k in keys)


def test_battery_custom_regimes() -> None:
    """User-supplied régimes override le grid par défaut."""
    oracle = SyntheticOracle(structure="random", seq_len=8, seed=0)
    custom = [RegimeSpec(omega=8, delta=64, entropy=0.5)]
    battery = Battery(properties=[B0IdentityDistance()], name="custom")
    results = battery.run(oracle, regimes=custom, n_examples_per_regime=2)
    assert len(results.per_regime) == 1


def test_battery_results_to_dict_serializable() -> None:
    """Le to_dict produit du JSON-friendly."""
    import json
    oracle = SyntheticOracle(seq_len=8, seed=0)
    battery = Battery(properties=[B0IdentityDistance()])
    results = battery.run(oracle, n_examples_per_regime=2)
    d = results.to_dict()
    # Doit sérialiser sans crash
    json_str = json.dumps(d)
    assert len(json_str) > 0
