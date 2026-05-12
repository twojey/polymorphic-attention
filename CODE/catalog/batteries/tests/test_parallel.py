"""Tests dispatch parallèle régimes (n_workers > 1)."""

from __future__ import annotations

import time

import pytest
import torch

from catalog.batteries.base import Battery
from catalog.oracles import SyntheticOracle
from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec
from catalog.properties.base import Property, PropertyContext


class _SlowProperty(Property):
    """Property qui sleep N ms (simule un calcul SVD coûteux)."""
    name = "Slow_test"
    family = "A"
    cost_class = 3
    scope = "per_regime"

    def __init__(self, sleep_ms: int = 50) -> None:
        self.sleep_ms = sleep_ms

    def compute(self, A, ctx):
        time.sleep(self.sleep_ms / 1000.0)
        return {"dummy": float(A.mean().item())}


class _SlowOracle(AbstractOracle):
    """Oracle qui sleep N ms par extract_regime (simule forward Oracle coûteux)."""
    oracle_id = "slow_oracle"
    domain = "test"
    n_layers = 1
    n_heads = 1

    def __init__(self, sleep_ms: int = 50, seq_len: int = 8) -> None:
        self.sleep_ms = sleep_ms
        self.seq_len = seq_len

    def extract_regime(self, regime, n_examples):
        time.sleep(self.sleep_ms / 1000.0)
        attn = torch.softmax(
            torch.randn(n_examples, 1, self.seq_len, self.seq_len), dim=-1,
        )
        return AttentionDump(
            attn=[attn],
            omegas=torch.full((n_examples,), regime.omega or 0.0),
            deltas=torch.full((n_examples,), regime.delta or 0.0),
            entropies=torch.zeros(n_examples),
            tokens=torch.zeros(n_examples, self.seq_len, dtype=torch.long),
            query_pos=torch.zeros(n_examples, dtype=torch.long),
            metadata={},
        )

    def regime_grid(self):
        return [RegimeSpec(omega=i, delta=8, entropy=0.0) for i in range(4)]


def test_battery_n_workers_validation() -> None:
    """n_workers < 1 lève ValueError."""
    with pytest.raises(ValueError, match="n_workers"):
        Battery([_SlowProperty()], n_workers=0)


def test_battery_n_workers_default_is_sequential() -> None:
    """n_workers=1 = comportement séquentiel original."""
    battery = Battery([_SlowProperty(sleep_ms=10)], name="seq")
    assert battery.n_workers == 1


def test_battery_parallel_produces_same_results() -> None:
    """Séquentiel et parallèle donnent les mêmes per_regime keys + Properties."""
    oracle = SyntheticOracle(seq_len=8, n_layers=2, seed=0)
    grid = oracle.regime_grid()
    prop = _SlowProperty(sleep_ms=5)
    bat_seq = Battery([prop], name="seq", n_workers=1)
    bat_par = Battery([prop], name="par", n_workers=4)
    r_seq = bat_seq.run(oracle, regimes=grid, n_examples_per_regime=2)
    r_par = bat_par.run(oracle, regimes=grid, n_examples_per_regime=2)
    assert set(r_seq.per_regime.keys()) == set(r_par.per_regime.keys())
    for key in r_seq.per_regime:
        seq_props = set(r_seq.per_regime[key].keys())
        par_props = set(r_par.per_regime[key].keys())
        assert seq_props == par_props


def test_battery_parallel_speedup() -> None:
    """4 régimes × sleep 80ms : parallel 4-workers nettement plus rapide.

    Sur CPU avec GIL libre durant time.sleep (et torch.linalg.* qui libère),
    on attend speedup ≥ 1.5×.
    """
    oracle = _SlowOracle(sleep_ms=80, seq_len=8)
    prop = _SlowProperty(sleep_ms=80)
    grid = oracle.regime_grid()

    bat_seq = Battery([prop], name="seq", n_workers=1)
    t0 = time.time()
    bat_seq.run(oracle, regimes=grid, n_examples_per_regime=1)
    dt_seq = time.time() - t0

    bat_par = Battery([prop], name="par", n_workers=4)
    t0 = time.time()
    bat_par.run(oracle, regimes=grid, n_examples_per_regime=1)
    dt_par = time.time() - t0

    speedup = dt_seq / max(dt_par, 1e-6)
    assert speedup >= 1.5, f"speedup={speedup:.2f}× insuffisant (seq={dt_seq:.2f}s, par={dt_par:.2f}s)"


def test_battery_parallel_handles_crashing_property() -> None:
    """Une Property qui crash sur un régime n'arrête pas les autres en parallèle."""
    class _Flakey(Property):
        name = "Flakey_test"
        family = "A"
        cost_class = 1
        scope = "per_regime"

        def compute(self, A, ctx):
            if ctx.regime.get("omega") == 2:
                raise RuntimeError("flake on omega=2")
            return {"ok": 1.0}

    oracle = SyntheticOracle(seq_len=8, n_layers=1, seed=0)
    grid = oracle.regime_grid()  # several regimes
    battery = Battery([_Flakey()], name="test", n_workers=4)
    results = battery.run(oracle, regimes=grid, n_examples_per_regime=2)
    # Tous les régimes apparaissent (même celui où Property crash : regime_out vide)
    assert len(results.per_regime) == len(grid)


def test_battery_parallel_handles_oracle_failure() -> None:
    """Si extract_regime échoue sur un régime, il est skipé sans bloquer les autres."""

    class _CrashOracle(AbstractOracle):
        oracle_id = "crash"
        domain = "test"
        n_layers = 1
        n_heads = 1

        def extract_regime(self, regime, n_examples):
            if regime.omega == 1:
                raise RuntimeError("crash on omega=1")
            attn = torch.softmax(torch.randn(n_examples, 1, 4, 4), dim=-1)
            return AttentionDump(
                attn=[attn],
                omegas=torch.full((n_examples,), regime.omega or 0.0),
                deltas=torch.full((n_examples,), regime.delta or 0.0),
                entropies=torch.zeros(n_examples),
                tokens=torch.zeros(n_examples, 4, dtype=torch.long),
                query_pos=torch.zeros(n_examples, dtype=torch.long),
            )

        def regime_grid(self):
            return [RegimeSpec(omega=i, delta=4, entropy=0.0) for i in range(3)]

    from catalog.properties.family_b_structural.b_identity_distance import (
        B0IdentityDistance,
    )

    battery = Battery([B0IdentityDistance()], name="test", n_workers=2)
    oracle = _CrashOracle()
    results = battery.run(oracle, n_examples_per_regime=2)
    # 2 régimes OK (omega=0, omega=2), 1 régime skipé (omega=1)
    assert len(results.per_regime) == 2


def test_battery_parallel_metadata_records_n_workers() -> None:
    """results.metadata garde la trace de n_workers utilisé."""
    oracle = SyntheticOracle(seq_len=4, n_layers=1, seed=0)
    battery = Battery([_SlowProperty(sleep_ms=1)], name="test", n_workers=3)
    results = battery.run(oracle, regimes=oracle.regime_grid()[:2],
                          n_examples_per_regime=1)
    assert results.metadata["n_workers"] == 3
