"""Tests d'intégration end-to-end catalog (smoke + level_research + report)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from catalog.batteries import (
    level_extended, level_full, level_minimal, level_principal, level_research,
)
from catalog.oracles import SyntheticOracle
from catalog.report import render_markdown_report


@pytest.mark.parametrize("level_fn,expected_min_props", [
    (level_minimal, 1),
    (level_principal, 5),
    (level_extended, 10),
    (level_full, 15),
    (level_research, 50),
])
def test_each_level_runs_on_synthetic(level_fn, expected_min_props) -> None:
    """Chaque niveau de Battery tourne end-to-end sur SyntheticOracle."""
    oracle = SyntheticOracle(seq_len=8, n_layers=2, seed=0)
    battery = level_fn()
    assert len(battery.properties) >= expected_min_props
    results = battery.run(oracle, n_examples_per_regime=4)
    assert len(results.per_regime) > 0
    # Vérifier que chaque régime a au moins quelques sorties
    sample_regime = next(iter(results.per_regime.values()))
    assert len(sample_regime) > 0


def test_research_level_includes_all_families() -> None:
    """Niveau research doit couvrir au moins 15 familles distinctes."""
    battery = level_research()
    families = {p.family for p in battery.properties}
    assert len(families) >= 15


def test_full_pipeline_synthetic_to_report() -> None:
    """Pipeline complet : SyntheticOracle → research Battery → JSON → Markdown."""
    oracle = SyntheticOracle(seq_len=8, n_layers=2, seed=0)
    battery = level_research()
    results = battery.run(oracle, n_examples_per_regime=4)

    # Sérialisation JSON
    data = results.to_dict()
    serialized = json.dumps(data, default=str)
    assert isinstance(serialized, str)
    assert "metadata" in serialized

    # Rapport Markdown
    md = render_markdown_report(data)
    assert "# Rapport batterie catalog" in md
    assert "synthetic" in md.lower()


def test_battery_resilient_to_property_crash(tmp_path) -> None:
    """Une Property qui crash ne tue pas le run."""
    from catalog.batteries.base import Battery
    from catalog.properties.base import Property, PropertyContext

    class CrashingProperty(Property):
        name = "Crash_test"
        family = "A"
        cost_class = 1
        scope = "per_regime"

        def compute(self, A, ctx):
            raise RuntimeError("intentional crash")

    # Ajouter une Property qui marche au cas où on filtre les Properties cassées
    from catalog.properties.family_b_structural.b_identity_distance import (
        B0IdentityDistance,
    )

    battery = Battery(
        [CrashingProperty(), B0IdentityDistance()],
        name="test", device="cpu", dtype=torch.float64,
    )
    oracle = SyntheticOracle(seq_len=8, n_layers=1, seed=0)
    results = battery.run(oracle, n_examples_per_regime=2)
    # Le run doit avoir des résultats malgré le crash
    assert len(results.per_regime) > 0
    sample = next(iter(results.per_regime.values()))
    # B0 doit être dans les résultats
    assert "B0_identity_distance" in sample


def test_battery_handles_oracle_failure() -> None:
    """Une extraction qui échoue skip le régime, n'arrête pas le run."""
    from catalog.batteries.base import Battery
    from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec

    class FlakeyOracle(AbstractOracle):
        domain = "test"
        oracle_id = "flakey"
        n_layers = 1
        n_heads = 1

        def extract_regime(self, regime, n_examples):
            if regime.omega == 0:
                raise RuntimeError("flake on omega=0")
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
            return [
                RegimeSpec(omega=0, delta=16, entropy=0.0),
                RegimeSpec(omega=1, delta=16, entropy=0.0),
            ]

    from catalog.properties.family_b_structural.b_identity_distance import (
        B0IdentityDistance,
    )
    battery = Battery([B0IdentityDistance()], name="test", device="cpu", dtype=torch.float64)
    oracle = FlakeyOracle()
    results = battery.run(oracle, n_examples_per_regime=2)
    # Doit avoir 1 régime (omega=1), pas le omega=0 qui crash
    assert len(results.per_regime) == 1
