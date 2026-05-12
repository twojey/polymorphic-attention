"""Tests orchestrateur livrables.run_all."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_fake_results(oracle_id: str, prop_values: dict) -> dict:
    return {
        "metadata": {"oracle_id": oracle_id, "battery_name": "test"},
        "per_regime": {
            "(0, 16, 0.0, ())": {
                prop: {"median": v} for prop, v in prop_values.items()
            },
            "(2, 16, 0.0, ())": {
                prop: {"median": v * 1.1} for prop, v in prop_values.items()
            },
        },
        "cross_regime": {},
    }


def test_run_all_minimal(tmp_path: Path) -> None:
    """run_all sur 2 Oracles, sans predictions, produit tous les artefacts core."""
    from livrables.run_all import run_all
    res = {
        "OR": _make_fake_results("OR", {"A1": 5.0, "B1": 0.10}),
        "DV": _make_fake_results("DV", {"A1": 3.0, "B1": 0.05}),
    }
    artifacts = run_all(res, tmp_path / "out")
    # Au moins ces artefacts attendus :
    expected = {"signatures_table_json", "signatures_table_md",
                "signatures_variance_md", "signatures_per_oracle_json",
                "signatures_per_oracle_md", "index", "figure_signatures"}
    assert expected.issubset(set(artifacts.keys()))
    for name, path in artifacts.items():
        assert path.is_file(), f"{name} : path {path} not file"


def test_run_all_with_predictions(tmp_path: Path) -> None:
    """run_all avec predictions YAML produit predictions_evaluation."""
    from livrables.run_all import run_all
    # Crée un YAML minimaliste
    pred_path = tmp_path / "preds.yaml"
    pred_path.write_text("""
- prop_metric: A1.median
  oracle: OR
  prediction: low
  threshold_low: 10.0
""")
    res = {"OR": _make_fake_results("OR", {"A1": 5.0})}
    artifacts = run_all(res, tmp_path / "out", predictions_path=pred_path)
    assert "predictions_json" in artifacts
    assert "predictions_md" in artifacts
    # Vérifier le contenu
    with open(artifacts["predictions_json"]) as f:
        data = json.load(f)
    assert data["summary"]["n_total"] == 1


def test_run_all_no_results_raises(tmp_path: Path) -> None:
    """Pas de results → ValueError explicite."""
    from livrables.run_all import run_all
    with pytest.raises(ValueError, match="Aucun"):
        run_all({}, tmp_path / "out")


def test_run_all_missing_predictions_does_not_break(tmp_path: Path) -> None:
    """Predictions YAML absent → log warning, autres livrables OK."""
    from livrables.run_all import run_all
    res = {"OR": _make_fake_results("OR", {"A1": 5.0})}
    artifacts = run_all(res, tmp_path / "out",
                        predictions_path=tmp_path / "nope.yaml")
    assert "predictions_json" not in artifacts
    assert "signatures_table_json" in artifacts


def test_run_all_index_lists_all_artifacts(tmp_path: Path) -> None:
    from livrables.run_all import run_all
    res = {"OR": _make_fake_results("OR", {"A1": 5.0}),
           "DV": _make_fake_results("DV", {"A1": 3.0})}
    artifacts = run_all(res, tmp_path / "out")
    index_md = artifacts["index"].read_text()
    assert "signatures_table" in index_md
    assert "OR" in index_md
    assert "DV" in index_md
