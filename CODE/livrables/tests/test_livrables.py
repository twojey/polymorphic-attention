"""Tests livrables — synthèse cross-Oracle, predictions, signatures, verdict, figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_fake_results(oracle_id: str, prop_values: dict) -> dict:
    """Construit un fake results.json avec des Properties contrôlées."""
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


def test_build_signatures_table() -> None:
    from livrables.cross_oracle_synthesis import build_signatures_table
    fake_a = _make_fake_results("OracleA", {"A1": 5.0, "B1": 0.10})
    fake_b = _make_fake_results("OracleB", {"A1": 15.0, "B1": 0.50})
    table, md = build_signatures_table({"OracleA": fake_a, "OracleB": fake_b})
    assert "A1.median" in table
    assert table["A1.median"]["OracleA"] < table["A1.median"]["OracleB"]
    assert "OracleA" in md
    assert "OracleB" in md


def test_compute_signature_variance() -> None:
    from livrables.cross_oracle_synthesis import compute_signature_variance
    fake_a = _make_fake_results("OracleA", {"A1": 5.0, "B1": 0.5})
    fake_b = _make_fake_results("OracleB", {"A1": 50.0, "B1": 0.5})
    variances = compute_signature_variance({"OracleA": fake_a, "OracleB": fake_b})
    # A1 doit avoir variance plus grande que B1
    var_dict = dict(variances)
    assert var_dict["A1.median"] > var_dict["B1.median"]


def test_evaluate_prediction_validated() -> None:
    from livrables.partie1_predictions_vs_measured import evaluate_prediction
    fake = _make_fake_results("DV", {"A1": 5.0})
    pred = {"prop_metric": "A1.median", "oracle": "DV",
            "prediction": "low", "threshold_low": 10.0}
    out = evaluate_prediction(pred, {"DV": fake})
    assert out["validated"] is True
    assert out["measured"] is not None


def test_evaluate_prediction_refuted() -> None:
    from livrables.partie1_predictions_vs_measured import evaluate_prediction
    fake = _make_fake_results("SC", {"A1": 50.0})
    pred = {"prop_metric": "A1.median", "oracle": "SC",
            "prediction": "low", "threshold_low": 10.0}
    out = evaluate_prediction(pred, {"SC": fake})
    assert out["validated"] is False


def test_build_signature_for_oracle() -> None:
    from livrables.partie1_signatures import build_signature_for_oracle
    fake = _make_fake_results("Test", {
        "A1_r_eff_theta099": 3.0,  # low (< 16)
        "B1_toeplitz_distance": 0.05,  # toeplitz (< 0.10)
    })
    # Adapter le format pour matcher la structure que la fonction attend
    fake["per_regime"]["(0, 16, 0.0, ())"]["A1_r_eff_theta099"] = {"r_eff_median": 3.0}
    fake["per_regime"]["(0, 16, 0.0, ())"]["B1_toeplitz_distance"] = {"toeplitz_distance_relative_median": 0.05}
    sig = build_signature_for_oracle(fake, "Test", N_max=64)
    assert sig["oracle_id"] == "Test"
    assert any("low-rank" in s for s in sig["properties_satisfied"])


def test_build_verdict_go() -> None:
    from livrables.partie2_asp_verdict import build_verdict
    test_results = {
        "5a": {"passed": True}, "5b": {"passed": True},
        "5c": {"passed": True}, "5d": {"passed": True},
        "5e": {"passed": True}, "6c": {"passed": True},
    }
    v = build_verdict(test_results)
    assert v.verdict == "GO"
    assert v.mandatory_passed == 3
    assert v.bonus_passed == 3


def test_build_verdict_partial() -> None:
    from livrables.partie2_asp_verdict import build_verdict
    test_results = {
        "5a": {"passed": True}, "5b": {"passed": False},
        "5c": {"passed": True}, "5d": {"passed": True},
        "5e": {"passed": True}, "6c": {"passed": True},
    }
    v = build_verdict(test_results)
    assert v.verdict == "PARTIAL"
    assert v.mandatory_passed == 3
    assert v.bonus_passed == 2


def test_build_verdict_nogo() -> None:
    from livrables.partie2_asp_verdict import build_verdict
    test_results = {"5a": {"passed": False}, "5c": {"passed": True},
                    "6c": {"passed": True}}
    v = build_verdict(test_results)
    assert v.verdict == "NO-GO"


def test_paper_figures_signatures(tmp_path: Path) -> None:
    from livrables.paper_figures import generate_figure_signatures
    table = {
        "A1.median": {"OR": 2.5, "LL": 5.0, "SC": 20.0},
        "B1.median": {"OR": 0.1, "LL": 0.2, "SC": 0.3},
    }
    out = generate_figure_signatures(table, tmp_path / "sig.png")
    assert out.is_file()


def test_paper_figures_predictions(tmp_path: Path) -> None:
    from livrables.paper_figures import generate_figure_predictions_vs_measured
    detailed = [
        {"validated": True}, {"validated": True}, {"validated": False},
        {"validated": None},
    ]
    out = generate_figure_predictions_vs_measured(detailed, tmp_path / "pred.png")
    assert out.is_file()


def test_paper_figures_pareto(tmp_path: Path) -> None:
    from livrables.paper_figures import generate_figure_pareto_curves
    data = {"Oracle": [(1e9, 0.95), (1e10, 0.97)],
            "ASP": [(1e8, 0.90), (1e9, 0.93), (1e10, 0.94)]}
    out = generate_figure_pareto_curves(data, tmp_path / "pareto.png")
    assert out.is_file()
