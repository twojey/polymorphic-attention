"""Tests catalog.report."""

from __future__ import annotations

import json

import pytest

from catalog.report import render_markdown_report


def test_render_empty() -> None:
    md = render_markdown_report({})
    assert "# Rapport batterie catalog" in md
    assert "unknown" in md  # oracle_id default


def test_render_minimal_metadata() -> None:
    data = {
        "metadata": {
            "oracle_id": "synthetic_test",
            "battery_name": "principal",
            "n_regimes": 3,
            "n_examples_per_regime": 32,
            "device": "cpu",
            "properties": ["A1", "B0"],
        },
        "per_regime": {},
        "cross_regime": {},
    }
    md = render_markdown_report(data)
    assert "synthetic_test" in md
    assert "principal" in md
    assert "N régimes** : 3" in md


def test_render_with_results() -> None:
    data = {
        "metadata": {
            "oracle_id": "test",
            "battery_name": "minimal",
            "properties": ["A1_r_eff"],
        },
        "per_regime": {
            "(0, 16, 0.0, ())": {
                "A1_r_eff": {"layer0_r_eff_median": 2.0, "layer0_r_eff_mean": 2.5},
            },
            "(1, 16, 0.0, ())": {
                "A1_r_eff": {"layer0_r_eff_median": 5.0, "layer0_r_eff_mean": 4.8},
            },
        },
        "cross_regime": {
            "M2_stress_variation": {"rho_spearman_r_eff_vs_omega": 0.95},
        },
    }
    md = render_markdown_report(data)
    assert "A1_r_eff" in md
    assert "M2_stress_variation" in md
    assert "rho_spearman_r_eff_vs_omega" in md
    # Tableau Property × régime
    assert "| Property |" in md
