"""Tests phase 1 report : verdict go/no-go, recommandation R_max."""

from __future__ import annotations

import math

import numpy as np

from phase1_metrologie.report import evaluate_go_no_go, recommend_r_max, render_markdown_report
from shared.aggregation import RegimeStats, regime_stats


def _stats_constant(value: float, n: int = 100) -> RegimeStats:
    return regime_stats(np.full(n, value))


def test_recommend_r_max_uses_p90_with_margin() -> None:
    rank_stats = {
        ("omega", 1): _stats_constant(10),
        ("omega", 4): _stats_constant(40),
        ("delta", 16): _stats_constant(20),
    }
    r_max = recommend_r_max(rank_stats, seq_len=128, margin=1.5)
    # max p90 = 40, margin 1.5 → 60
    assert r_max == 60


def test_recommend_r_max_capped_by_seq_len() -> None:
    rank_stats = {("omega", 1): _stats_constant(100)}
    r_max = recommend_r_max(rank_stats, seq_len=64, margin=2.0)
    assert r_max == 64


def test_go_no_go_go_when_compressible() -> None:
    # Rang faible (10) sur séquence longue (128) : 10/128 < 0.5 → compressible
    rank_stats = {
        ("omega", 1): _stats_constant(10),
        ("omega", 2): _stats_constant(20),
        ("omega", 4): _stats_constant(30),
    }
    entropy_stats = {k: _stats_constant(1.0) for k in rank_stats}
    accuracy = {k: 0.95 for k in rank_stats}
    verdict = evaluate_go_no_go(
        rank_stats_per_regime=rank_stats,
        entropy_stats_per_regime=entropy_stats,
        accuracy_per_regime=accuracy,
        seq_len=128, max_rank_ratio=0.5, max_entropy_ratio=0.5,
        acc_floor=0.9, min_portion=0.1,
    )
    assert verdict.decision == "go"


def test_go_no_go_nogo_when_full_rank() -> None:
    rank_stats = {("omega", 1): _stats_constant(120)}  # ≈ N=128
    entropy_stats = {("omega", 1): _stats_constant(math.log(128) - 0.01)}
    accuracy = {("omega", 1): 0.95}
    verdict = evaluate_go_no_go(
        rank_stats_per_regime=rank_stats,
        entropy_stats_per_regime=entropy_stats,
        accuracy_per_regime=accuracy,
        seq_len=128, max_rank_ratio=0.5, max_entropy_ratio=0.5,
        acc_floor=0.9, min_portion=0.1,
    )
    assert verdict.decision == "no-go"


def test_render_markdown_smoke() -> None:
    rank_stats = {("omega", 1): _stats_constant(10)}
    entropy_stats = {("omega", 1): _stats_constant(0.5)}
    accuracy = {("omega", 1): 0.95}
    verdict = evaluate_go_no_go(
        rank_stats_per_regime=rank_stats, entropy_stats_per_regime=entropy_stats,
        accuracy_per_regime=accuracy, seq_len=128, max_rank_ratio=0.5,
        max_entropy_ratio=0.5, acc_floor=0.9, min_portion=0.1,
    )
    md = render_markdown_report(
        run_id="test", git_hash="abc1234", domain="smnist",
        rank_stats=rank_stats, entropy_stats=entropy_stats,
        accuracy=accuracy, verdict=verdict, r_max=15, figures=[],
    )
    assert "# Rapport phase 1" in md
    assert "GO" in md.upper()
    assert "abc1234" in md
