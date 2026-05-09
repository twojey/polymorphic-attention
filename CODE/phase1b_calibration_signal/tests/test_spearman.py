"""Tests Spearman + critères phase 1.5."""

from __future__ import annotations

import numpy as np

from phase1b_calibration_signal.bench.spearman import (
    bootstrap_spearman_ci,
    passes_phase1b_criteria,
    signal_correlations,
)


def test_spearman_perfect_correlation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 3 * x  # monotone parfait
    result = bootstrap_spearman_ci(x, y, n_boot=200, seed=0)
    assert result.rho > 0.99
    assert result.ci_low > 0.95


def test_spearman_no_correlation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    result = bootstrap_spearman_ci(x, y, n_boot=200, seed=0)
    assert abs(result.rho) < 0.2


def test_signal_correlations_matrix() -> None:
    rng = np.random.default_rng(0)
    n = 300
    omega = rng.integers(0, 12, size=n).astype(np.float64)
    delta = rng.integers(0, 4096, size=n).astype(np.float64)
    entropy = rng.uniform(0, 1, size=n).astype(np.float64)
    s_kl = 0.8 * omega / 12 + 0.1 * rng.normal(size=n)
    s_spectral = 0.85 * delta / 4096 + 0.1 * rng.normal(size=n)
    s_grad = rng.normal(size=n)  # bruit pur

    signals = {"S_KL": s_kl, "S_Spectral": s_spectral, "S_Grad": s_grad}
    stress = {"omega": omega, "delta": delta, "entropy": entropy}
    corr = signal_correlations(signals=signals, stress=stress, n_boot=200, seed=0)
    verdict = passes_phase1b_criteria(corr, threshold_structural=0.7, threshold_noise=0.2)

    # S_KL devrait passer sur omega
    assert verdict["S_KL"]["passed"], verdict["S_KL"]
    assert "omega" in verdict["S_KL"]["axes_covered"]
    # S_Spectral devrait passer sur delta
    assert verdict["S_Spectral"]["passed"], verdict["S_Spectral"]
    assert "delta" in verdict["S_Spectral"]["axes_covered"]
    # S_Grad ne doit PAS passer
    assert not verdict["S_Grad"]["passed"], verdict["S_Grad"]
