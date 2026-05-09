"""Smoke tests des helpers plotting (vérifie qu'ils retournent une Figure sans exception)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.plotting import (
    correlation_matrix_heatmap,
    heatmap_2d,
    monovariate_curve,
    save_figure,
    stress_rank_map,
)


def test_monovariate_curve_smoke(tmp_path) -> None:
    fig = monovariate_curve(
        x=np.array([1, 2, 4, 8]),
        y_mean=np.array([1.0, 1.5, 2.0, 3.0]),
        y_p10=np.array([0.8, 1.2, 1.6, 2.4]),
        y_p90=np.array([1.2, 1.8, 2.4, 3.6]),
        title="test",
    )
    p = save_figure(fig, tmp_path / "mono.png")
    assert p.exists()
    assert p.stat().st_size > 100
    plt.close("all")


def test_heatmap_2d_smoke(tmp_path) -> None:
    fig = heatmap_2d(
        matrix=np.random.rand(4, 5),
        xticks=[0, 16, 64, 256, 1024],
        yticks=[0, 1, 2, 4],
    )
    p = save_figure(fig, tmp_path / "heat.png")
    assert p.exists()
    plt.close("all")


def test_correlation_heatmap_smoke(tmp_path) -> None:
    fig = correlation_matrix_heatmap(
        rho_matrix=np.array([[0.8, 0.1, 0.05], [0.2, 0.9, 0.0], [0.1, 0.1, 0.1]]),
        signal_names=["S_KL", "S_Spectral", "S_Grad"],
        axis_names=["ω", "Δ", "ℋ"],
    )
    p = save_figure(fig, tmp_path / "corr.png")
    assert p.exists()
    plt.close("all")


def test_stress_rank_map_smoke(tmp_path) -> None:
    fig = stress_rank_map(
        omega_values=[0, 1, 2, 4],
        delta_values=[0, 16, 64, 256],
        r_eff_median=np.random.rand(4, 4),
        r_eff_iqr=np.random.rand(4, 4),
    )
    p = save_figure(fig, tmp_path / "srm.png")
    assert p.exists()
    plt.close("all")
