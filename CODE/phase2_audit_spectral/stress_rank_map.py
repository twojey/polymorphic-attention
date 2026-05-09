"""
stress_rank_map.py — Stress-Rank Map enrichie (DOC/02 V3.5).

V3.5 : SCH comme distribution P(r_eff | stress, domaine), donc on rapporte :
- médiane, IQR
- p10, p90
- queue (p99)

Pour chaque axe (ω, Δ, ℋ), monovariate. Pour les paires, 2D.
"""

from __future__ import annotations

import numpy as np

from shared.aggregation import RegimeStats, aggregate_by_regime, aggregate_by_regime_2d


def build_monovariate_srm(
    *,
    r_eff_values: np.ndarray,        # (N_examples,) — un r_eff par exemple ou par token
    omega: np.ndarray,
    delta: np.ndarray,
    entropy: np.ndarray,
) -> dict[str, dict[float, RegimeStats]]:
    """Stress-Rank Map monovariate pour chaque axe."""
    out: dict[str, dict[float, RegimeStats]] = {"omega": {}, "delta": {}, "entropy": {}}
    monovariate = aggregate_by_regime(r_eff_values, omega=omega, delta=delta, entropy=entropy)
    for (axis, val), stats in monovariate.items():
        out[axis][val] = stats
    return out


def build_2d_srm(
    *,
    r_eff_values: np.ndarray,
    axis1_name: str, axis1: np.ndarray,
    axis2_name: str, axis2: np.ndarray,
) -> dict[tuple[float, float], RegimeStats]:
    """Stress-Rank Map 2D pour une paire d'axes."""
    return aggregate_by_regime_2d(
        r_eff_values, axis1_name=axis1_name, axis1=axis1.astype(float),
        axis2_name=axis2_name, axis2=axis2.astype(float),
    )


def median_grid(srm_2d: dict[tuple[float, float], RegimeStats]) -> tuple[list[float], list[float], np.ndarray]:
    """Convertit le SRM 2D en grille (xticks, yticks, matrix médiane)."""
    keys = list(srm_2d.keys())
    a1_vals = sorted({k[0] for k in keys})
    a2_vals = sorted({k[1] for k in keys})
    matrix = np.zeros((len(a1_vals), len(a2_vals)))
    for i, a in enumerate(a1_vals):
        for j, b in enumerate(a2_vals):
            stats = srm_2d.get((a, b))
            matrix[i, j] = stats.median if stats is not None else float("nan")
    return a1_vals, a2_vals, matrix


def iqr_grid(srm_2d: dict[tuple[float, float], RegimeStats]) -> tuple[list[float], list[float], np.ndarray]:
    keys = list(srm_2d.keys())
    a1_vals = sorted({k[0] for k in keys})
    a2_vals = sorted({k[1] for k in keys})
    matrix = np.zeros((len(a1_vals), len(a2_vals)))
    for i, a in enumerate(a1_vals):
        for j, b in enumerate(a2_vals):
            stats = srm_2d.get((a, b))
            matrix[i, j] = stats.iqr if stats is not None else float("nan")
    return a1_vals, a2_vals, matrix
