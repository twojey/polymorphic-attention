"""
spearman.py — corrélations de Spearman avec bootstrap IC95% pour la matrice
3×3 du banc phase 1.5 (DOC/01b §2.2, §4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SpearmanResult:
    rho: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> SpearmanResult:
    """Spearman ρ avec IC bootstrap.

    Sous-échantillonnage par paire (i, i) — pour respecter l'indépendance,
    cf. DOC/01b §8 : sous-échantillonner en amont avant d'appeler cette
    fonction (un token tous les K, par exemple).
    """
    assert x.shape == y.shape and x.ndim == 1
    n = x.size
    if n < 4:
        return SpearmanResult(rho=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=n)
    rho_full = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[b] = float(stats.spearmanr(x[idx], y[idx]).statistic)
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return SpearmanResult(rho=rho_full, ci_low=lo, ci_high=hi, n=n)


def signal_correlations(
    *,
    signals: dict[str, np.ndarray],   # nom -> (N_tokens,) valeur
    stress: dict[str, np.ndarray],    # "omega" / "delta" / "entropy" -> (N_tokens,)
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[tuple[str, str], SpearmanResult]:
    """Calcule la matrice ρ pour chaque (signal, axe de stress)."""
    out: dict[tuple[str, str], SpearmanResult] = {}
    for s_name, s_vals in signals.items():
        for axis, axis_vals in stress.items():
            out[(s_name, axis)] = bootstrap_spearman_ci(
                s_vals, axis_vals, n_boot=n_boot, seed=seed
            )
    return out


def passes_phase1b_criteria(
    correlations: dict[tuple[str, str], SpearmanResult],
    *,
    threshold_structural: float = 0.70,
    threshold_noise: float = 0.20,
) -> dict[str, dict[str, object]]:
    """Applique les critères DOC/01b §4 à la matrice de corrélations.

    Pour chaque signal : passe si max(|ρ_ω|, |ρ_Δ|) > threshold_structural
    ET |ρ_ℋ| < threshold_noise.
    """
    signals = {s for s, _ in correlations.keys()}
    out: dict[str, dict[str, object]] = {}
    for s in signals:
        rho_omega = abs(correlations[(s, "omega")].rho)
        rho_delta = abs(correlations[(s, "delta")].rho)
        rho_entropy = abs(correlations[(s, "entropy")].rho)
        max_struct = max(rho_omega, rho_delta)
        passed = (max_struct > threshold_structural) and (rho_entropy < threshold_noise)
        out[s] = {
            "passed": passed,
            "max_structural_rho": max_struct,
            "noise_rho": rho_entropy,
            "axes_covered": [
                axis for axis, rho in [("omega", rho_omega), ("delta", rho_delta)]
                if rho > threshold_structural
            ],
        }
    return out
