"""
spearman.py — corrélations de Spearman avec bootstrap IC95% pour la matrice
3×3 du banc phase 1.5 (DOC/01b §2.2, §4).

Bootstrap parallélisé via multiprocessing.Pool (fork-shared x, y sur Linux) :
sur N CPU, gain quasi-linéaire jusqu'à ~min(N_cpu, n_boot).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import Pool, get_context

import numpy as np
from scipy import stats


@dataclass
class SpearmanResult:
    rho: float
    ci_low: float
    ci_high: float
    n: int


# Variables globales remplies par _init_worker via initializer.
# Évite de copier x, y dans chaque tâche (fork partage la mémoire COW).
_WORKER_X: np.ndarray | None = None
_WORKER_Y: np.ndarray | None = None


def _init_worker(x: np.ndarray, y: np.ndarray) -> None:
    global _WORKER_X, _WORKER_Y
    _WORKER_X = x
    _WORKER_Y = y


def _bootstrap_chunk(args: tuple[int, int]) -> np.ndarray:
    """Worker : exécute `n_iter` itérations bootstrap avec seed donnée."""
    seed, n_iter = args
    assert _WORKER_X is not None and _WORKER_Y is not None
    x, y = _WORKER_X, _WORKER_Y
    n = x.size
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        rhos[i] = float(stats.spearmanr(x[idx], y[idx]).statistic)
    return rhos


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    n_workers: int | None = None,
) -> SpearmanResult:
    """Spearman ρ avec IC bootstrap parallélisé.

    Sous-échantillonnage par paire (i, i) — pour respecter l'indépendance,
    cf. DOC/01b §8 : sous-échantillonner en amont avant d'appeler cette
    fonction (un token tous les K, par exemple).

    `n_workers=None` → utilise `os.cpu_count() - 1` par défaut. Pour
    désactiver le parallélisme (debug, environnement contraint) :
    `n_workers=1` → exécution séquentielle dans le process principal.
    """
    assert x.shape == y.shape and x.ndim == 1
    n = x.size
    if n < 4:
        return SpearmanResult(rho=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=n)

    rho_full = float(stats.spearmanr(x, y).statistic)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, n_boot)

    if n_workers <= 1:
        # Pas de parallélisme — chemin séquentiel (utile pour debug).
        rng = np.random.default_rng(seed)
        rhos = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            rhos[b] = float(stats.spearmanr(x[idx], y[idx]).statistic)
    else:
        # Découpe les n_boot itérations en n_workers chunks équilibrés.
        chunk_base = n_boot // n_workers
        remainder = n_boot % n_workers
        chunks: list[tuple[int, int]] = []
        for w in range(n_workers):
            n_iter = chunk_base + (1 if w < remainder else 0)
            if n_iter > 0:
                # Seeds distincts par worker pour indépendance reproductible
                chunks.append((seed + w * 100003, n_iter))

        # Fork context : partage x, y en COW (zero-copy sur Linux).
        ctx = get_context("fork")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(x, y)) as pool:
            results = pool.map(_bootstrap_chunk, chunks)
        rhos = np.concatenate(results)

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
