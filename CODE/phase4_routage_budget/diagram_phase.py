"""
diagram_phase.py — Diagramme de Phase et courbe Pareto λ_budget.

Spec : DOC/04 §4 (livrables) ; ROADMAP 4.6.

Le Diagramme de Phase représente R_target ∝ stress structurel — vérifie
que le Spectromètre alloue plus de rang quand le stress augmente.

La courbe Pareto λ_budget : pour chaque valeur de λ_budget, on récupère
(qualité, complexité moyenne) → on construit la frontière Pareto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseDiagramPoint:
    omega: float
    delta: float
    entropy: float
    R_target_mean: float
    R_target_p75: float


def build_phase_diagram(
    *,
    R_target: np.ndarray,           # (N,)
    omega: np.ndarray,
    delta: np.ndarray,
    entropy: np.ndarray,
) -> list[PhaseDiagramPoint]:
    """Construit un Diagramme de Phase agrégé par régime (ω, Δ, ℋ)."""
    rows: list[PhaseDiagramPoint] = []
    seen = set()
    for o, d, h in zip(omega, delta, entropy, strict=True):
        key = (float(o), float(d), float(h))
        if key in seen:
            continue
        seen.add(key)
        mask = (omega == o) & (delta == d) & (entropy == h)
        vals = R_target[mask]
        if vals.size == 0:
            continue
        rows.append(PhaseDiagramPoint(
            omega=float(o), delta=float(d), entropy=float(h),
            R_target_mean=float(vals.mean()),
            R_target_p75=float(np.quantile(vals, 0.75)),
        ))
    return rows


def is_phase_diagram_increasing(
    diagram: list[PhaseDiagramPoint], axis: str = "omega"
) -> bool:
    """Vérifie que R_target_mean croît avec l'axe de stress.

    Critère pré-enregistré phase 4 (DOC/04 §4.8).
    """
    points = [(getattr(p, axis), p.R_target_mean) for p in diagram]
    grouped: dict[float, list[float]] = {}
    for x, y in points:
        grouped.setdefault(x, []).append(y)
    sorted_keys = sorted(grouped.keys())
    means = [np.mean(grouped[k]) for k in sorted_keys]
    diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    if not diffs:
        return False
    return sum(d > 0 for d in diffs) > len(diffs) // 2  # majorité de diffs positives


@dataclass
class ParetoPoint:
    lambda_budget: float
    quality: float
    avg_rank: float
    avg_flops: float | None = None


def build_pareto_curve(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Frontière Pareto sur (avg_rank, quality) — minimiser rang, maximiser qualité.

    Retourne les points non dominés.
    """
    pareto: list[ParetoPoint] = []
    sorted_pts = sorted(points, key=lambda p: p.avg_rank)
    best_quality = -float("inf")
    for p in sorted_pts:
        if p.quality > best_quality:
            pareto.append(p)
            best_quality = p.quality
    return pareto
