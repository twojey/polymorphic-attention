"""
test_6c_rmax_half.py — Test 6c (R_max réduit, V3.5).

Spec : DOC/05 §6c, ROADMAP 5.6b.

Test décisif pour la claim "ASP plus efficace que l'Oracle" :
- Calculer r_med = médiane(r_eff_oracle | régime moyen)
- Entraîner ASPLayer avec R_max = r_med / 2
- Évaluer qualité sur la suite phase 5

Critères :
- Succès strict : qualité ≥ 95% Oracle → ASP a structurellement dépassé l'Oracle
- Succès partiel : qualité ∈ [80%, 95%] → ASP compétitif, pas strictement supérieur

Statut : optionnel pour Pareto strict, OBLIGATOIRE pour toute claim de
supériorité structurelle vs Oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RMaxHalfResult:
    r_med_oracle: float
    R_max_used: int
    quality_asp: float                 # accuracy ASP avec R_max = r_med/2
    quality_oracle: float
    quality_ratio: float                # asp / oracle
    verdict: str                        # "strict" | "partial" | "fail"


def compute_r_med_oracle(r_eff_per_regime: dict[tuple, np.ndarray]) -> float:
    """Médiane globale du r_eff Oracle."""
    all_vals = []
    for vals in r_eff_per_regime.values():
        all_vals.extend(vals.tolist() if hasattr(vals, "tolist") else list(vals))
    if not all_vals:
        return 0.0
    return float(np.median(all_vals))


def evaluate_rmax_half(
    *,
    quality_asp: float,
    quality_oracle: float,
    R_max_used: int,
    r_med_oracle: float,
    strict_threshold: float = 0.95,
    partial_threshold: float = 0.80,
) -> RMaxHalfResult:
    ratio = quality_asp / max(quality_oracle, 1e-9)
    if ratio >= strict_threshold:
        verdict = "strict"
    elif ratio >= partial_threshold:
        verdict = "partial"
    else:
        verdict = "fail"
    return RMaxHalfResult(
        r_med_oracle=r_med_oracle,
        R_max_used=R_max_used,
        quality_asp=quality_asp,
        quality_oracle=quality_oracle,
        quality_ratio=ratio,
        verdict=verdict,
    )
