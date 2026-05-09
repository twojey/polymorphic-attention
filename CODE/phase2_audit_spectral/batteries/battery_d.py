"""
battery_d.py — Batterie D : détection out-of-catalogue.

Spec : DOC/02 §5c batterie D.

- D.1 : régimes orphelins (ε_min trop élevé même avec composition)
- D.2 : signature spectrale fréquentielle (cf. battery_b FFT)
- D.3 : eigendecomposition vs SVD — asymétrie non capturée par projecteurs symétriques
- D.4 : test non-linéaire A ≈ f(M) avec f non-linéaire simple (V2)

Régime orphelin = aucun projecteur (et leur composition additive) ne capture
la structure → la classe nécessaire n'est pas dans le catalogue actuel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class BatteryDResult:
    orphan_regimes: list[tuple] = field(default_factory=list)
    eigen_svd_asymmetry: dict[tuple, float] = field(default_factory=dict)
    hypotheses: list[str] = field(default_factory=list)


def detect_orphan_regimes(
    epsilon_per_regime: dict[tuple, dict[str, float]],
    *,
    additive_epsilon: dict[tuple, float] | None = None,
    threshold: float = 0.30,
) -> list[tuple]:
    """Régime orphelin : ε_min > threshold même avec composition additive."""
    orphans: list[tuple] = []
    for regime, eps_dict in epsilon_per_regime.items():
        eps_min = min(eps_dict.values())
        if additive_epsilon and regime in additive_epsilon:
            eps_min = min(eps_min, additive_epsilon[regime])
        if eps_min > threshold:
            orphans.append(regime)
    return orphans


def eigen_svd_asymmetry(A: torch.Tensor) -> float:
    """Mesure l'asymétrie : ‖A − A.T‖_F / ‖A‖_F.

    Si grande, la matrice est asymétrique et les projecteurs symétriques
    (Toeplitz/Hankel) ne capturent qu'une part. Suggère qu'une classe
    asymétrique manque au catalogue.
    """
    A_fp64 = A.to(torch.float64)
    asym = (A_fp64 - A_fp64.transpose(-1, -2)).norm()
    norm = A_fp64.norm().clamp_min(1e-30)
    return float((asym / norm).item())


def battery_d_analysis(
    *,
    epsilon_per_regime: dict[tuple, dict[str, float]],
    A_per_regime: dict[tuple, torch.Tensor],
    additive_epsilon: dict[tuple, float] | None = None,
    orphan_threshold: float = 0.30,
    asymmetry_threshold: float = 0.50,
) -> BatteryDResult:
    orphans = detect_orphan_regimes(
        epsilon_per_regime, additive_epsilon=additive_epsilon, threshold=orphan_threshold
    )
    asymmetry = {regime: eigen_svd_asymmetry(A) for regime, A in A_per_regime.items()}
    hypotheses: list[str] = []
    for regime in orphans:
        asym = asymmetry.get(regime, 0.0)
        if asym > asymmetry_threshold:
            hypotheses.append(
                f"Régime {regime} orphelin avec asymétrie {asym:.2f} → "
                f"hypothèse : classe asymétrique manquante (ex. forme triangulaire structurée)"
            )
        else:
            hypotheses.append(
                f"Régime {regime} orphelin sans asymétrie marquée → "
                f"hypothèse : classe non-linéaire ou compositionnelle hors catalogue"
            )
    return BatteryDResult(orphan_regimes=orphans, eigen_svd_asymmetry=asymmetry, hypotheses=hypotheses)
