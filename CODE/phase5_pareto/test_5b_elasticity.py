"""
test_5b_elasticity.py — Test 5b (Élasticité).

Spec : DOC/05 §2, ROADMAP 5.2.

Générateur séquences sandwich [bruit][structure][bruit].
Mesure courbe R_target(t) en fonction de la position dans la séquence.
Calcul du **Lag de Réaction** : nombre de tokens entre l'entrée dans la
zone structurée et le moment où R_target dépasse un seuil de seuil.

Critère : Lag < seuil pré-enregistré ET descente symétrique.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from phase5_pareto.abstract import ASPLayerEvaluable


@dataclass
class ElasticityResult:
    lag_to_rise: int            # tokens avant que R_target ne monte
    lag_to_fall: int            # tokens avant que R_target ne descende
    symmetric: bool
    passed: bool
    R_target_curve: list[float]


def detect_lag(
    R_target: np.ndarray,
    *,
    structure_start: int,
    structure_end: int,
    threshold: float,
    direction: str = "up",
) -> int:
    """Détecte le lag entre une transition de zone et la réponse R_target.

    direction = "up" : compte tokens depuis structure_start jusqu'à R_target ≥ threshold
    direction = "down" : compte tokens depuis structure_end jusqu'à R_target ≤ threshold
    """
    if direction == "up":
        for i in range(structure_start, len(R_target)):
            if R_target[i] >= threshold:
                return i - structure_start
        return len(R_target) - structure_start
    else:
        for i in range(structure_end, len(R_target)):
            if R_target[i] <= threshold:
                return i - structure_end
        return len(R_target) - structure_end


def run_elasticity_test(
    *,
    asp_layer: ASPLayerEvaluable,
    sandwich_tokens: torch.Tensor,         # (1, N) — un sandwich
    query_pos: torch.Tensor,               # (1,)
    structure_start: int,
    structure_end: int,
    threshold: float = 2.0,
    max_lag: int = 8,
    symmetry_tolerance: int = 3,
) -> ElasticityResult:
    _, R_target = asp_layer.forward_eval(sandwich_tokens, query_pos)
    R = R_target.squeeze(0).cpu().numpy()
    lag_up = detect_lag(R, structure_start=structure_start, structure_end=structure_end,
                       threshold=threshold, direction="up")
    lag_down = detect_lag(R, structure_start=structure_start, structure_end=structure_end,
                         threshold=threshold, direction="down")
    symmetric = abs(lag_up - lag_down) <= symmetry_tolerance
    passed = (lag_up <= max_lag) and (lag_down <= max_lag) and symmetric
    return ElasticityResult(
        lag_to_rise=lag_up,
        lag_to_fall=lag_down,
        symmetric=symmetric,
        passed=passed,
        R_target_curve=R.tolist(),
    )
