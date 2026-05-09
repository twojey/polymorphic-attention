"""
test_5e_ood.py — Test 5e (OOD croisé).

Spec : DOC/05, ROADMAP 5.7.

Train récursion (ω varié, Δ minimal) → eval binding (Δ varié, ω minimal).
Train binding → eval récursion.
Train mixte → eval tâches algorithmiques tierces.

Critère : R_target s'élève sur axe inédit, pas figé sur axe vu.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from phase5_pareto.abstract import ASPLayerEvaluable


@dataclass
class OODResult:
    R_target_train_axis_mean: float
    R_target_eval_axis_mean: float
    eval_axis_higher: bool             # R_target_eval > R_target_train ?
    elevation_ratio: float             # eval / train
    passed: bool


def run_ood_test(
    *,
    asp_layer: ASPLayerEvaluable,
    train_axis_tokens: torch.Tensor,        # batch sur axe entraîné
    train_axis_query_pos: torch.Tensor,
    eval_axis_tokens: torch.Tensor,         # batch sur axe inédit
    eval_axis_query_pos: torch.Tensor,
    elevation_threshold: float = 1.2,       # eval doit être ≥ 1.2× train
) -> OODResult:
    _, R_train = asp_layer.forward_eval(train_axis_tokens, train_axis_query_pos)
    _, R_eval = asp_layer.forward_eval(eval_axis_tokens, eval_axis_query_pos)
    train_mean = float(R_train.mean().item())
    eval_mean = float(R_eval.mean().item())
    ratio = eval_mean / max(train_mean, 1e-9)
    passed = ratio >= elevation_threshold
    return OODResult(
        R_target_train_axis_mean=train_mean,
        R_target_eval_axis_mean=eval_mean,
        eval_axis_higher=eval_mean > train_mean,
        elevation_ratio=ratio,
        passed=passed,
    )
