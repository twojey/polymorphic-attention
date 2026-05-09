"""
test_5a_identifiability.py — Test 5a (Identifiabilité).

Spec : DOC/05 §1, ROADMAP 5.1.

5a.i — Anti-fraude :
- Banc bruit blanc gaussien + tokens uniformes + ℋ max SSG
- Mesure distribution R_target
- Critère : R_target reste au plancher

5a.ii — Activation différentielle (V3.5) :
- 4 conditions : (a) bruit blanc, (b) null/empty, (c) répétition triviale, (d) input structuré
- diff_score = somme des KL divergences entre paires
- Critère : diff_score > seuil_actif ET R_target sur (d) > R_target sur (a, b, c)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from phase5_pareto.abstract import ASPLayerEvaluable


@dataclass
class AntiFraudResult:
    R_target_floor_observed: float       # médiane R_target sur bruit
    R_target_floor_threshold: float      # seuil acceptable
    passed: bool


@dataclass
class DifferentialActivationResult:
    diff_score: float
    threshold_active: float
    threshold_silent: float
    R_target_per_condition: dict[str, float]
    structured_above_others: bool
    passed: bool


def run_anti_fraud(
    *,
    asp_layer: ASPLayerEvaluable,
    noise_tokens: torch.Tensor,
    query_pos: torch.Tensor,
    floor_threshold: float = 1.0,
) -> AntiFraudResult:
    """5a.i : sur du bruit, R_target doit rester au plancher (≈ 1)."""
    _, R_target = asp_layer.forward_eval(noise_tokens, query_pos)
    median = float(R_target.median().item())
    return AntiFraudResult(
        R_target_floor_observed=median,
        R_target_floor_threshold=floor_threshold,
        passed=median <= floor_threshold,
    )


def _kl_divergence_distributions(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) entre deux distributions discrètes."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def run_differential_activation(
    *,
    asp_layer: ASPLayerEvaluable,
    conditions: dict[str, tuple[torch.Tensor, torch.Tensor]],   # name -> (tokens, query_pos)
    threshold_active: float = 1.0,
    threshold_silent: float = 0.1,
    R_max_for_hist: int | None = None,
) -> DifferentialActivationResult:
    """5a.ii : 4 conditions. diff_score = sum KL pairwise.

    `conditions` doit contenir les clés "noise", "null", "trivial", "structured".
    """
    R_target_means: dict[str, float] = {}
    distributions: dict[str, np.ndarray] = {}
    R_max = R_max_for_hist or asp_layer.R_max
    bins = np.arange(0, R_max + 2)
    for name, (tokens, qpos) in conditions.items():
        _, R_target = asp_layer.forward_eval(tokens, qpos)
        R_np = R_target.cpu().numpy().reshape(-1)
        R_target_means[name] = float(R_np.mean())
        hist, _ = np.histogram(R_np, bins=bins)
        distributions[name] = hist.astype(np.float64)

    keys = list(distributions.keys())
    diff_score = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff_score += _kl_divergence_distributions(distributions[keys[i]], distributions[keys[j]])

    structured_above = (
        "structured" in R_target_means
        and all(
            R_target_means["structured"] > R_target_means[k]
            for k in ("noise", "null", "trivial")
            if k in R_target_means
        )
    )
    return DifferentialActivationResult(
        diff_score=diff_score,
        threshold_active=threshold_active,
        threshold_silent=threshold_silent,
        R_target_per_condition=R_target_means,
        structured_above_others=structured_above,
        passed=(diff_score > threshold_active) and structured_above,
    )
