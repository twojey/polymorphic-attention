"""
sanity.py — sanity checks ASPLayer phase 3.

Spec : DOC/03 §3.6 — quatre checks obligatoires :

1. **Saturation** : à m_t = 1 partout (r = R_max), atteint la borne phase 1.
2. **Effondrement** : à m_t = 0 partout (r = 0), ≡ Backbone seul.
3. **Monotonie** : qualité décroît monotonement quand r baisse.
4. **Lissité** : |q(r+1) − 2q(r) + q(r−1)| < seuil (pas de saut).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch

from phase3_kernel_asp.asp_layer import ASPLayer


@dataclass
class SanityResult:
    saturation_passed: bool = False
    saturation_quality: float = 0.0
    saturation_oracle: float = 0.0
    collapse_passed: bool = False
    collapse_diff: float = 0.0
    monotone_passed: bool = False
    smoothness_passed: bool = False
    smoothness_max_jump: float = 0.0


@torch.no_grad()
def sanity_check_collapse(
    asp_layer: ASPLayer, x: torch.Tensor, *, atol: float = 1e-5
) -> tuple[bool, float]:
    """À r=0, ASPLayer doit être ≡ Backbone (norm appliquée des deux côtés
    quand on compare). On compare les deux sorties mises au même traitement.
    """
    asp_layer.eval()
    backbone_only = asp_layer.norm(asp_layer.backbone(x))
    asp_r0 = asp_layer.forward_with_rank(x, 0)
    diff = (backbone_only - asp_r0).abs().max().item()
    return diff < atol, diff


@torch.no_grad()
def sanity_check_monotone_quality(
    asp_layer: ASPLayer,
    x: torch.Tensor,
    *,
    quality_fn: Callable[[torch.Tensor], float],
    rank_grid: list[int],
) -> tuple[bool, list[float]]:
    """quality_fn(output) doit retourner un scalaire (acc, -loss, etc.) qui
    croît avec la qualité. On vérifie que c'est monotone croissant en r.
    """
    asp_layer.eval()
    qualities: list[float] = []
    for r in sorted(rank_grid):
        out = asp_layer.forward_with_rank(x, r)
        qualities.append(quality_fn(out))
    monotone = all(qualities[i] <= qualities[i + 1] + 1e-6 for i in range(len(qualities) - 1))
    return monotone, qualities


def sanity_check_smoothness(qualities: list[float], *, max_jump: float = 0.5) -> tuple[bool, float]:
    """|q(r+1) − 2q(r) + q(r−1)| < seuil (deuxième différence)."""
    if len(qualities) < 3:
        return True, 0.0
    second_diffs = [
        abs(qualities[i + 1] - 2 * qualities[i] + qualities[i - 1])
        for i in range(1, len(qualities) - 1)
    ]
    max_d = max(second_diffs) if second_diffs else 0.0
    return max_d < max_jump, max_d


@torch.no_grad()
def run_all_sanity_checks(
    asp_layer: ASPLayer,
    x: torch.Tensor,
    *,
    quality_fn: Callable[[torch.Tensor], float],
    oracle_quality: float,
    saturation_tolerance: float = 0.10,    # qualité ASP @ r=R_max ≥ oracle - tol
    smoothness_max_jump: float = 0.5,
) -> SanityResult:
    result = SanityResult()
    rank_grid = list(range(0, asp_layer.cfg.R_max + 1, max(1, asp_layer.cfg.R_max // 8)))
    if asp_layer.cfg.R_max not in rank_grid:
        rank_grid.append(asp_layer.cfg.R_max)
    rank_grid = sorted(set(rank_grid))

    monotone, qualities = sanity_check_monotone_quality(
        asp_layer, x, quality_fn=quality_fn, rank_grid=rank_grid,
    )
    result.monotone_passed = monotone

    # Saturation : r=R_max ≈ oracle
    result.saturation_quality = qualities[-1]
    result.saturation_oracle = oracle_quality
    result.saturation_passed = qualities[-1] >= oracle_quality - saturation_tolerance

    # Effondrement
    collapse, diff = sanity_check_collapse(asp_layer, x)
    result.collapse_passed = collapse
    result.collapse_diff = diff

    # Lissité
    smooth, max_d = sanity_check_smoothness(qualities, max_jump=smoothness_max_jump)
    result.smoothness_passed = smooth
    result.smoothness_max_jump = max_d
    return result
