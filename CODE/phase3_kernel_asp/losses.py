"""
losses.py — Losses spécifiques à l'ASPLayer.

Spec : DOC/03 §3, §4.

- L_matriochka : output-based, sommée sur tirages r ∈ S
  Σ_r w_r · L_task(ASPLayer(x; mask=[1×r, 0×(R_max-r)]), y_target)

- L_consistency : E_{r, δ}[‖y(x;r) − y(x;r+δ)‖²]
  Pénalise les sauts de qualité quand on incrémente r d'un cran.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch
from torch import nn


def matriochka_rank_schedule(R_max: int, n_samples: int = 4, seed: int | None = None) -> list[int]:
    """Retourne une liste de rangs à échantillonner pour L_matriochka.

    Stratégie V1 : combinaison de R_max (saturation), R_max/2, R_max/4, et
    quelques tirages aléatoires.
    """
    rng = random.Random(seed)
    fixed = [R_max, max(1, R_max // 2), max(1, R_max // 4)]
    n_random = max(0, n_samples - len(fixed))
    sampled = [rng.randint(1, R_max) for _ in range(n_random)]
    return list(set(fixed + sampled))


def matriochka_weights(ranks: list[int], strategy: str = "uniform") -> dict[int, float]:
    """Poids w_r par rang. uniform = même poids ; decreasing = poids plus
    important aux rangs élevés (encourage la qualité avec plus de capacité).
    """
    if strategy == "uniform":
        w = 1.0 / len(ranks)
        return {r: w for r in ranks}
    if strategy == "decreasing":
        # w_r ∝ r
        total = sum(ranks)
        return {r: r / total for r in ranks}
    raise ValueError(f"strategy inconnue : {strategy}")


def loss_matriochka(
    *,
    asp_layer_forward: Callable[[torch.Tensor, int], torch.Tensor],
    x: torch.Tensor,
    y_target: torch.Tensor,
    task_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ranks: list[int],
    weights: dict[int, float],
) -> torch.Tensor:
    """L_matriochka. asp_layer_forward(x, r) doit appliquer ASPLayer avec un
    masque [1×r, 0×(R_max-r)] et retourner les logits/output.
    """
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    for r in ranks:
        out = asp_layer_forward(x, r)
        loss_r = task_loss(out, y_target)
        total = total + weights[r] * loss_r
    return total


def loss_consistency(
    *,
    asp_layer_forward: Callable[[torch.Tensor, int], torch.Tensor],
    x: torch.Tensor,
    R_max: int,
    n_samples: int = 4,
    delta: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """L_consistency = E_{r, δ}[‖y(x;r) − y(x;r+δ)‖²]."""
    rng = random.Random(seed)
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    n_evaluated = 0
    for _ in range(n_samples):
        r = rng.randint(1, max(1, R_max - delta))
        y_r = asp_layer_forward(x, r)
        y_r_delta = asp_layer_forward(x, r + delta)
        total = total + (y_r - y_r_delta).pow(2).mean()
        n_evaluated += 1
    return total / max(n_evaluated, 1)
