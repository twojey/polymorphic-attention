"""
soft_mask.py — Soft-Mask continu pour la sélection de rang.

Spec : DOC/03 §3 (Soft-Mask) et §4 (STE).

m_{t,i} = σ(β · (α · R_max − i + ½))

- α ∈ [0, 1] : "fraction de R_max activée" prédite par le Spectromètre
- β : raideur de la sigmoïde (calibré phase 3)
- m_{t,i} ∈ [0, 1] : poids du i-ème vecteur Matriochka pour le token t

Garantie : m_{t,1} ≥ m_{t,2} ≥ … ≥ m_{t, R_max} par construction (i croissant
dans la sigmoïde décroissante).
"""

from __future__ import annotations

import torch


def soft_mask(*, alpha: torch.Tensor, R_max: int, beta: float = 4.0) -> torch.Tensor:
    """Calcule m_{t,i} ∈ [0,1] pour i ∈ [1, R_max].

    alpha : (..., 1) ou (...,) — "fraction" prédite. R_max et beta scalaires.
    Retourne (..., R_max).
    """
    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)
    if alpha.shape[-1] != 1:
        alpha = alpha.unsqueeze(-1)
    i = torch.arange(1, R_max + 1, device=alpha.device, dtype=alpha.dtype)  # (R_max,)
    # broadcast : alpha (..., 1) × i (R_max,) → (..., R_max)
    z = beta * (alpha * R_max - i + 0.5)
    return torch.sigmoid(z)


def hard_threshold_ste(soft: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator : forward = hard mask (0/1 selon soft > 0.5),
    backward = gradient du soft mask.
    """
    hard = (soft > 0.5).to(soft.dtype)
    return hard + (soft - soft.detach())


def gumbel_softmax_mask(*, logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Variante Gumbel-Softmax (en réserve, DOC/03 §3)."""
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
