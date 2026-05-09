"""
sparsity_loss.py — L_sparsity (forme pondérée Matriochka recommandée).

Spec : DOC/04 §2, ROADMAP 4.2.

L_sparsity pénalise l'utilisation de rang :
    L_sparsity = Σ_i w_i · m_{t,i}

avec w_i pondération qui pénalise plus les rangs élevés.

Stratégies V1 :
- "uniform" : w_i = 1 (pénalité = nombre de m_i actifs)
- "linear"  : w_i = i (poids croissant avec i)
- "exponential" : w_i = 2^(i-1)
"""

from __future__ import annotations

import torch


def sparsity_weights(R_max: int, strategy: str = "linear") -> torch.Tensor:
    """Poids w_i ∈ R^R_max pour pénaliser m_i."""
    if strategy == "uniform":
        return torch.ones(R_max)
    if strategy == "linear":
        return torch.arange(1, R_max + 1, dtype=torch.float32)
    if strategy == "exponential":
        return torch.pow(2.0, torch.arange(R_max, dtype=torch.float32))
    raise ValueError(f"strategy inconnue : {strategy}")


def loss_sparsity(
    *,
    mask: torch.Tensor,             # (B, N, R_max) ∈ [0, 1]
    weights: torch.Tensor,          # (R_max,)
    reduction: str = "mean",
) -> torch.Tensor:
    """L_sparsity = Σ_i w_i · m_i, agrégé sur (B, N) selon `reduction`."""
    assert mask.size(-1) == weights.numel(), \
        f"R_max mismatch : mask {mask.shape} vs weights {weights.shape}"
    weighted = mask * weights.to(mask.device, mask.dtype)
    summed = weighted.sum(dim=-1)   # (B, N)
    if reduction == "mean":
        return summed.mean()
    if reduction == "sum":
        return summed.sum()
    if reduction == "none":
        return summed
    raise ValueError(f"reduction inconnue : {reduction}")
