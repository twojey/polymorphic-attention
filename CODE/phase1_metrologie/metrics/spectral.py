"""
spectral.py — entropie spectrale d'une matrice d'attention.

Définition (DOC/01 §3.2) :

    p_k = σ_k² / Σ_j σ_j²
    H_spectrale(A) = − Σ_k p_k log p_k

Bornes :
- H ≈ 0           : A essentiellement de rang 1
- H ≈ log N       : A uniformément distribuée sur ses modes
"""

from __future__ import annotations

import math

import torch


def spectral_entropy(A: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Entropie spectrale de A.

    A : (..., M, N), float. Retourne un tensor (...,) de la même device/dtype
    (cast en FP64 en interne pour la stabilité numérique).
    """
    assert A.ndim >= 2
    A_fp64 = A.to(torch.float64)
    s = torch.linalg.svdvals(A_fp64)         # (..., min(M, N))
    s2 = s.pow(2)
    z = s2.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = s2 / z
    p_safe = p.clamp_min(eps)
    H = -(p_safe * p_safe.log()).sum(dim=-1)
    return H.to(A.dtype)


def normalized_spectral_entropy(A: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """H / log(N) ∈ [0, 1] approximativement (utile pour comparaison cross-N)."""
    H = spectral_entropy(A, eps=eps)
    N = A.size(-1)
    return H / math.log(N)
