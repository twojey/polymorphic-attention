"""
svd_pipeline.py — pipeline SVD batché FP64 sur matrices d'attention audit_svd.

Spec : DOC/02 §2.1, DOC/01 §8.4 (extraction FP64), §8.7 (pas d'agrégation pré-SVD).

Pour chaque matrice A (par couche, par tête, par exemple) :
- SVD complète FP64
- Calcul de r_eff(θ) pour θ ∈ {0.95, 0.99}
  r_eff(θ) = nombre minimum de valeurs singulières expliquant ≥ θ de la variance

Sortie : tensor (L, B, H) ou (B, H) par couche selon mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def r_eff_from_singular_values(s: torch.Tensor, theta: float) -> torch.Tensor:
    """r_eff(θ) à partir des valeurs singulières.

    s : (..., k) FP64. Retourne (..., ) int : plus petit r tel que
    Σ_{i<r} s_i² / Σ s² ≥ θ.
    """
    s2 = s.pow(2)
    cumsum = s2.cumsum(dim=-1)
    total = s2.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    ratio = cumsum / total
    # premier index où ratio ≥ θ
    # on ajoute 1 car r_eff = nombre de valeurs requises (pas l'index)
    above = ratio >= theta
    # argmax retourne premier True
    r_eff = above.float().argmax(dim=-1) + 1
    # corner case : si toutes les valeurs sont 0, ratio = 0 partout, argmax retourne 0
    all_zero = (s2.sum(dim=-1) == 0)
    r_eff = torch.where(all_zero, torch.zeros_like(r_eff), r_eff)
    return r_eff


@dataclass
class SVDResult:
    """Résultat SVD pour une matrice d'attention."""
    s: torch.Tensor       # valeurs singulières FP64 (..., k)
    r_eff_95: torch.Tensor   # entier
    r_eff_99: torch.Tensor


def svd_attention(A: torch.Tensor, *, theta_values: tuple[float, ...] = (0.95, 0.99)) -> dict[str, torch.Tensor]:
    """SVD batché FP64 sur (..., M, N).

    Retourne {"s": ..., "r_eff_<θ>": ...}. r_eff arrondi entier.
    """
    A_fp64 = A.to(torch.float64)
    s = torch.linalg.svdvals(A_fp64)  # (..., min(M,N))
    out: dict[str, torch.Tensor] = {"s": s}
    for theta in theta_values:
        out[f"r_eff_{int(theta*100)}"] = r_eff_from_singular_values(s, theta)
    return out


def hankelize_attention_lines(A: torch.Tensor) -> torch.Tensor:
    """Pour chaque ligne i de A, construit la matrice de Hankel et retourne
    un tenseur (..., N, p, q). Convention V1 — voir phase1_metrologie.metrics.hankel.
    """
    *batch_dims, M, N = A.shape
    p = N // 2
    q = N - p + 1
    # H[i, ., .] avec H[i,a,b] = A[i, a+b] sur [i, N) — convention causale
    # Pour simplicité on construit la Hankel sur la ligne entière A[i, :] sans
    # restriction causale. Les lignes triangulaires inférieures ont moins de
    # contenu mais la Hankel reste définie.
    H = torch.empty(*batch_dims, M, p, q, dtype=A.dtype, device=A.device)
    for a in range(p):
        H[..., :, a, :] = A[..., :, a : a + q]
    return H
