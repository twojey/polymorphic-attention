"""
p1_hankel_realization.py — Property P1 : rang Hankel block (Ho-Kalman V1).

Spec : DOC/CATALOGUE §P1 + Ho-Kalman.

Pour une "séquence" (les lignes de A vue comme évolution temporelle de la
distribution d'attention), on construit la matrice de Hankel "block":

    H[i, j] = A[i + j, :]   pour i + j < N

Le rang de H est l'ordre minimal du système linéaire qui génère la
séquence. C'est l'invariant Ho-Kalman canonique.

V1 : on construit H avec moitié des lignes en première moitié, le reste en
2ème, et on calcule r_eff + spectre (HSV = Hankel Singular Values).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P1HankelRealization(Property):
    """P1 — rang Hankel block (Ho-Kalman) + spectre HSV."""

    name = "P1_hankel_realization"
    family = "P"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N < 4:
            return {"n_matrices": int(B * H), "skip_reason": "N too small"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Construire H[i, j] = A[i + j, :] en respectant i + j < N.
        # Découpage : k = N // 2 lignes en bloc i, n_col = N - k lignes en bloc j.
        k = N // 2
        n_col = N - k
        # H_block ∈ (B, H, k, n_col · N) — pour chaque i ∈ [0, k), j ∈ [0, n_col),
        # on stocke la ligne A[i + j, :].
        H_block = torch.zeros(B, H, k, n_col * N2, device=A_work.device, dtype=A_work.dtype)
        for i in range(k):
            for j in range(n_col):
                if i + j < N:
                    H_block[..., i, j * N2: (j + 1) * N2] = A_work[..., i + j, :]

        # SVD de H_block
        sigmas = torch.linalg.svdvals(H_block)  # (B, H, min(k, n_col*N))
        # r_eff = nb σ pour θ
        s2 = sigmas.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        ratio = cumsum / total
        r_eff = (ratio >= self.theta).float().argmax(dim=-1) + 1  # (B, H)

        # HSV : top-3 singular values normalisés par σ_1
        sigma_1 = sigmas[..., 0].clamp_min(self.eps_floor)
        hsv_top2_ratio = (sigmas[..., 1] / sigma_1) if sigmas.shape[-1] > 1 else torch.zeros_like(sigma_1)
        hsv_top3_ratio = (sigmas[..., 2] / sigma_1) if sigmas.shape[-1] > 2 else torch.zeros_like(sigma_1)

        r_eff_flat = r_eff.float().flatten()
        h2_flat = hsv_top2_ratio.float().flatten()
        h3_flat = hsv_top3_ratio.float().flatten()

        return {
            "hankel_rank_median": float(r_eff_flat.median().item()),
            "hankel_rank_mean": float(r_eff_flat.mean().item()),
            "hankel_rank_max": float(r_eff_flat.max().item()),
            "hsv_sigma2_over_sigma1_median": float(h2_flat.median().item()),
            "hsv_sigma3_over_sigma1_median": float(h3_flat.median().item()),
            "fraction_low_order_le_3": float((r_eff_flat <= 3).float().mean().item()),
            "k_split": k,
            "n_matrices": int(B * H),
            "theta": self.theta,
        }
