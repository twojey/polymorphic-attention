"""
p4_minimal_order.py — Property P4 : ordre minimal Akaike / BIC robuste.

Spec : DOC/CATALOGUE §P4 "ordre minimal du système LTI par critère
information AIC/BIC sur l'erreur de reconstruction Hankel".

Pour chaque k ∈ [1, K_max], on tronque la SVD du Hankel à r=k, on
calcule l'erreur de reconstruction ε(k) = ‖H − H_k‖_F / ‖H‖_F.

Critères :
- AIC(k) = N · log(ε²) + 2k
- BIC(k) = N · log(ε²) + k · log(N)
- Ordre minimal sélectionné = argmin{AIC ou BIC}

Plus robuste que P1 (qui dépend du θ choisi).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P4MinimalOrder(Property):
    """P4 — ordre minimal sélectionné par AIC/BIC sur reconstruction Hankel."""

    name = "P4_minimal_order"
    family = "P"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, k_max: int = 12, eps_floor: float = 1e-30) -> None:
        self.k_max = k_max
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
        k = N // 2
        n_col = N - k
        H_block = torch.zeros(B, H, k, n_col * N2, device=A_work.device, dtype=A_work.dtype)
        for i in range(k):
            for j in range(n_col):
                if i + j < N:
                    H_block[..., i, j * N2: (j + 1) * N2] = A_work[..., i + j, :]

        sigmas = torch.linalg.svdvals(H_block)  # (B, H, K)
        K = min(self.k_max, sigmas.shape[-1])
        s_top = sigmas[..., :K]  # (B, H, K)
        s2 = s_top.pow(2)
        total_energy = sigmas.pow(2).sum(dim=-1).clamp_min(self.eps_floor)  # (B, H)

        # Erreur de troncature à rang r : eps²(r) = Σ_{i>r} σ_i² / Σ σ_i²
        cum_energy = s2.cumsum(dim=-1)
        all_energy = sigmas.pow(2).sum(dim=-1, keepdim=True)
        eps2 = ((all_energy - cum_energy) / total_energy.unsqueeze(-1)).clamp_min(self.eps_floor)
        log_eps2 = eps2.log()  # (B, H, K)

        n_obs = float(N * N2)
        ks_range = torch.arange(1, K + 1, dtype=log_eps2.dtype, device=log_eps2.device)
        # AIC(k) = n_obs · log(eps²) + 2k
        # BIC(k) = n_obs · log(eps²) + k · log(n_obs)
        aic = n_obs * log_eps2 + 2.0 * ks_range
        bic = n_obs * log_eps2 + ks_range * math.log(n_obs)

        k_aic = aic.argmin(dim=-1) + 1  # 1-indexed
        k_bic = bic.argmin(dim=-1) + 1

        return {
            "minimal_order_aic_median": float(k_aic.float().median().item()),
            "minimal_order_bic_median": float(k_bic.float().median().item()),
            "minimal_order_aic_mean": float(k_aic.float().mean().item()),
            "minimal_order_bic_mean": float(k_bic.float().mean().item()),
            "fraction_order_le_3_bic": float((k_bic <= 3).float().mean().item()),
            "fraction_order_eq_kmax_bic": float((k_bic == K).float().mean().item()),
            "K_max_searched": int(K),
            "n_matrices": int(B * H),
        }
