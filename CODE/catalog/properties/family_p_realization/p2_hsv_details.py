"""
p2_hsv_details.py — Property P2 : Hankel Singular Values détaillées.

Spec : DOC/CATALOGUE §P2 + §P3 ordre minimal.

Étend P1 en exposant la distribution complète des HSV (Hankel singular
values) et le **balanced order** : nb de HSV non-négligeables (HSV > τ · σ_1).
Donne une mesure plus fine de l'ordre minimal d'un système linéaire.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P2HSVDetails(Property):
    """P2 — distribution HSV et ordre minimal balanced."""

    name = "P2_hsv_details"
    family = "P"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(
        self,
        balance_thresholds: tuple[float, ...] = (0.01, 0.10),
        eps_floor: float = 1e-30,
    ) -> None:
        self.thresholds = balance_thresholds
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
        # Hankel block construit comme P1 : k=N/2 lignes en bloc i, n_col en bloc j
        k = N // 2
        n_col = N - k
        H_block = torch.zeros(B, H, k, n_col * N2, device=A_work.device, dtype=A_work.dtype)
        for i in range(k):
            for j in range(n_col):
                if i + j < N:
                    H_block[..., i, j * N2: (j + 1) * N2] = A_work[..., i + j, :]

        sigmas = torch.linalg.svdvals(H_block)
        sigma_1 = sigmas[..., 0].clamp_min(self.eps_floor)
        # Normalisé
        sigmas_normed = sigmas / sigma_1.unsqueeze(-1)  # (B, H, K)

        results: dict[str, float | int | str | bool] = {}
        for tau in self.thresholds:
            order = (sigmas_normed > tau).sum(dim=-1)  # (B, H)
            order_flat = order.float().flatten()
            tag = f"{tau:.2f}".replace(".", "p")
            results[f"balanced_order_tau_{tag}_median"] = float(order_flat.median().item())
            results[f"balanced_order_tau_{tag}_mean"] = float(order_flat.mean().item())

        # Top-k HSV ratios
        K = sigmas.shape[-1]
        for j in (2, 3, 5):
            if j < K:
                ratio = (sigmas[..., j - 1] / sigma_1).float().flatten()
                results[f"hsv_sigma{j}_over_sigma1_median"] = float(ratio.median().item())

        results["n_hsv"] = K
        results["n_matrices"] = int(B * H)
        return results
