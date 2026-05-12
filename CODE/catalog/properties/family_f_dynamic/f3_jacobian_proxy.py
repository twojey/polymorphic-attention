"""
f3_jacobian_proxy.py — Property F3 : proxy de la norme du Jacobien.

Pour A : Δ_softmax, le Jacobien ∂A_t / ∂x_t a pour spectral norm bornée par
‖diag(A_t) − A_t Aᵀ_t‖. On approxime par la norme spectrale de la matrice
de variance ligne par ligne :
    J_t = diag(p_t) − p_t p_tᵀ  où p_t = A[t, :]

Norme spectrale ‖J_t‖_2 = max_i p_{t,i}(1 − p_{t,i}) (top eigenvalue de J_t).

Mesure la sensibilité locale du softmax. Faible = saturation (peu sensible).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class F3JacobianProxy(Property):
    """F3 — proxy Jacobien softmax : max p(1 − p)."""

    name = "F3_jacobian_proxy"
    family = "F"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30) -> None:
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).clamp_min(self.eps_floor)
        # J_t spectral upper bound : max p(1-p)
        p1mp = A_work * (1.0 - A_work)  # (B, H, N, N)
        # Per row, max over j
        jac_per_row = p1mp.max(dim=-1).values  # (B, H, N)
        # Per matrix, max over rows = ‖J‖_∞ proxy
        jac_per_mat = jac_per_row.max(dim=-1).values  # (B, H)

        # Frobenius proxy : Σ_t Σ_i p_{ti}(1-p_{ti})
        jac_frob = p1mp.sum(dim=(-2, -1))  # (B, H)

        j_inf = jac_per_mat.float().flatten()
        j_frob = jac_frob.float().flatten()

        return {
            "jacobian_max_per_row_median": float(j_inf.median().item()),
            "jacobian_max_per_row_mean": float(j_inf.mean().item()),
            "jacobian_frob_proxy_median": float(j_frob.median().item()),
            "fraction_saturated_below_0p10": float(
                (j_inf < 0.10).float().mean().item()
            ),
            "fraction_sensitive_above_0p20": float(
                (j_inf > 0.20).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
