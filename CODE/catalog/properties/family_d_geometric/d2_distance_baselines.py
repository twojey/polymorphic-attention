"""
d2_distance_baselines.py — Property D2 : distances Frobenius à uniform / identité.

Spec : DOC/CATALOGUE §D2.

Mesure ‖A − M‖_F / ‖A‖_F pour M ∈ {U (uniform row), I_n (identity)}.
Diagnostic rapide : où se situe A entre "concentré" (proche I) et
"dispersé" (proche uniform) ?
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class D2DistanceBaselines(Property):
    """D2 — distance relative à matrices uniform et identity."""

    name = "D2_distance_baselines"
    family = "D"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Uniform : 1/N partout
        U = torch.full_like(A_work, 1.0 / N2)
        eye = torch.eye(N, N2, device=A_work.device, dtype=A_work.dtype)
        I_full = eye.expand_as(A_work)

        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        d_uniform = (A_work - U).flatten(start_dim=-2).norm(dim=-1) / A_norm
        d_identity = (A_work - I_full).flatten(start_dim=-2).norm(dim=-1) / A_norm

        du_flat = d_uniform.float().flatten()
        di_flat = d_identity.float().flatten()

        return {
            "distance_uniform_median": float(du_flat.median().item()),
            "distance_uniform_mean": float(du_flat.mean().item()),
            "distance_identity_median": float(di_flat.median().item()),
            "distance_identity_mean": float(di_flat.mean().item()),
            "closer_to_uniform_fraction": float(
                (du_flat < di_flat).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
