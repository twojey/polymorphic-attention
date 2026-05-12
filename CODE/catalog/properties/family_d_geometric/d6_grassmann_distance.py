"""
d6_grassmann_distance.py — Property D6 : distance Grassmannienne cross-layer.

Pour chaque tête (h), compare le top-r row-subspace de deux couches voisines
(ℓ, ℓ+1) via la distance de Grassmann d_G = √Σ sin²(θ_i) où θ_i sont les
angles principaux.

Mesure la "dérive" du sous-espace dominant à travers les couches.
- d_G ≈ 0 : têtes alignées ⇒ couches redondantes
- d_G ≈ √r : sous-espaces orthogonaux ⇒ couches indépendantes

Scope per_regime_layers (compare ℓ vs ℓ+1).
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class D6GrassmannDistance(Property):
    """D6 — distance Grassmann cross-layer (top-r row-subspace dérive)."""

    name = "D6_grassmann_distance"
    family = "D"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime_layers"

    def __init__(self, top_r: int = 4, eps_floor: float = 1e-12) -> None:
        if top_r < 1:
            raise ValueError(f"top_r doit être ≥ 1, reçu {top_r}")
        self.top_r = top_r
        self.eps_floor = eps_floor

    def compute(
        self, A: list[torch.Tensor], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(A, (list, tuple)) or len(A) < 2:
            return {
                "skip_reason": "need ≥ 2 layers for cross-layer Grassmann",
                "n_layers": int(len(A) if isinstance(A, (list, tuple)) else 0),
            }
        L = len(A)
        if A[0].ndim != 4:
            raise ValueError(f"A[0] doit être (B, H, N, N), reçu {A[0].shape}")

        ds = []
        for l_idx in range(L - 1):
            X1 = A[l_idx].to(device=ctx.device, dtype=ctx.dtype)
            X2 = A[l_idx + 1].to(device=ctx.device, dtype=ctx.dtype)
            if X1.shape != X2.shape:
                continue
            _, _, N, _ = X1.shape
            r = min(self.top_r, N)
            _, _, V1h = torch.linalg.svd(X1, full_matrices=False)
            _, _, V2h = torch.linalg.svd(X2, full_matrices=False)
            B1 = V1h[..., :r, :]  # (B, H, r, N)
            B2 = V2h[..., :r, :]
            cross = torch.einsum("bhri,bhsi->bhrs", B1, B2)
            sigmas = torch.linalg.svdvals(cross).clamp(
                -1.0 + self.eps_floor, 1.0 - self.eps_floor
            )  # cos θ
            sin2 = (1.0 - sigmas.pow(2)).clamp_min(0.0)
            d_g = sin2.sum(dim=-1).sqrt()  # (B, H)
            ds.append(d_g.float().flatten())

        if not ds:
            return {"skip_reason": "shape mismatch all pairs",
                    "n_layers": int(L)}
        all_d = torch.cat(ds)
        return {
            "grassmann_distance_median": float(all_d.median().item()),
            "grassmann_distance_mean": float(all_d.mean().item()),
            "grassmann_distance_max": float(all_d.max().item()),
            "fraction_aligned": float((all_d < 0.3).float().mean().item()),
            "fraction_orthogonal": float(
                (all_d > math.sqrt(self.top_r) - 0.3).float().mean().item()
            ),
            "top_r": int(self.top_r),
            "n_layers": int(L),
            "n_pairs": int(all_d.numel()),
        }
