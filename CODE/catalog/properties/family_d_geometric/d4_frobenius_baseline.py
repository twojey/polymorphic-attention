"""
d4_frobenius_baseline.py — Property D4 : distance Frobenius à matrice aléatoire.

Mesure ‖A − R‖_F / ‖A‖_F où R est une matrice aléatoire normalisée par ligne
(softmax bruit gaussien). Diagnostic "à quel point A diffère d'un random
softmax". Combine avec D2 pour situer A entre {uniform, identity, random}.

Seed déterministe via ctx.metadata['seed'] (défaut 0).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class D4FrobeniusBaseline(Property):
    """D4 — distance Frobenius relative à random softmax baseline."""

    name = "D4_frobenius_baseline"
    family = "D"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, seed_offset: int = 0) -> None:
        self.seed_offset = seed_offset

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        seed = int(ctx.metadata.get("seed", 0)) + self.seed_offset
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(B, H, N, N2, generator=g, dtype=A_work.dtype)
        noise = noise.to(device=A_work.device)
        # Soft-max row-wise (causal mask if A est sup-triangulaire)
        R = torch.softmax(noise, dim=-1)

        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        diff = (A_work - R).flatten(start_dim=-2).norm(dim=-1)
        d_rand = diff / A_norm
        d_flat = d_rand.float().flatten()

        return {
            "distance_random_softmax_median": float(d_flat.median().item()),
            "distance_random_softmax_mean": float(d_flat.mean().item()),
            "fraction_close_to_random": float(
                (d_flat < 0.30).float().mean().item()
            ),
            "n_matrices": int(B * H),
        }
