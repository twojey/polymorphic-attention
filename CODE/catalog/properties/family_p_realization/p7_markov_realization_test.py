"""
p7_markov_realization_test.py — Property P7 : test "realizable Markov chain".

Étend la famille P : si A est issue d'un opérateur Markov LTI cachée
de dimension n, alors la matrice de Hankel des produits successifs
{A, A², A³, A⁴} a rang ≈ n.

V1 : on calcule r_eff sur le concat horizontal [A | A² | A³ | A⁴].
Une réalisation finie ⇔ rang faible de cette concat.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class P7MarkovRealizationTest(Property):
    """P7 — r_eff([A | A² | A³ | A⁴]) = proxy ordre Markov caché."""

    name = "P7_markov_realization_test"
    family = "P"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        theta_cumulative: float = 0.99,
        powers: tuple[int, ...] = (1, 2, 3, 4),
        eps_floor: float = 1e-30,
    ) -> None:
        self.theta = theta_cumulative
        self.powers = tuple(sorted(powers))
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            return {"skip_reason": "non-square", "n_matrices": int(B * H)}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Compute powers
        Ak = A_work.clone()
        blocks = [Ak]
        for _ in range(1, max(self.powers)):
            Ak = Ak @ A_work
            blocks.append(Ak)
        # Concat along last axis
        H_concat = torch.cat([blocks[p - 1] for p in self.powers], dim=-1)  # (B, H, N, k·N)

        s = torch.linalg.svdvals(H_concat)
        s2 = s.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        ratio = cumsum / total
        above = ratio >= self.theta
        r_eff = (above.float().argmax(dim=-1) + 1).float()  # (B, H)
        r_flat = r_eff.flatten()

        K = s.shape[-1]
        return {
            "markov_realization_rank_median": float(r_flat.median().item()),
            "markov_realization_rank_max": float(r_flat.max().item()),
            "markov_realization_rank_normalized": float(
                (r_flat / max(K, 1)).median().item()
            ),
            "fraction_low_order_below_5": float(
                (r_flat < 5.0).float().mean().item()
            ),
            "powers": str(self.powers),
            "n_matrices": int(B * H),
        }
