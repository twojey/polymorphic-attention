"""
u1_butterfly_distance.py — Property U1 : distance à la classe Butterfly (V1 mask).

Spec : DOC/CATALOGUE §U1 "Butterfly".

V1 : utilise ButterflyMask Projector qui projette sur le support théorique
d'une matrice Butterfly à 2 niveaux. ε_butterfly_min ≤ ε_butterfly_vrai
(borne inférieure, car la vraie classe Butterfly est plus large que le
mask de support).

V2 (Sprint A4+) : factorisation ALS Butterfly avec gradient descent.
"""

from __future__ import annotations

import torch

from catalog.projectors import ButterflyMask
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class U1ButterflyDistance(Property):
    """U1 — borne inf ε_butterfly via mask sparsity 2-niveaux."""

    name = "U1_butterfly_distance"
    family = "U"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self) -> None:
        self._projector = ButterflyMask()

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        eps = self._projector.epsilon(A_work)  # (B, H)
        eps_flat = eps.float().flatten()

        # Density du mask (fraction d'entrées non-nulles)
        mask = self._projector._get_mask(N, A_work.device, A_work.dtype)
        density = float(mask.float().mean().item())

        return {
            "epsilon_butterfly_lb_median": float(eps_flat.median().item()),
            "epsilon_butterfly_lb_mean": float(eps_flat.mean().item()),
            "epsilon_butterfly_lb_min": float(eps_flat.min().item()),
            "epsilon_butterfly_lb_max": float(eps_flat.max().item()),
            "fraction_close_to_butterfly_below_0p30": float(
                (eps_flat < 0.30).float().mean().item()
            ),
            "mask_density": density,
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
