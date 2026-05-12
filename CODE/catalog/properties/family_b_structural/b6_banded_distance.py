"""
b6_banded_distance.py — Property B6 : distance à la classe à bande (banded).

Spec : DOC/CATALOGUE B6. Mesure ε_B(A) sur grille de bandwidth + bandwidth
optimal. Une attention locale focalisée (SWA-like) aura ε petit pour un
small bandwidth.

Important pour identifier des patterns "attention locale" qu'on ne capture
pas avec Toeplitz/Hankel (qui sont globales).
"""

from __future__ import annotations

import torch

from catalog.projectors import Banded
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B6BandedDistance(Property):
    """B6 — meilleure distance Frobenius à un banded sur grille de bandwidth."""

    name = "B6_banded_distance"
    family = "B"
    cost_class = 2
    scope = "per_regime"

    def __init__(self, bandwidths: list[int] | None = None) -> None:
        self._user_bandwidths = bandwidths

    def _candidate_bandwidths(self, N: int) -> list[int]:
        if self._user_bandwidths is not None:
            return [w for w in self._user_bandwidths if 0 <= w < N]
        cands = {1, 2, 4, 8, 16, N // 8, N // 4}
        return sorted(w for w in cands if 0 <= w < N)

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        N = A.size(-1)
        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        best_eps: float | None = None
        best_w: int = 0
        all_eps: dict[int, float] = {}
        for w in self._candidate_bandwidths(N):
            proj = Banded(bandwidth=w).project(A_work)
            eps = (
                (A_work - proj).flatten(start_dim=-2).norm(dim=-1)
                / A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
            )
            eps_med = float(eps.float().median().item())
            all_eps[w] = eps_med
            if best_eps is None or eps_med < best_eps:
                best_eps = eps_med
                best_w = w

        return {
            "epsilon_best": float(best_eps if best_eps is not None else 1.0),
            "bandwidth_best": int(best_w),
            "epsilon_at_w1": float(all_eps.get(1, -1.0)),
            "epsilon_at_w8": float(all_eps.get(8, -1.0)),
            "n_matrices": int(A.size(0) * A.size(1)),
        }
