"""
b5_block_diagonal_distance.py — Property B5 : distance à la classe block-diagonale.

Spec : DOC/CATALOGUE B5. Mesure ε_BD(A) pour une taille de bloc donnée.

Pour balayer plusieurs `block_size` candidats, on renvoie la **meilleure**
(plus petit ε) et le block_size correspondant. Choix par défaut des
candidats : 2, 4, 8, 16, N//4, N//2.
"""

from __future__ import annotations

import torch

from catalog.projectors import BlockDiagonal
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class B5BlockDiagonalDistance(Property):
    """B5 — meilleure distance Frobenius à un block-diagonal sur grille de block_size.

    Retourne le block_size optimal + son ε. Permet d'identifier les
    attentions modulaires (group attention, mixture-of-experts patterns).
    """

    name = "B5_block_diagonal_distance"
    family = "B"
    cost_class = 2
    scope = "per_regime"

    def __init__(self, block_sizes: list[int] | None = None) -> None:
        # None → calculé à partir de N au runtime (cf. compute)
        self._user_block_sizes = block_sizes

    def _candidate_block_sizes(self, N: int) -> list[int]:
        if self._user_block_sizes is not None:
            return [b for b in self._user_block_sizes if 1 <= b <= N]
        # Grille adaptative par défaut
        cands = {2, 4, 8, 16, N // 4, N // 2}
        return sorted(b for b in cands if 1 <= b <= N)

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        N = A.size(-1)
        A_work = A.to(device=ctx.device, dtype=ctx.dtype)

        best_eps: float | None = None
        best_bs: int = 0
        all_eps: dict[int, float] = {}
        for bs in self._candidate_block_sizes(N):
            proj = BlockDiagonal(block_size=bs).project(A_work)
            eps = (
                (A_work - proj).flatten(start_dim=-2).norm(dim=-1)
                / A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
            )
            eps_med = float(eps.float().median().item())
            all_eps[bs] = eps_med
            if best_eps is None or eps_med < best_eps:
                best_eps = eps_med
                best_bs = bs

        return {
            "epsilon_best": float(best_eps if best_eps is not None else 1.0),
            "block_size_best": int(best_bs),
            "epsilon_at_bs2": float(all_eps.get(2, -1.0)),
            "epsilon_at_bs8": float(all_eps.get(8, -1.0)),
            "n_matrices": int(A.size(0) * A.size(1)),
        }
