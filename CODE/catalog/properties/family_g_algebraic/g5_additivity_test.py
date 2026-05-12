"""
g5_additivity_test.py — Property G5 : test d'additivité (linéarité approximative).

Spec : DOC/CATALOGUE §G5 "‖A(x+y) − A(x) − A(y)‖ / max(‖A(x)‖,‖A(y)‖)".

Pour les attentions de transformer, A n'est pas linéaire en l'entrée (la
softmax casse la linéarité). Mais on peut quantifier "à quel point" en
appliquant deux fois A à différentes entrées et en vérifiant l'écart.

V1 simple : on traite A directement comme un opérateur agissant sur des
vecteurs aléatoires x, y, et on mesure ε_add = ‖A(x+y) − A·x − A·y‖_F /
max(‖A·x‖, ‖A·y‖). Pour une matrice (opérateur linéaire pur), ε_add = 0.
Pour un opérateur non-linéaire wrappant A, ε > 0.

Note : ici A est juste la matrice (linéaire), donc le test est dégénéré
(ε = 0 toujours pour matrice). On garde la Property pour cohérence avec
catalogue + extensibilité V2 (passer un Callable[A, x] au lieu de A).
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class G5AdditivityTest(Property):
    """G5 — test additivité (linéarité opérateur)."""

    name = "G5_additivity_test"
    family = "G"
    cost_class = 1
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, n_pairs: int = 4, seed: int = 0, eps_floor: float = 1e-30) -> None:
        if n_pairs < 1:
            raise ValueError("n_pairs ≥ 1")
        self.n_pairs = n_pairs
        self.seed = seed
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        eps_all: list[torch.Tensor] = []
        for _ in range(self.n_pairs):
            x = torch.randn(N2, generator=gen).to(device=A_work.device, dtype=A_work.dtype)
            y = torch.randn(N2, generator=gen).to(device=A_work.device, dtype=A_work.dtype)
            Ax = A_work @ x.view(-1, 1)
            Ay = A_work @ y.view(-1, 1)
            A_xy = A_work @ (x + y).view(-1, 1)
            diff = A_xy - Ax - Ay
            diff_norm = diff.flatten(start_dim=-2).norm(dim=-1)
            ref = torch.maximum(
                Ax.flatten(start_dim=-2).norm(dim=-1),
                Ay.flatten(start_dim=-2).norm(dim=-1),
            ).clamp_min(self.eps_floor)
            eps_all.append((diff_norm / ref).float().flatten())

        all_eps = torch.cat(eps_all)
        return {
            "additivity_eps_median": float(all_eps.median().item()),
            "additivity_eps_mean": float(all_eps.mean().item()),
            "additivity_eps_max": float(all_eps.max().item()),
            "fraction_strictly_linear_below_1e-10": float(
                (all_eps < 1e-10).float().mean().item()
            ),
            "n_pairs": self.n_pairs,
            "n_matrices": int(B * H),
        }
