"""
t1_permutation_invariance.py — Property T1 : invariance par permutation.

Spec : DOC/CATALOGUE §T1.

Pour mesurer si A est équivariante à permutation des tokens, on tire
plusieurs permutations aléatoires π et calcule :

    ε_π = ‖A − π A π^T‖_F / ‖A‖_F

Si A est strictement permutation-équivariante, ε_π = 0 pour tout π.
Pour attention dense avec position embeddings, ε_π > 0 (la causalité tue
l'équivariance).

Output : ε_π moyenne sur les permutations, fraction "approximativement
équivariantes".
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class T1PermutationInvariance(Property):
    """T1 — distance à l'équivariance par permutation (test stochastique)."""

    name = "T1_permutation_invariance"
    family = "T"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, n_permutations: int = 8, seed: int = 0, eps_floor: float = 1e-30) -> None:
        if n_permutations < 1:
            raise ValueError(f"n_permutations doit être ≥ 1")
        self.n_permutations = n_permutations
        self.seed = seed
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        all_eps: list[torch.Tensor] = []
        for k in range(self.n_permutations):
            perm = torch.randperm(N, generator=gen).to(device=A_work.device)
            # π A π^T : permute lignes puis colonnes
            A_perm = A_work[..., perm, :][..., :, perm]
            diff = (A_work - A_perm).flatten(start_dim=-2).norm(dim=-1)
            eps = diff / A_norm
            all_eps.append(eps.float().flatten())

        all_eps_concat = torch.cat(all_eps)
        # Moyenne par batch elem sur les n permutations
        per_mat_means = torch.stack(all_eps).mean(dim=0)  # (B*H,)

        return {
            "epsilon_perm_median": float(all_eps_concat.median().item()),
            "epsilon_perm_mean": float(all_eps_concat.mean().item()),
            "epsilon_perm_min": float(all_eps_concat.min().item()),
            "epsilon_perm_max": float(all_eps_concat.max().item()),
            "epsilon_perm_per_mat_median": float(per_mat_means.median().item()),
            "fraction_approx_equivariant_below_0p10": float(
                (per_mat_means < 0.10).float().mean().item()
            ),
            "n_permutations": self.n_permutations,
            "n_matrices": int(B * H),
        }
