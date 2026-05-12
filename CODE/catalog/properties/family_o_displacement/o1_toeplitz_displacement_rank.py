"""
o1_toeplitz_displacement_rank.py — Property O1 : rang de déplacement Toeplitz.

Spec : DOC/00b §O1.

L'opérateur de déplacement Toeplitz est :

    ∇_Z(A) = A − Z·A·Zᵀ

avec Z le shift down (matrice (N,N) avec 1 sur sous-diagonale, 0 ailleurs).

**Invariant fondamental** : pour toute matrice Toeplitz T, rang(∇_Z T) ≤ 2.
Pour A quelconque, le rang de ∇_Z(A) mesure "à quel point A est Toeplitz" :
- rang = 1 ou 2 → structure Toeplitz parfaite (cf. Kailath)
- rang faible → quasi-Toeplitz
- rang ≈ N → orthogonal à Toeplitz

C'est plus fin que B1 ε_Frobenius parce qu'invariant aux normalisations
et conditionné par la *structure* plutôt que par la *position des entrées*.

Référence : Kailath & Sayed, "Fast Reliable Algorithms for Matrices with
Structure" (1999). Voir aussi Pan "Structured Matrices and Polynomials".
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _shift_down(n: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Shift down operator Z[i, j] = 1 si i = j+1, 0 sinon. Shape (n, n)."""
    Z = torch.zeros(n, n, device=device, dtype=dtype)
    if n > 1:
        idx = torch.arange(n - 1, device=device)
        Z[idx + 1, idx] = 1.0
    return Z


@register_property
class O1ToeplitzDisplacementRank(Property):
    """O1 — rang effectif du résidu ∇_Z(A) = A − Z·A·Zᵀ.

    Calcule rang(∇_Z A) au sens numérique (r_eff θ-cumulative). Pour
    Toeplitz parfaite : rang ≤ 2. Pour attention dense softmax : rang
    typiquement modéré (5-20 sur N=24 selon régime).

    Sortie principale : `displacement_rank_eff` médiane sur (B, H).
    """

    name = "O1_toeplitz_displacement_rank"
    family = "O"
    cost_class = 3  # SVD batché sur (B, H, N, N)
    requires_fp64 = True  # SVD-sensible
    scope = "per_regime"

    def __init__(
        self,
        theta_cumulative: float = 0.99,
        rank_threshold_atol: float = 1e-10,
    ) -> None:
        self.theta = theta_cumulative
        self.rank_atol = rank_threshold_atol

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        Z = _shift_down(N, device=str(A_work.device), dtype=A_work.dtype)

        # ∇_Z(A) = A − Z·A·Zᵀ — batché sur (B, H)
        nabla = A_work - Z @ A_work @ Z.T  # (B, H, N, N)

        # Cache la SVD si une autre Property la réutilise (peu probable mais
        # cohérent avec le pattern PropertyContext)
        cache_key = ctx.cache_key(
            "svdvals_nabla_toeplitz", tuple(A_work.shape), str(A_work.dtype)
        )

        def _svdvals() -> torch.Tensor:
            return torch.linalg.svdvals(nabla)  # (B, H, N)

        sigmas = ctx.get_or_compute(cache_key, _svdvals)
        # r_eff(θ) par batch elem
        energies = (sigmas ** 2).flip(dims=(-1,))  # croissant pour cumsum simple
        # On veut Σ_{i<k} σ_i² / Σ σ_i² ≥ θ → premier k où cumsum_desc ≥ θ
        sigmas_desc = sigmas  # supposé déjà décroissant (sortie svdvals)
        cumsum = (sigmas_desc ** 2).cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(1e-30)
        ratio = cumsum / total  # (B, H, N)
        # premier k tel que ratio[k-1] >= theta → r_eff = k
        above = ratio >= self.theta
        # argmax sur bool : retourne le premier True
        r_eff = above.float().argmax(dim=-1) + 1  # (B, H), 1-indexed

        # Rang strict (numerical) au tol absolue
        rank_strict = (sigmas > self.rank_atol).sum(dim=-1)  # (B, H)

        # Stats
        r_eff_flat = r_eff.float().flatten()
        rank_strict_flat = rank_strict.float().flatten()

        # Fraction matrices avec rang Toeplitz ≤ 2 (= "presque Toeplitz")
        is_toeplitz_strict = (rank_strict <= 2).float().mean()

        # Norm Frobenius du résidu (utile pour comparer avec B1)
        nabla_norm = nabla.flatten(start_dim=-2).norm(dim=-1)
        A_norm = A_work.flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-30)
        rel_norm = (nabla_norm / A_norm).float().flatten()

        return {
            "displacement_rank_eff_median": float(r_eff_flat.median().item()),
            "displacement_rank_eff_mean": float(r_eff_flat.mean().item()),
            "displacement_rank_eff_max": float(r_eff_flat.max().item()),
            "displacement_rank_strict_median": float(rank_strict_flat.median().item()),
            "displacement_rank_strict_max": float(rank_strict_flat.max().item()),
            "fraction_rank_le_2_strict": float(is_toeplitz_strict.item()),
            "nabla_relative_norm_median": float(rel_norm.median().item()),
            "nabla_relative_norm_mean": float(rel_norm.mean().item()),
            "theta": self.theta,
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
