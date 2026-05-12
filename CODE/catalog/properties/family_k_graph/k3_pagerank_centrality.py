"""
k3_pagerank_centrality.py — Property K3 : centralité PageRank-like.

Spec : DOC/CATALOGUE §K3 "eigencentrality".

PageRank classique : π · Aᵀ = π (vecteur propre gauche de A si A est
stochastique en lignes). On résoud par power iteration : π_{t+1} = π_t · Aᵀ
puis renormalise.

Pour attention dense softmax (lignes sum=1, donc A est ROW-stochastique
= matrice de Markov), π est la distribution stationnaire. Convergence
garantie par Perron-Frobenius si A irréductible.

Métriques exposées :
- entropy(π) : si π piqué → quelques tokens dominent l'information flow
- max(π) : "alpha-mass" sur le token central
- Gini(π) : inégalité de distribution
- power_iter_residual : ‖π_t · Aᵀ − π_t‖ après iteration → indique
  convergence
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class K3PageRankCentrality(Property):
    """K3 — distribution stationnaire de A vue comme matrice de transition Markov."""

    name = "K3_pagerank_centrality"
    family = "K"
    cost_class = 3  # power iteration + entropie
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        max_iter: int = 100,
        damping: float = 0.85,
        tol: float = 1e-6,
        eps_floor: float = 1e-30,
    ) -> None:
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping doit être ∈ (0, 1], reçu {damping}")
        self.max_iter = max_iter
        self.damping = damping
        self.tol = tol
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape
        if N != N2:
            raise ValueError(f"A doit être carrée, reçu N={N} != {N2}")

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Renormalise rows pour s'assurer A row-stochastique
        row_sum = A_work.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        P = A_work / row_sum  # (B, H, N, N), row-stochastique

        # Power iteration : π_{t+1}ᵀ = damping · π_tᵀ · P + (1-damping) · uniform
        # On veut le left-eigenvector de P (= eigenvector dominant droit de Pᵀ)
        pi = torch.ones(B, H, N, device=A_work.device, dtype=A_work.dtype) / N
        teleport = torch.ones(B, H, N, device=A_work.device, dtype=A_work.dtype) / N

        final_residual = torch.zeros(B, H, device=A_work.device, dtype=A_work.dtype)
        for _ in range(self.max_iter):
            # π_t · Pᵀ = (π_t @ Pᵀ_per_batch)
            new_pi = self.damping * torch.einsum("bhi,bhij->bhj", pi, P) + (
                1.0 - self.damping
            ) * teleport
            # Normalize (devrait sommer à 1 mais évite la dérive numérique)
            new_pi = new_pi / new_pi.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
            diff = (new_pi - pi).abs().sum(dim=-1)  # L1 (TV-like)
            final_residual = diff
            pi = new_pi
            if (diff < self.tol).all():
                break

        # Métriques sur π
        pi_safe = pi.clamp_min(self.eps_floor)
        log_pi = pi_safe.log()
        entropy = -(pi_safe * log_pi).sum(dim=-1)  # (B, H), nats
        entropy_norm = entropy / math.log(N)
        max_mass = pi.max(dim=-1).values  # (B, H)

        # Gini sur π
        sorted_asc, _ = pi.sort(dim=-1)
        idx = torch.arange(1, N + 1, device=A_work.device, dtype=A_work.dtype)
        weighted = (sorted_asc * idx).sum(dim=-1)
        total = sorted_asc.sum(dim=-1).clamp_min(self.eps_floor)
        gini = (2 * weighted - (N + 1) * total) / (N * total)

        ent_flat = entropy_norm.float().flatten()
        max_flat = max_mass.float().flatten()
        gini_flat = gini.float().flatten()
        resid_flat = final_residual.float().flatten()

        return {
            "pi_entropy_norm_median": float(ent_flat.median().item()),
            "pi_entropy_norm_mean": float(ent_flat.mean().item()),
            "pi_max_mass_median": float(max_flat.median().item()),
            "pi_max_mass_mean": float(max_flat.mean().item()),
            "pi_max_mass_max": float(max_flat.max().item()),
            "pi_gini_median": float(gini_flat.median().item()),
            "pi_gini_mean": float(gini_flat.mean().item()),
            "power_iter_residual_median": float(resid_flat.median().item()),
            "power_iter_residual_max": float(resid_flat.max().item()),
            "damping": self.damping,
            "max_iter": self.max_iter,
            "n_matrices": int(B * H),
            "seq_len": int(N),
        }
