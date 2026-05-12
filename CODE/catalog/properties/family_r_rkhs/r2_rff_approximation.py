"""
r2_rff_approximation.py — Property R2 : approximation Random Fourier Features.

Spec : DOC/CATALOGUE §R2 + Bochner.

Pour un noyau translation-invariant K(x, y) = k(x − y), Bochner garantit
qu'il existe une décomposition K(x, y) ≈ Φ(x)ᵀ Φ(y) avec Φ = features
sinusoïdales tirées d'une distribution liée au spectre de K.

V1 simple : on tire D features RFF aléatoires fixes (Gaussian kernel
approximation), on construit K_approx = Φᵀ Φ, et on mesure
ε_RFF = ‖A_sym − K_approx‖_F / ‖A_sym‖_F après ajustement par
projection orthogonale Frobenius.

Donne une borne supérieure sur la distance à un noyau RFF de dimension D.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class R2RFFApproximation(Property):
    """R2 — approximation noyau via Random Fourier Features."""

    name = "R2_rff_approximation"
    family = "R"
    cost_class = 3
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        n_features: int = 32,
        sigma: float = 1.0,
        seed: int = 0,
        eps_floor: float = 1e-30,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features ≥ 1")
        self.D = n_features
        self.sigma = sigma
        self.seed = seed
        self.eps_floor = eps_floor

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, N2 = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Symétriser pour comparer à un kernel
        A_sym = 0.5 * (A_work + A_work.transpose(-1, -2))

        # RFF : positions x_i = i, frequencies w ~ N(0, 1/sigma²)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        w = torch.randn(self.D, 1, generator=gen).to(A_work.device, A_work.dtype) / self.sigma
        b = (torch.rand(self.D, generator=gen).to(A_work.device, A_work.dtype) * 2 * math.pi)

        positions = torch.arange(N, device=A_work.device, dtype=A_work.dtype).view(1, -1)
        # Phi(x_i) = sqrt(2/D) * cos(w·x_i + b)
        Phi = math.sqrt(2.0 / self.D) * torch.cos(w * positions + b.view(-1, 1))  # (D, N)
        K_approx = Phi.T @ Phi  # (N, N)

        # Distance Frobenius après projection orthogonale (rank-1 sur span(K_approx))
        K_norm_sq = (K_approx ** 2).sum().clamp_min(self.eps_floor)
        inner = (A_sym * K_approx).flatten(start_dim=-2).sum(dim=-1)
        scale = inner / K_norm_sq
        proj = scale.unsqueeze(-1).unsqueeze(-1) * K_approx
        residual = A_sym - proj
        eps = (
            residual.flatten(start_dim=-2).norm(dim=-1)
            / A_sym.flatten(start_dim=-2).norm(dim=-1).clamp_min(self.eps_floor)
        )
        eps_flat = eps.float().flatten()

        return {
            "rff_epsilon_median": float(eps_flat.median().item()),
            "rff_epsilon_mean": float(eps_flat.mean().item()),
            "rff_n_features": self.D,
            "rff_sigma": self.sigma,
            "n_matrices": int(B * H),
        }
