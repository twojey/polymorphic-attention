"""
e5_entropy_rate.py — Property E5 : taux d'entropie conditionnel.

Pour une chaîne de Markov de transition T = A, le taux d'entropie est :
    h(A) = -Σ_i π_i Σ_j T_{ij} log T_{ij}

où π est la distribution stationnaire (vecteur propre gauche de A
correspondant à la valeur propre 1). Pour les Oracles dense, A est
stochastique par ligne (post-softmax), donc T existe.

h proche de 0 = chaîne quasi-déterministe (transitions concentrées).
h proche de log(N) = chaîne mixante uniforme.
"""

from __future__ import annotations

import math

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class E5EntropyRate(Property):
    """E5 — taux d'entropie h = -Σ π_i Σ_j T_ij log T_ij."""

    name = "E5_entropy_rate"
    family = "E"
    cost_class = 3
    requires_fp64 = True
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30, max_power_iter: int = 50) -> None:
        self.eps_floor = eps_floor
        self.max_power_iter = max_power_iter

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).clamp_min(self.eps_floor)
        # H_row[i] = -Σ_j T_ij log T_ij
        H_row = -(A_work * A_work.log()).sum(dim=-1)  # (B, H, N)

        # Stationnaire π par power iteration : π = π · A
        pi = torch.full((B, H, N), 1.0 / N, device=A_work.device, dtype=A_work.dtype)
        for _ in range(self.max_power_iter):
            pi_new = torch.einsum("bhi,bhij->bhj", pi, A_work)
            pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
            if torch.allclose(pi, pi_new, atol=1e-7):
                pi = pi_new
                break
            pi = pi_new
        # h = Σ π_i H_row[i]
        h = (pi * H_row).sum(dim=-1)  # (B, H), nats
        h_norm = h / math.log(max(N, 2))
        h_flat = h.float().flatten()
        hn_flat = h_norm.float().flatten()

        return {
            "entropy_rate_nats_median": float(h_flat.median().item()),
            "entropy_rate_nats_mean": float(h_flat.mean().item()),
            "entropy_rate_normalized_median": float(hn_flat.median().item()),
            "fraction_quasi_deterministic": float(
                (hn_flat < 0.15).float().mean().item()
            ),
            "fraction_mixing": float((hn_flat > 0.85).float().mean().item()),
            "n_matrices": int(B * H),
        }
