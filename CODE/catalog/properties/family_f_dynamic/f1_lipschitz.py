"""
f1_lipschitz.py — Property F1 : Lipschitzness vs entrée.

Spec : DOC/CATALOGUE §F1 "Lipschitz constant L : ‖A(x) − A(x')‖ ≤ L · ‖x − x'‖".

V1 sans 2-dumps : on utilise les paires INTRA-batch.
Pour chaque paire (i, j) dans le batch :
  - distance d'entrée : Hamming(tokens_i, tokens_j) si tokens dispo, sinon
    distance L2 entre row-sums (proxy d'inputs effectifs)
  - distance d'attention : ‖A_i − A_j‖_F (moyennée sur les heads)
  - ratio r_{ij} = ‖A_i − A_j‖_F / d(tokens_i, tokens_j)
Sortie : max r_{ij} (Lipschitz proxy), mean, fraction de paires "stables"
(r < seuil).

Limite assumée : c'est une borne empirique, pas une vraie constante Lipschitz.
V2 (Sprint phase 1.5+) : appariera 2 dumps perturbés.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class F1Lipschitz(Property):
    """F1 — Lipschitzness empirique vs distance d'entrée intra-batch."""

    name = "F1_lipschitz"
    family = "F"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(
        self,
        max_pairs: int = 64,
        eps_floor: float = 1e-6,
        stable_threshold: float = 1.0,
    ) -> None:
        self.max_pairs = max_pairs
        self.eps_floor = eps_floor
        self.stable_threshold = stable_threshold

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape
        if B < 2:
            return {"n_matrices": int(B * H), "skip_reason": "batch<2"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        # Distance attention : moyenne Frobenius sur heads
        A_mean = A_work.mean(dim=1)  # (B, N, N)

        # Distance d'entrée : tokens si dispo (Hamming normalisée), sinon
        # row-sums distance L2 (proxy faible)
        tokens = ctx.metadata.get("tokens", None) if ctx.metadata else None
        if tokens is not None and tokens.shape[0] == B:
            tok = tokens.to(device=ctx.device)
            # Hamming pairwise : count(t_i != t_j) / N
            t_i = tok.unsqueeze(1)  # (B, 1, N)
            t_j = tok.unsqueeze(0)  # (1, B, N)
            d_in = (t_i != t_j).float().mean(dim=-1)  # (B, B)
            input_kind = "hamming"
        else:
            row_sums = A_work.sum(dim=-1).mean(dim=1)  # (B, N)
            r_i = row_sums.unsqueeze(1)  # (B, 1, N)
            r_j = row_sums.unsqueeze(0)  # (1, B, N)
            d_in = (r_i - r_j).norm(dim=-1)  # (B, B)
            input_kind = "row_sums_l2"

        # Distance attention pairwise (B, B) Frobenius
        diff = A_mean.unsqueeze(1) - A_mean.unsqueeze(0)  # (B, B, N, N)
        d_attn = diff.flatten(start_dim=2).norm(dim=-1)  # (B, B)

        # Upper triangle (paires distinctes)
        iu = torch.triu_indices(B, B, offset=1)
        d_in_pairs = d_in[iu[0], iu[1]].float()
        d_attn_pairs = d_attn[iu[0], iu[1]].float()

        # Limiter au top max_pairs paires (par d_attn pour repérer le pire cas)
        n_pairs = d_in_pairs.numel()
        if n_pairs > self.max_pairs:
            top = torch.topk(d_attn_pairs, k=self.max_pairs).indices
            d_in_pairs = d_in_pairs[top]
            d_attn_pairs = d_attn_pairs[top]

        # Filtrer paires avec input distance trop petite (mal conditionné)
        mask = d_in_pairs > self.eps_floor
        if mask.sum().item() == 0:
            return {
                "n_matrices": int(B * H), "skip_reason": "all input distances zero",
                "input_kind": input_kind,
            }
        ratios = d_attn_pairs[mask] / d_in_pairs[mask].clamp_min(self.eps_floor)

        return {
            "lipschitz_max": float(ratios.max().item()),
            "lipschitz_p90": float(ratios.quantile(0.90).item()),
            "lipschitz_median": float(ratios.median().item()),
            "lipschitz_mean": float(ratios.mean().item()),
            "fraction_stable_below_threshold": float(
                (ratios < self.stable_threshold).float().mean().item()
            ),
            "n_pairs_valid": int(mask.sum().item()),
            "n_matrices": int(B * H),
            "input_kind": input_kind,
        }
