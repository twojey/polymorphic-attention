"""
m1_token_type_sensitivity.py — Property M1 : sensitivity par type de token.

Spec : DOC/CATALOGUE §M1 "S_{type} : variation moyenne ‖A‖ vs type de token query".

Pour chaque type de token t dans le vocab effectif du batch :
  - poids attention moyen quand query = t : μ_query(t) = mean A[i, h, q, :] où tokens[i, q] == t
  - poids attention moyen quand key = t : μ_key(t) = mean A[i, h, :, k] où tokens[i, k] == t
Sensitivity = variance ces moyennes vs moyenne globale.

Nécessite ctx.metadata["tokens"] de shape (B, N) ints.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class M1TokenTypeSensitivity(Property):
    """M1 — variation attention vs type de token (query/key)."""

    name = "M1_token_type_sensitivity"
    family = "M"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, top_k_types: int = 16, min_count: int = 3) -> None:
        self.top_k_types = top_k_types
        self.min_count = min_count

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        tokens = ctx.metadata.get("tokens", None) if ctx.metadata else None
        if tokens is None or tokens.shape[0] != B:
            return {"n_matrices": int(B * H), "skip_reason": "tokens missing"}

        A_work = A.to(device=ctx.device, dtype=ctx.dtype)
        tok = tokens.to(device=ctx.device, dtype=torch.long)  # (B, N)

        # Row-mean attention par (B, H, q) : moyenne sur la dim key
        # Pour query type analysis : on regarde A[i, h, q, :] où tokens[i, q] == t
        # On utilise un proxy plus simple : moyenne attention vers/depuis token t

        row_w = A_work.mean(dim=-1)  # (B, H, N), poids moyen sortant de la position q
        col_w = A_work.mean(dim=-2)  # (B, H, N), poids moyen vers la position k

        # Top-k tokens types observés (par fréquence)
        unique, counts = torch.unique(tok.flatten(), return_counts=True)
        topk = torch.topk(counts, k=min(self.top_k_types, counts.numel()))
        top_types = unique[topk.indices]  # (K,)

        per_type_query: list[float] = []
        per_type_key: list[float] = []
        for t in top_types.tolist():
            mask = tok == t  # (B, N)
            count = int(mask.sum().item())
            if count < self.min_count:
                continue
            # Expand mask to (B, H, N) by broadcasting along H
            mask_bhn = mask.unsqueeze(1).expand(-1, H, -1)
            mu_q = row_w[mask_bhn].float().mean().item()
            mu_k = col_w[mask_bhn].float().mean().item()
            per_type_query.append(float(mu_q))
            per_type_key.append(float(mu_k))

        if len(per_type_query) < 2:
            return {
                "n_matrices": int(B * H), "skip_reason": "not enough token types",
                "n_types_observed": int(unique.numel()),
            }

        q_vals = torch.tensor(per_type_query)
        k_vals = torch.tensor(per_type_key)
        baseline = float(row_w.float().mean().item())
        return {
            "query_sensitivity_std": float(q_vals.std().item()),
            "key_sensitivity_std": float(k_vals.std().item()),
            "query_sensitivity_range": float((q_vals.max() - q_vals.min()).item()),
            "key_sensitivity_range": float((k_vals.max() - k_vals.min()).item()),
            "query_cv": float((q_vals.std() / max(abs(q_vals.mean()), 1e-30)).item()),
            "key_cv": float((k_vals.std() / max(abs(k_vals.mean()), 1e-30)).item()),
            "baseline_mean_weight": baseline,
            "n_token_types_used": int(q_vals.numel()),
            "n_token_types_observed": int(unique.numel()),
            "n_matrices": int(B * H),
        }
