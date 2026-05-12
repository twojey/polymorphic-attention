"""
m4_token_class_sensitivity.py — Property M4 : sensitivity par classe de token.

Si ctx.metadata['tokens'] disponible (Tensor (B, N) entiers), on calcule :
- Pour chaque classe de token c, la moyenne et variance des entropies
  H(A[b, h, t, :]) où le token à position t est de classe c
- Différence inter-classes = à quel point l'attention diffère selon le
  type de token attended

Diagnostic : sur LLM, on s'attend à ce que stopwords aient entropies très
différents des content words.

Skip cleanly si tokens absent.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class M4TokenClassSensitivity(Property):
    """M4 — variance des entropies row-wise groupée par classe de token."""

    name = "M4_token_class_sensitivity"
    family = "M"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def __init__(self, eps_floor: float = 1e-30, min_count_per_class: int = 5) -> None:
        self.eps_floor = eps_floor
        self.min_count_per_class = min_count_per_class

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        tokens = ctx.metadata.get("tokens", None) if ctx.metadata else None
        if tokens is None:
            return {
                "skip_reason": "tokens absent in ctx.metadata",
                "n_matrices": int(B * H),
            }
        tok = tokens.to(device=ctx.device)
        if tok.shape != (B, N):
            return {
                "skip_reason": f"tokens shape {tok.shape} != ({B},{N})",
                "n_matrices": int(B * H),
            }

        A_work = A.to(device=ctx.device, dtype=ctx.dtype).clamp_min(self.eps_floor)
        H_row = -(A_work * A_work.log()).sum(dim=-1)  # (B, H, N)

        # Group by token class
        unique = torch.unique(tok)
        class_means = []
        class_counts = []
        for c in unique.tolist():
            mask = (tok == c)  # (B, N)
            count = int(mask.sum().item())
            if count < self.min_count_per_class:
                continue
            # H_row at positions where tok == c. Broadcast (B, N) -> (B, H, N)
            mask_exp = mask.unsqueeze(1).expand(-1, H, -1)
            selected = H_row[mask_exp]  # flat
            class_means.append(float(selected.float().mean().item()))
            class_counts.append(count)

        if len(class_means) < 2:
            return {
                "skip_reason": f"<2 classes with ≥ {self.min_count_per_class} tokens",
                "n_classes_total": int(unique.numel()),
                "n_classes_valid": int(len(class_means)),
                "n_matrices": int(B * H),
            }

        cm = torch.tensor(class_means)
        inter_class_var = float(cm.var(unbiased=False).item())
        inter_class_range = float((cm.max() - cm.min()).item())

        return {
            "inter_class_entropy_variance": inter_class_var,
            "inter_class_entropy_range": inter_class_range,
            "inter_class_entropy_max": float(cm.max().item()),
            "inter_class_entropy_min": float(cm.min().item()),
            "n_classes_valid": int(len(class_means)),
            "n_matrices": int(B * H),
        }
