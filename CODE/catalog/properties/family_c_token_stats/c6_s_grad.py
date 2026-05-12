"""
c6_s_grad.py — Property C6 : S_Grad — gradient norm task loss.

Spec : DOC/CATALOGUE §C6 "S_Grad : ‖∇_x_t L_task‖, sensibilité du loss
de tâche au token t".

V1 squelette : nécessite l'accès au modèle (forward + backward) au moment
de l'extraction Oracle. La Property reçoit seulement A en input, donc
NE PEUT PAS calculer S_Grad sans accès au modèle. Pour activer C6 il faut :
1. Pendant Oracle.extract_regime : enregistrer grad_norms[i, t] dans
   dump.metadata["s_grad"] avec shape (B, N).
2. C6 récupère depuis ctx.metadata et agrège.

Si non disponible : skip cleanly avec skip_reason explicite.
"""

from __future__ import annotations

import torch

from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


@register_property
class C6SGrad(Property):
    """C6 — proxy gradient task loss vs token (nécessite Oracle qui le calcule)."""

    name = "C6_s_grad"
    family = "C"
    cost_class = 2
    requires_fp64 = False
    scope = "per_regime"

    def compute(
        self, A: torch.Tensor, ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if A.ndim != 4:
            raise ValueError(f"A doit être (B, H, N, N), reçu {A.shape}")
        B, H, N, _ = A.shape

        s_grad = ctx.metadata.get("s_grad", None) if ctx.metadata else None
        if s_grad is None:
            return {
                "n_matrices": int(B * H),
                "skip_reason": "s_grad not in ctx.metadata — requires Oracle "
                "to inject gradient norms (training-time only)",
                "s_grad_available": False,
            }

        # Si disponible : agrège
        sg = s_grad.to(device=ctx.device, dtype=torch.float32)
        if sg.dim() != 2 or sg.shape != (B, N):
            return {
                "n_matrices": int(B * H),
                "skip_reason": f"s_grad shape mismatch : got {sg.shape}, expected ({B},{N})",
                "s_grad_available": False,
            }
        per_seq = sg.norm(dim=-1)  # (B,)
        per_token = sg  # (B, N)
        return {
            "s_grad_per_seq_median": float(per_seq.median().item()),
            "s_grad_per_seq_max": float(per_seq.max().item()),
            "s_grad_per_token_max_median": float(per_token.amax(dim=-1).median().item()),
            "s_grad_per_token_mean_median": float(per_token.mean(dim=-1).median().item()),
            "s_grad_log_sum_exp_median": float(
                sg.float().abs().clamp_min(1e-30).log().sum(dim=-1).median().item()
            ),
            "s_grad_available": True,
            "n_matrices": int(B * H),
        }
