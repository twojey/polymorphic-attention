"""
s_kl.py — signal S_KL : KL local entre p(x_t | x_<t) et un baseline empirique global.

Spec : DOC/01b §1, §8 (calibration baseline avant le test).

Coût : O(V) par token, V = vocabulaire.

Convention :
- p(x_t | x_<t) : distribution prédite par le modèle (next-token prediction).
  Pour l'Oracle classification (output = 1 logit final), on utilise les
  attention weights par couche/tête comme proxy de "distribution d'attention".
- Baseline empirique : moyenne empirique de p(x_t | x_<t) calculée une fois
  sur un échantillon stationnaire de bruit, avant le banc de test.

Pour V1 phase 1.5, on calcule S_KL **sur les distributions d'attention**
A[ℓ, h, t, :] de chaque couche × tête, et on les compare à un baseline
empirique précalibré sur du bruit. Le baseline = distribution moyenne
d'attention sur des séquences purement aléatoires.

Convention de réduction (DOC/01b §3) : Max-Pool sur les têtes pour obtenir
un signal par couche × token, puis Concat sur la deep-stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GlobalKLBaseline:
    """Baseline empirique de l'attention sur séquences-bruit.

    `baseline[ℓ, h, :]` = distribution moyenne d'attention de la couche ℓ,
    tête h, calculée sur des séquences sans structure.

    Calibré une fois pour toutes avant le banc de test (DOC/01b §8).
    """

    baseline: torch.Tensor   # (L, H, N) FP64, normalisé sur N

    @classmethod
    def from_attention_dumps(cls, dumps: list[torch.Tensor]) -> "GlobalKLBaseline":
        """Construit le baseline à partir d'extractions Oracle sur du bruit.

        `dumps` : liste de matrices (B, H, N, N) FP64 par couche.
        Le baseline pour la couche ℓ : moyenne sur (batch, query position) de A.
        """
        baseline_per_layer = []
        for A_layer in dumps:
            # A : (B, H, N, N). Moyenne sur batch et query positions → (H, N)
            mean_dist = A_layer.mean(dim=(0, 2))
            mean_dist = mean_dist / mean_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            baseline_per_layer.append(mean_dist)
        baseline = torch.stack(baseline_per_layer, dim=0)  # (L, H, N)
        return cls(baseline=baseline)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p || q), p et q sur le dernier axe."""
    p_safe = p.clamp_min(eps)
    q_safe = q.clamp_min(eps)
    return (p_safe * (p_safe.log() - q_safe.log())).sum(dim=-1)


def compute_s_kl(
    attn_per_layer: list[torch.Tensor],
    baseline: GlobalKLBaseline,
) -> torch.Tensor:
    """S_KL pour chaque (couche, tête, exemple, token).

    `attn_per_layer` : liste de L tensors (B, H, N, N) FP64.
    Retourne (L, B, H, N) FP64.
    """
    L = len(attn_per_layer)
    B, H, N, _ = attn_per_layer[0].shape
    device = attn_per_layer[0].device

    out = torch.zeros(L, B, H, N, dtype=torch.float64, device=device)
    for ell, A_layer in enumerate(attn_per_layer):
        # A_layer : (B, H, N, N). Pour chaque token t, p_t = A_layer[..., t, :].
        # baseline.baseline[ell] : (H, N), broadcast sur batch et token.
        baseline_dist = baseline.baseline[ell].to(A_layer.device)  # (H, N)
        # Reshape pour broadcast avec p (B, H, N, N) → q (1, H, 1, N)
        q = baseline_dist.unsqueeze(0).unsqueeze(2)  # (1, H, 1, N)
        out[ell] = _kl_divergence(A_layer, q)
    return out
