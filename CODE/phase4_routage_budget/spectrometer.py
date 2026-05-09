"""
spectrometer.py — Spectromètre.

Spec : DOC/04 §1, ROADMAP 4.1.

Entrée : signaux validés en phase 1.5 (concat des Max-Pool layer = (B, L, N)
par signal, donc (B, L · n_signaux, N) en entrée).

Sortie : α_t ∈ [0, 1] (utilisé par soft_mask phase 3) ou m_t directement.

Architecture V1 : MLP léger token-level, conv 1D optionnel pour capture
contextuelle locale.

Le Spectromètre est instancié post-phase-1.5, après que les signaux retenus
soient connus. Cette classe accepte donc input_dim variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SpectrometerConfig:
    input_dim: int                  # = L · n_signaux retenus
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.0
    output_mode: str = "alpha"      # "alpha" → ∈ [0, 1] ; "logits" → libre, à passer en sigmoid externe
    monotone_via_cumulative: bool = False  # si True : output ∈ R^R_max via cumulative softmax (V2)


class Spectrometer(nn.Module):
    """Spectromètre token-level. Entrée : (B, N, input_dim). Sortie : (B, N) ∈ [0,1]."""

    def __init__(self, cfg: SpectrometerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        prev = cfg.input_dim
        for _ in range(cfg.n_layers - 1):
            layers.append(nn.Linear(prev, cfg.hidden_dim))
            layers.append(nn.GELU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = cfg.hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """signals : (B, N, input_dim). Retourne α (B, N) ∈ [0,1]."""
        logits = self.net(signals).squeeze(-1)
        if self.cfg.output_mode == "alpha":
            return torch.sigmoid(logits)
        return logits


class FrozenAlphaSpectrometer(nn.Module):
    """Spectromètre figé à α=1 partout. Utilisé phase 4a warm-up et sanity
    checks d'ablation (m=1 → ASPLayer pleine, équivalent saturation phase 3).
    """

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return torch.ones(signals.shape[:2], device=signals.device, dtype=signals.dtype)
