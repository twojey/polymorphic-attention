"""
evaluable_wrapper.py — wrapper concret ASPLayerEvaluable pour phase 5.

Spec : DOC/05_phase_pareto.md.

Cette classe combine ASPLayer (phase 3) + Spectrometer (phase 4) +
classifier head (entraîné phase 4) pour fournir l'API attendue par
les tests phase 5 :

    forward_eval(tokens, query_pos) → (logits, R_target_per_token)

Convention :
- tokens : (B, N) int64 (vocab ids)
- query_pos : (B,) int64 — position du token query dans chaque exemple
- logits : (B, n_classes)
- R_target_per_token : (B, N) — rang alloué par Spectrometer par token

Le wrapper charge depuis checkpoint si dispo, sinon instancie un layer
aléatoire (utile pour smoke run hors pod).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from phase3_kernel_asp.asp_layer import ASPLayer, ASPLayerConfig
from phase4_routage_budget.spectrometer import Spectrometer, SpectrometerConfig


class ASPLayerEvaluableImpl(nn.Module):
    """Adapter concret implémentant l'interface ASPLayerEvaluable.

    Charge ASPLayer + Spectrometer + classifier head depuis un checkpoint
    phase 4. Si checkpoint absent : init aléatoire pour smoke run.
    """

    def __init__(
        self,
        *,
        d_model: int,
        R_max: int,
        seq_len: int,
        n_classes: int,
        signals_dim: int = 1,
        device: str = "cpu",
        vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self._R_max = R_max
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.signals_dim = signals_dim
        self.vocab_size = vocab_size
        self._device = device

        # Embedding tokens → vecteurs d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        # ASPLayer (phase 3 V2 attention mode)
        self.asp_layer = ASPLayer(
            ASPLayerConfig(d_model=d_model, R_max=R_max,
                           delta_attn_mode="attention"),
        )
        # Spectrometer learnable (input = signaux dummy de shape signals_dim)
        self.spectrometer = Spectrometer(
            SpectrometerConfig(input_dim=signals_dim, hidden_dim=32,
                                n_layers=2, output_mode="alpha"),
        )
        # Classifier head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )
        # Tout sur device
        self.to(device=device, dtype=torch.float32)

    @property
    def R_max(self) -> int:
        return self._R_max

    def _build_signals_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Construit un signal proxy à partir des tokens (B, N) → (B, N, signals_dim).

        Smoke proxy : norm L2 de l'embedding par token (1 signal). Sur pod
        réel, on remplacera par les signaux phase 1.5 retenus.
        """
        emb = self.embed(tokens)  # (B, N, d_model)
        sig = emb.norm(dim=-1, keepdim=True)  # (B, N, 1)
        if self.signals_dim > 1:
            sig = sig.expand(-1, -1, self.signals_dim).clone()
        return sig

    def forward_eval(
        self, tokens: torch.Tensor, query_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward complet : embed → ASPLayer + Spectrometer → cls head.

        Retourne (logits (B, n_classes), R_target (B, N)). Mode no_grad
        (tests phase 5 sont eval-only) — les tensors retournés sont detached.
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens doit être (B, N), reçu {tokens.shape}")
        B, N = tokens.shape
        with torch.no_grad():
            x = self.embed(tokens)  # (B, N, d_model)
            signals = self._build_signals_from_tokens(tokens)
            alpha = self.spectrometer(signals)  # (B, N) ∈ [0, 1]
            R_target = alpha * self._R_max  # (B, N) ∈ [0, R_max]

            from phase3_kernel_asp.soft_mask import soft_mask
            mask = soft_mask(
                alpha=alpha, R_max=self._R_max,
                beta=self.asp_layer.cfg.soft_mask_beta,
            )
            out = self.asp_layer.forward_with_mask(x, mask)  # (B, N, d)

            q_idx = query_pos.long().to(out.device)
            token_query = out.gather(
                1, q_idx.view(B, 1, 1).expand(-1, 1, self.d_model),
            ).squeeze(1)
            logits = self.cls_head(token_query)
        return logits.detach(), R_target.detach()

    @classmethod
    def load_or_init(
        cls,
        *,
        checkpoint_path: Path | None,
        d_model: int,
        R_max: int,
        seq_len: int,
        n_classes: int,
        signals_dim: int = 1,
        device: str = "cpu",
        vocab_size: int = 256,
    ) -> "ASPLayerEvaluableImpl":
        """Charge depuis checkpoint, ou init aléatoire si absent.

        Le checkpoint doit contenir un state_dict produit par phase 4
        (clés : embed, asp_layer, spectrometer, cls_head). Si la signature
        ne correspond pas → init aléatoire (smoke run).
        """
        impl = cls(
            d_model=d_model, R_max=R_max, seq_len=seq_len,
            n_classes=n_classes, signals_dim=signals_dim,
            device=device, vocab_size=vocab_size,
        )
        if checkpoint_path is not None and checkpoint_path.is_file():
            try:
                state = torch.load(
                    checkpoint_path, map_location=device, weights_only=False,
                )
                if isinstance(state, dict) and "asp_layer" in state:
                    impl.load_state_dict(state)
                    return impl
            except Exception:  # noqa: BLE001 — fallback smoke
                pass
        # Sinon : init aléatoire (smoke run)
        return impl
