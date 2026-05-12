"""
vision.py — VisionOracle complet : adapter MNIST patches / CIFAR / DINOv2.

Spec : DOC/CATALOGUE §3.3 "Vision".

V2 complet : deux backends supportés
1. HFVisionBackend : transformers.AutoModel (DINOv2, ViT, CLIP-ViT)
2. MinimalViTBackend : MinimalTransformer non-causal + patch embedding

Régime Vision : omega = patch_size, delta = n_classes, entropy = image_complexity.

Le générateur d'images produit :
- Patches aléatoires de complexité variable (basse entropie = patches presque
  uniformes ; haute entropie = bruit blanc)
- Patch tokenizer : flatten + linear projection vers d_model

Sans checkpoint pré-entraîné, les attentions sont random (Xavier init).
Sprint S5 : remplacer par DINOv2-base + ImageNet val sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from catalog.oracles._minimal_transformer import (
    MinimalTransformer,
    MinimalTransformerSpec,
)
from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


@dataclass
class VisionModelSpec:
    img_size: int = 32
    patch_size: int = 4
    n_channels: int = 3
    n_classes: int = 10
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512


# ----------------------------------------------------------------------------
# Patch tokenizer
# ----------------------------------------------------------------------------

class PatchTokenizer(nn.Module):
    """Découpe une image en patches et projette en embeddings."""

    def __init__(self, spec: VisionModelSpec) -> None:
        super().__init__()
        if spec.img_size % spec.patch_size != 0:
            raise ValueError(f"img_size={spec.img_size} pas divisible par patch_size={spec.patch_size}")
        self.spec = spec
        self.n_patches = (spec.img_size // spec.patch_size) ** 2
        patch_dim = spec.n_channels * spec.patch_size * spec.patch_size
        self.proj = nn.Linear(patch_dim, spec.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images : (B, C, H, W) → patches (B, N_patches, patch_dim)
        B, C, H, W = images.shape
        ps = self.spec.patch_size
        patches = images.unfold(2, ps, ps).unfold(3, ps, ps)  # (B, C, n_h, n_w, ps, ps)
        patches = patches.contiguous().view(B, C, -1, ps, ps).permute(0, 2, 1, 3, 4)
        patches = patches.reshape(B, self.n_patches, C * ps * ps)
        return self.proj(patches)  # (B, n_patches, d_model)


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

class _VisionBackend:
    n_layers: int
    n_heads: int

    def forward_with_attn(self, images: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class MinimalViTBackend(_VisionBackend):
    """Backend ViT minimal : PatchTokenizer + MinimalTransformer bidirectionnel."""

    def __init__(self, spec: VisionModelSpec,
                 checkpoint_path: str | Path | None = None,
                 device: str = "cpu") -> None:
        self.spec = spec
        self.patch_tok = PatchTokenizer(spec)
        n_patches = self.patch_tok.n_patches
        m_spec = MinimalTransformerSpec(
            vocab_size=1,  # ignored, on bypass tok_emb
            d_model=spec.d_model, n_heads=spec.n_heads,
            n_layers=spec.n_layers, d_ff=spec.d_ff,
            max_seq_len=n_patches + 1,  # +1 si CLS token
            causal=False,
        )
        self.model = MinimalTransformer(m_spec)
        # On bypass tok_emb : remplace par identity (on injectera les patches projetés)
        self.model.tok_emb = nn.Identity()  # type: ignore[assignment]
        self.pos_emb = nn.Embedding(n_patches, spec.d_model)
        if checkpoint_path is not None:
            import sys
            p = Path(checkpoint_path)
            if p.is_file():
                try:
                    state = torch.load(p, map_location="cpu", weights_only=True)
                except Exception as e:
                    print(
                        f"[MinimalViTBackend] checkpoint {p} invalide "
                        f"({type(e).__name__}) — random init conservé.",
                        file=sys.stderr,
                    )
                    state = None
                if isinstance(state, dict):
                    if "state_dict" in state:
                        state = state["state_dict"]
                    try:
                        self.model.load_state_dict(state, strict=False)
                        if any(k.startswith("patch_tok") for k in state):
                            self.patch_tok.load_state_dict(
                                {k.removeprefix("patch_tok."): v for k, v in state.items()
                                 if k.startswith("patch_tok.")},
                                strict=False,
                            )
                    except Exception as e:
                        print(
                            f"[MinimalViTBackend] state_dict mismatch "
                            f"({type(e).__name__}) — random init conservé.",
                            file=sys.stderr,
                        )
        self.model.eval()
        self.patch_tok.eval()
        self.model.to(device)
        self.patch_tok.to(device)
        self.pos_emb.to(device)
        self.device = device
        self.n_layers = spec.n_layers
        self.n_heads = spec.n_heads

    @torch.no_grad()
    def forward_with_attn(self, images: torch.Tensor) -> list[torch.Tensor]:
        images = images.to(self.device)
        patches = self.patch_tok(images)  # (B, N, D)
        B, N, D = patches.shape
        positions = torch.arange(N, device=images.device)
        x = patches + self.pos_emb(positions)
        # Manuel forward des layers (on bypass tok_emb)
        attns: list[torch.Tensor] = []
        for layer in self.model.layers:
            x, a = layer(x, return_attn=True)
            if a is not None:
                attns.append(a)
        return attns


class HFVisionBackend(_VisionBackend):
    """Backend HuggingFace AutoModel (DINOv2, ViT, etc.). Requiert `transformers`."""

    def __init__(self, model_name_or_path: str, device: str = "cpu",
                 torch_dtype: torch.dtype = torch.float32) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:
            raise ImportError(
                "HFVisionBackend requiert `transformers`. "
                "uv add transformers ou utiliser MinimalViTBackend."
            ) from e
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.n_layers = getattr(self.model.config, "num_hidden_layers", 12)
        self.n_heads = getattr(self.model.config, "num_attention_heads", 12)

    @torch.no_grad()
    def forward_with_attn(self, images: torch.Tensor) -> list[torch.Tensor]:
        # images : (B, C, H, W) float [0, 1]
        # On bypass le processor : on passe pixel_values directement
        pixel_values = images.to(self.device)
        out = self.model(pixel_values=pixel_values, output_attentions=True)
        return list(out.attentions)


# ----------------------------------------------------------------------------
# Image generator
# ----------------------------------------------------------------------------

def synthetic_images_batch(
    n: int, spec: VisionModelSpec, complexity: float, seed: int = 0
) -> torch.Tensor:
    """Génère n images RGB avec complexité variable.

    complexity ∈ [0, 1] :
    - 0 : image quasi-uniforme (faible entropie, attentions block-diag)
    - 1 : bruit blanc (haute entropie, attentions diffuses)
    """
    rng = torch.Generator().manual_seed(seed)
    base = torch.rand(n, spec.n_channels, spec.img_size, spec.img_size,
                      generator=rng)
    if complexity < 0.5:
        # Lisse via average pooling
        kernel = max(2, int((1 - complexity * 2) * spec.patch_size))
        if kernel > 1:
            base = torch.nn.functional.avg_pool2d(base, kernel_size=kernel, stride=1,
                                                  padding=kernel // 2)
            base = base[..., :spec.img_size, :spec.img_size]
    return base


# ----------------------------------------------------------------------------
# VisionOracle
# ----------------------------------------------------------------------------

class VisionOracle(AbstractOracle):
    """Oracle Vision Transformer (MNIST patches, CIFAR, DINOv2)."""

    domain = "vision"

    def __init__(
        self,
        *,
        backend: _VisionBackend | None = None,
        checkpoint_path: str | Path | None = None,
        model_spec: VisionModelSpec | None = None,
        device: str = "cpu",
        oracle_id: str | None = None,
        n_examples_max: int = 64,
    ) -> None:
        if backend is None:
            if checkpoint_path is not None:
                ckpt = Path(checkpoint_path)
                if not ckpt.is_file():
                    raise FileNotFoundError(f"Checkpoint Vision introuvable : {ckpt}")
            spec = model_spec or VisionModelSpec()
            backend = MinimalViTBackend(spec, checkpoint_path=checkpoint_path, device=device)
            self.spec = spec
            stem = Path(checkpoint_path).stem if checkpoint_path else "random_init"
            self.oracle_id = oracle_id or f"vision_minimal_{stem}"
        else:
            self.spec = model_spec or VisionModelSpec()
            self.oracle_id = oracle_id or "vision_external"
        self.backend = backend
        self.device = device
        self.n_layers = backend.n_layers
        self.n_heads = backend.n_heads
        self.n_examples_max = n_examples_max

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        patch_size = int(regime.omega) if regime.omega is not None else 4
        # Si patch_size diffère du model_spec, on doit recréer le backend
        # — pour V1, on assume que regime.omega DEFINIT le spec et on
        # reconstruit le backend ad hoc si nécessaire
        if isinstance(self.backend, MinimalViTBackend) and self.backend.spec.patch_size != patch_size:
            new_spec = VisionModelSpec(
                img_size=self.spec.img_size,
                patch_size=patch_size,
                n_channels=self.spec.n_channels,
                n_classes=self.spec.n_classes,
                d_model=self.spec.d_model,
                n_heads=self.spec.n_heads,
                n_layers=self.spec.n_layers,
                d_ff=self.spec.d_ff,
            )
            self.backend = MinimalViTBackend(new_spec, device=self.device)
            self.spec = new_spec

        complexity = float(regime.entropy) if regime.entropy is not None else 0.5
        n_examples = min(n_examples, self.n_examples_max)
        images = synthetic_images_batch(n_examples, self.spec, complexity)
        attns = self.backend.forward_with_attn(images)
        attns_fp64 = [a.to(torch.float64).cpu() for a in attns]
        N = attns_fp64[0].shape[-1]

        return AttentionDump(
            attn=attns_fp64,
            omegas=torch.full((n_examples,), float(patch_size)),
            deltas=torch.full((n_examples,), float(regime.delta or self.spec.n_classes)),
            entropies=torch.full((n_examples,), complexity),
            tokens=torch.zeros(n_examples, N, dtype=torch.long),
            query_pos=torch.zeros(n_examples, dtype=torch.long),
            metadata={
                "oracle_id": self.oracle_id,
                "patch_size": patch_size,
                "img_size": self.spec.img_size,
                "domain": self.domain,
            },
        )

    def regime_grid(self) -> list[RegimeSpec]:
        patch_sizes = [4, 7, 14] if self.spec.img_size in (28, 32) else [4, 8, 16]
        # Filtrer pour ne garder que celles qui divisent img_size
        patch_sizes = [p for p in patch_sizes if self.spec.img_size % p == 0]
        if not patch_sizes:
            patch_sizes = [4]
        n_classes_subset = [2, 5, 10]
        out: list[RegimeSpec] = []
        for ps in patch_sizes:
            for nc in n_classes_subset:
                out.append(RegimeSpec(omega=ps, delta=nc, entropy=0.0))
        return out
