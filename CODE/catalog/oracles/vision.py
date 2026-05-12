"""
vision.py — VisionOracle : adapter MNIST patches / CIFAR pour catalog.

Spec : DOC/CATALOGUE §3.3 "Vision".

V1 sans training : adapter pour modèle Vision Transformer entraîné sur
MNIST par patches (ou CIFAR). Le "régime" pour Vision n'est pas (ω, Δ, ℋ)
mais des proxies de stress structurel adaptés :

- `patch_size` : taille de patch (granularité spatiale)
- `n_classes` : nombre de classes du dataset
- `image_complexity` : entropie de l'image (proxy ℋ)

Pour mapper sur RegimeSpec(omega, delta, entropy), on adopte la convention :
- omega ← patch_size
- delta ← n_classes
- entropy ← image_complexity (entropie image)

V1 squelette : checkpoint + dataset adapter à compléter Sprint S5+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from catalog.oracles.base import AbstractOracle, AttentionDump, RegimeSpec


@dataclass
class VisionModelSpec:
    img_size: int
    patch_size: int
    n_channels: int
    n_classes: int
    d_model: int
    n_heads: int
    n_layers: int


class VisionOracle(AbstractOracle):
    """Oracle Vision Transformer (MNIST patches, CIFAR, etc.)."""

    domain = "vision"

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_spec: VisionModelSpec | None = None,
        *,
        device: str = "cpu",
        oracle_id: str | None = None,
    ) -> None:
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint Vision introuvable : {ckpt}")
        self.checkpoint_path = ckpt
        self.model_spec = model_spec
        self.device = device
        self.oracle_id = oracle_id or f"vision_{ckpt.stem}"
        self.n_layers = model_spec.n_layers if model_spec else 6
        self.n_heads = model_spec.n_heads if model_spec else 8

    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Extrait n_examples attentions Vision pour ce régime.

        V1 : NotImplementedError ; à compléter Sprint S5 avec :
        - load_model_from_checkpoint (ViT-like)
        - dataset MNIST/CIFAR avec patch_size = regime.omega
        - forward pass + extraction attention par couche
        """
        raise NotImplementedError(
            "VisionOracle.extract_regime : Sprint S5 prep — V1 squelette. "
            "Compléter avec ViT loader + dataset patches."
        )

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille Vision : patch_size × n_classes_subset × complexity."""
        patch_sizes = [4, 7, 14]  # MNIST 28×28 / 4, 7, 14
        n_classes_subset = [2, 5, 10]
        out: list[RegimeSpec] = []
        for ps in patch_sizes:
            for nc in n_classes_subset:
                out.append(RegimeSpec(omega=ps, delta=nc, entropy=0.0))
        return out
