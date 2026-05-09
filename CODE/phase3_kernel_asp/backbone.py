"""
backbone.py — Interface abstraite pour le Backbone de l'ASPLayer.

Le Backbone est *dérivé* du dictionnaire SCH phase 2 (DOC/03 §1.1, principe
Discovery > Reproduction). Cette interface laisse le choix concret en attente
de la sortie de phase 2.

Implémentations concrètes possibles (à instancier après phase 2) :
- ToeplitzConvBackbone : convolution causale longue FFT-based
- HankelSSMBackbone   : SSM générique (A, B, C appris, sans HiPPO/selectivity)
- CauchyBackbone      : interpolation rationnelle paramétrée
- CompositeBackbone   : somme/composition de plusieurs

Tant que la phase 2 n'a pas tranché, on utilise IdentityBackbone (passthrough)
pour les tests structurels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Backbone(nn.Module, ABC):
    """Interface : prend (B, N, d_model) → retourne (B, N, d_model).

    Doit être causale (ne consulter que x[:, :t, :] pour produire la sortie
    en position t). Le Backbone NE consulte PAS la matrice d'attention
    explicite — seulement la séquence d'entrée.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        ...

    @property
    @abstractmethod
    def class_name(self) -> str:
        """Nom de la classe SCH dont ce backbone est dérivé."""
        ...


class IdentityBackbone(Backbone):
    """Passthrough. Utilisé pour les tests structurels avant que phase 2 ait
    tranché la classe SCH dominante.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def class_name(self) -> str:
        return "identity"


class LinearBackbone(Backbone):
    """Backbone trivial linéaire causal. Pour tests / sanity checks. Une
    implémentation réelle dérivée de phase 2 remplacera celle-ci.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @property
    def class_name(self) -> str:
        return "linear"
