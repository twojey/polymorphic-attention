"""
base.py — interface AbstractOracle + structures de données AttentionDump et RegimeSpec.

Un Oracle fournit des matrices d'attention pour un domaine donné. Le seam
Oracle permet la comparaison cross-domain (SMNIST × LL × vision) qui est
le cœur de la Partie 1 (DOC/CONTEXT.md §Oracle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RegimeSpec:
    """Spécification d'un régime de stress du sweep.

    Pour SMNIST : (omega, delta, entropy). Pour LL : autres axes (sequence_length,
    perplexity bucket, etc.). Le contrat Oracle ↔ Battery est que `key` est
    hashable et identifie uniquement le régime.
    """

    omega: int | None = None
    delta: int | None = None
    entropy: float | None = None
    custom: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        """Clé hashable pour usage en dict/set."""
        custom_items = tuple(sorted(self.custom.items()))
        return (self.omega, self.delta, self.entropy, custom_items)

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass
class AttentionDump:
    """Dump d'attention pour un (ou plusieurs) régime(s).

    Contrat fixe Oracle ↔ Battery :
    - `attn` : list[L tensors (B, H, N, N)] FP64 (ou FP32 selon Oracle)
    - `omegas`, `deltas`, `entropies` : tensors (B,) — axes de stress
    - `tokens` : (B, N) int — input pour replay
    - `query_pos` : (B,) int — position du token cible (classification SMNIST)
    - `metadata` : oracle_id, seed, run_id, etc.

    Compatible avec le format dump phase 1 V2 existant pour migration douce.
    """

    attn: list[torch.Tensor]
    omegas: torch.Tensor
    deltas: torch.Tensor
    entropies: torch.Tensor
    tokens: torch.Tensor | None = None
    query_pos: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.attn)

    @property
    def n_heads(self) -> int:
        return self.attn[0].size(1) if self.attn else 0

    @property
    def n_examples(self) -> int:
        return self.attn[0].size(0) if self.attn else 0

    @property
    def seq_len(self) -> int:
        return self.attn[0].size(2) if self.attn else 0

    def validate(self) -> None:
        """Vérifie l'invariance cross-layer (même L, H, B, N). Lève SystemExit
        avec message clair si corrompu."""
        if not self.attn:
            raise ValueError("AttentionDump : `attn` est vide")
        ref = self.attn[0].shape
        for ell, A in enumerate(self.attn):
            if A.shape != ref:
                raise ValueError(
                    f"AttentionDump : layer {ell} shape {A.shape} != ref {ref}"
                )
        B = self.attn[0].size(0)
        for name, t in (("omegas", self.omegas), ("deltas", self.deltas),
                        ("entropies", self.entropies)):
            if t.numel() != B:
                raise ValueError(
                    f"AttentionDump : `{name}` size {t.numel()} != B={B}"
                )


class AbstractOracle(ABC):
    """Source d'attention dense pour la Battery.

    Sous-classes :
    - `SyntheticOracle` : génère des attentions softmax aléatoires (tests)
    - `SMNISTOracle` : wrappe phase 1 (à coder en suivant)
    - `LanguageOracle` : Oracle LL (à coder Sprint S7)

    Convention metadata (class attributes) :
    - `oracle_id` : identifiant unique (ex `smnist_e2f0b5e`)
    - `domain` : nom du domaine (ex `smnist`, `language`, `vision`)
    """

    oracle_id: str = ""
    domain: str = ""

    @abstractmethod
    def extract_regime(
        self, regime: RegimeSpec, n_examples: int
    ) -> AttentionDump:
        """Extrait n_examples matrices d'attention pour ce régime de stress.

        L'Oracle gère ses propres ressources (model load, forward, etc.).
        La Battery appelle uniquement cette méthode.

        Retourne un AttentionDump conforme au contrat.
        """
        ...

    def regime_grid(self) -> list[RegimeSpec]:
        """Grille de régimes par défaut couverts par cet Oracle.

        Override dans sous-classes pour fournir le sweep canonique. Battery
        peut aussi recevoir une liste custom via override.
        """
        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.oracle_id!r}, domain={self.domain!r})"
