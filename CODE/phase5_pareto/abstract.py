"""
abstract.py — Interface ASPLayer pour les tests phase 5.

Phase 5 doit pouvoir tester n'importe quelle implémentation de l'ASPLayer
(la version finale construite phase 3 + phase 4) sans coupler les tests à
l'architecture spécifique.

Cette interface capture le contrat minimal : "applique ASPLayer sur une
séquence d'entrée et retourne (output, R_target_par_token)".
"""

from __future__ import annotations

from typing import Protocol

import torch


class ASPLayerEvaluable(Protocol):
    """Contrat pour qu'un ASPLayer soit testable phase 5.

    Doit fournir :
    - `forward_eval(tokens, query_pos)` → (logits, R_target_par_token)
    - `R_max` accessible (pour test 6c R_max/2)
    """

    def forward_eval(
        self, tokens: torch.Tensor, query_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward complet sur un batch.

        Retourne :
        - logits : (B, n_classes)
        - R_target_par_token : (B, N) — rang alloué par le Spectromètre par token
        """
        ...

    @property
    def R_max(self) -> int:
        ...
