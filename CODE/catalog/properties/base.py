"""
base.py — interface abstraite Property et PropertyContext.

Spec : DOC/CONTEXT.md §Property, §PropertyContext.

Une Property calcule UNE mesure mathématique sur une matrice d'attention.
Single-responsibility : NE logge PAS à MLflow elle-même, retourne un dict
sérialisable. La Battery agrège les sorties cross-régime en RegimeStats
(V3.5 distributional) et orchestre le logging.

Le PropertyContext sert de cache lazy pour pre-computations partagées entre
Properties (ex : SVD batchée réutilisée par B1_toeplitz et B5_block_diag).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch


PropertyScope = Literal["per_regime", "cross_regime"]


@dataclass
class PropertyContext:
    """Contexte partagé entre Properties d'un même run.

    Sert de cache lazy pour pre-computations coûteuses (SVD, projections,
    eigvalsh) réutilisées entre Properties de la même Battery. Évite la
    duplication N×M entre N properties et M régimes.

    Champs :
    - `device` : politique machine (résolue par `infra/machine.py`).
    - `dtype` : précision (FP64 stricte spec, FP32 GPU Blackwell consumer).
    - `regime` : paramètres de stress du régime courant (ω, Δ, ℋ) — seulement
                 défini si scope=per_regime.
    - `cache` : dict opaque pour mémoïser les calculs partagés.
    - `metadata` : informations transversales (oracle_id, seed, run_name).
    """

    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    regime: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def cache_key(self, *parts: Any) -> str:
        """Construit une clé de cache déterministe à partir de fragments."""
        return ":".join(str(p) for p in parts)

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Lazy memoization : compute_fn() exécuté une seule fois par clé."""
        if key not in self.cache:
            self.cache[key] = compute_fn()
        return self.cache[key]

    def clear_cache(self) -> None:
        """Libère le cache (entre régimes ou après une Property volumineuse)."""
        self.cache.clear()


class Property(ABC):
    """Une mesure mathématique sur une matrice d'attention.

    Spec : DOC/00b catalogue (A1-W6, 131 propriétés).
    Sous-classes : un fichier dédié par mesure dans
    `catalog/properties/family_<x>_<nom>/`.

    Convention metadata (class attributes à override) :
    - `name` : identifiant unique (ex `A1_r_eff_theta099`)
    - `family` : lettre catalogue DOC/00b (A, B, C, ..., W)
    - `cost_class` : 1 (fast < 1s/régime) à 5 (slow > 1min)
    - `requires_fp64` : True si la précision FP32 dégrade le verdict
    - `requires_symmetric` : True si la mesure suppose A = Aᵀ
    - `scope` : "per_regime" (défaut, A par régime) ou "cross_regime"
                (A indexé par tous les régimes, ex : head_specialization)
    """

    # Metadata — à override dans les sous-classes
    name: str = ""
    family: str = ""
    cost_class: int = 3
    requires_fp64: bool = False
    requires_symmetric: bool = False
    scope: PropertyScope = "per_regime"

    @abstractmethod
    def compute(
        self,
        A: torch.Tensor | dict[Any, torch.Tensor],
        ctx: PropertyContext,
    ) -> dict[str, float | int | str | bool]:
        """Calcule la mesure.

        Args
        ----
        A : Tensor (B, H, N, N) FP64 si scope="per_regime" ; dict
            {regime_key → Tensor (B, H, N, N)} si scope="cross_regime".
        ctx : PropertyContext partagé pour cache + metadata.

        Returns
        -------
        Dict de scalaires nommés. Tous JSON-sérialisables (float, int, str,
        bool). Pas de Tensor, pas de dataclass, pas de None. La Battery
        loggue ce dict à MLflow et agrège cross-régime en RegimeStats.

        Convention de nommage des clés : préfixer par `name` court (ex
        `toeplitz_distance`, `class_winner`) pour éviter les collisions
        cross-Property dans la sortie agrégée.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, family={self.family!r}, "
            f"cost={self.cost_class}, scope={self.scope})"
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Garde-fou : vérifie que les sous-classes ont rempli leur metadata."""
        super().__init_subclass__(**kwargs)
        # On accepte des classes abstraites intermédiaires (sans name)
        if not getattr(cls, "__abstractmethods__", None) and cls.__name__ != "Property":
            if not cls.name:
                raise TypeError(
                    f"Property subclass {cls.__name__} doit définir `name` "
                    f"(class attribute non vide)."
                )
            if cls.family not in {chr(c) for c in range(ord("A"), ord("W") + 1)}:
                raise TypeError(
                    f"Property {cls.name} : family={cls.family!r} doit être "
                    f"une lettre A-W (catalogue DOC/00b)."
                )
            if cls.cost_class not in (1, 2, 3, 4, 5):
                raise TypeError(
                    f"Property {cls.name} : cost_class={cls.cost_class} doit "
                    f"être un entier 1-5."
                )
