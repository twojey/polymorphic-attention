"""
registry.py — registre global des Properties enregistrées.

Permet à Battery de filtrer/composer par metadata (family, cost_class) sans
imports manuels. Une Property se déclare via `@register_property` ou via
`REGISTRY.add(Property_class)`.

Pattern : équivalent à un service locator scopé au catalogue. Permet à
Battery de demander "donne-moi toutes les Properties de family B avec
cost_class ≤ 3" sans connaître les classes concrètes.
"""

from __future__ import annotations

from typing import TypeVar

from catalog.properties.base import Property

T = TypeVar("T", bound=type[Property])


class PropertyRegistry:
    """Registre des Properties disponibles, indexé par nom + filtrable."""

    def __init__(self) -> None:
        self._by_name: dict[str, type[Property]] = {}

    def register(self, cls: type[Property]) -> type[Property]:
        """Ajoute une classe Property au registre. Idempotent."""
        if not cls.name:
            raise ValueError(f"{cls.__name__} sans `name` : ne peut être registered.")
        if cls.name in self._by_name and self._by_name[cls.name] is not cls:
            raise ValueError(
                f"Property name '{cls.name}' déjà enregistrée par "
                f"{self._by_name[cls.name].__name__} ; conflit avec {cls.__name__}."
            )
        self._by_name[cls.name] = cls
        return cls

    def get(self, name: str) -> type[Property]:
        if name not in self._by_name:
            raise KeyError(
                f"Property '{name}' inconnue. Disponibles : "
                f"{sorted(self._by_name.keys())}"
            )
        return self._by_name[name]

    def filter(
        self,
        *,
        family: str | None = None,
        cost_class_max: int | None = None,
        scope: str | None = None,
        requires_fp64: bool | None = None,
    ) -> list[type[Property]]:
        """Filtre les Properties par critères. None = pas de filtre."""
        out: list[type[Property]] = []
        for cls in self._by_name.values():
            if family is not None and cls.family != family:
                continue
            if cost_class_max is not None and cls.cost_class > cost_class_max:
                continue
            if scope is not None and cls.scope != scope:
                continue
            if requires_fp64 is not None and cls.requires_fp64 != requires_fp64:
                continue
            out.append(cls)
        return sorted(out, key=lambda c: (c.family, c.name))

    def all(self) -> list[type[Property]]:
        return sorted(self._by_name.values(), key=lambda c: (c.family, c.name))

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        return f"PropertyRegistry(n={len(self)})"


# Singleton global
REGISTRY = PropertyRegistry()


def register_property(cls: T) -> T:
    """Décorateur : enregistre une Property class dans le registre global.

    Usage :
        @register_property
        class A1Reff(Property):
            name = "A1_r_eff_theta099"
            family = "A"
            ...
    """
    REGISTRY.register(cls)  # type: ignore[arg-type]
    return cls
