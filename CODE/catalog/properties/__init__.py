"""Properties — mesures mathématiques individuelles sur matrices d'attention.

Une Property = un fichier ~30-80 lignes, identifiée par `name` et `family`,
sérialisable (retourne dict de scalaires). Composée dans Battery selon niveau.

Voir DOC/CONTEXT.md §Property + DOC/00b catalogue exhaustif.

Auto-discovery : ce __init__ importe tous les sous-modules family_*/ pour
déclencher les `@register_property` décorateurs. Les Properties deviennent
ainsi accessibles via `REGISTRY.get(name)` sans imports manuels par caller.
"""

from catalog.properties.base import Property, PropertyContext, PropertyScope
from catalog.properties.registry import REGISTRY, register_property


def _discover_properties() -> None:
    """Importe récursivement tous les modules family_*/ pour enregistrer les Properties.

    Idempotent : appelable plusieurs fois sans effet de bord. Silencieux sur
    les sous-modules qui ne contiennent pas de Property (utility files).
    """
    import importlib
    import pkgutil
    from pathlib import Path

    pkg_root = Path(__file__).parent
    for sub in pkg_root.iterdir():
        if sub.is_dir() and sub.name.startswith("family_"):
            sub_pkg = f"catalog.properties.{sub.name}"
            for mod_info in pkgutil.iter_modules([str(sub)]):
                importlib.import_module(f"{sub_pkg}.{mod_info.name}")


_discover_properties()


__all__ = [
    "Property",
    "PropertyContext",
    "PropertyScope",
    "REGISTRY",
    "register_property",
]
