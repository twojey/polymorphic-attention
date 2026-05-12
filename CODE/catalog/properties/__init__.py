"""Properties — mesures mathématiques individuelles sur matrices d'attention.

Une Property = un fichier ~30-80 lignes, identifiée par `name` et `family`,
sérialisable (retourne dict de scalaires). Composée dans Battery selon niveau.

Voir DOC/CONTEXT.md §Property + DOC/00b catalogue exhaustif.
"""

from catalog.properties.base import Property, PropertyContext, PropertyScope
from catalog.properties.registry import REGISTRY, register_property

__all__ = [
    "Property",
    "PropertyContext",
    "PropertyScope",
    "REGISTRY",
    "register_property",
]
