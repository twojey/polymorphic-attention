"""
levels.py — niveaux de Battery paramétrables (minimal | principal | extended | full | research).

Spec : DOC/00_FONDATIONS.md §Battery.

Une fonction par niveau retourne une `Battery` pré-composée. La sélection
des Properties activées suit la convention `cost_class` (≤ 2 pour minimal,
≤ 3 principal, ≤ 4 extended, tout pour full, recherche pour research).

Pattern : on construit en lisant le PropertyRegistry, donc l'ajout d'une
Property la rend automatiquement disponible au niveau approprié (selon son
`cost_class`).
"""

from __future__ import annotations

import torch

import catalog.properties  # noqa: F401 — déclenche auto-discovery
from catalog.batteries.base import Battery
from catalog.properties.registry import REGISTRY


def level_minimal(
    *, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> Battery:
    """Battery minimal — Properties cost_class ≤ 1 (smoke check ~1 min)."""
    props = [cls() for cls in REGISTRY.filter(cost_class_max=1)]
    return Battery(props, name="minimal", device=device, dtype=dtype)


def level_principal(
    *, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> Battery:
    """Battery principal — Properties cost_class ≤ 2 (priorité haute ~10 min)."""
    props = [cls() for cls in REGISTRY.filter(cost_class_max=2)]
    return Battery(props, name="principal", device=device, dtype=dtype)


def level_extended(
    *, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> Battery:
    """Battery extended — Properties cost_class ≤ 3 (priorité med+haute ~1h)."""
    props = [cls() for cls in REGISTRY.filter(cost_class_max=3)]
    return Battery(props, name="extended", device=device, dtype=dtype)


def level_full(
    *, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> Battery:
    """Battery full — toutes les Properties enregistrées (sauf research)."""
    props = [cls() for cls in REGISTRY.filter(cost_class_max=4)]
    return Battery(props, name="full", device=device, dtype=dtype)


def level_research(
    *, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> Battery:
    """Battery research — toutes incluant Properties cost_class 5 (frontières)."""
    props = [cls() for cls in REGISTRY.all()]
    return Battery(props, name="research", device=device, dtype=dtype)
