"""
sweeps.py — balayages monovariés et croisés sur les axes ω/Δ/ℋ.

Spec : DOC/01 §2 (principe d'isolation : monovariés d'abord, puis croisés).

Pour chaque sweep, retourne un ConcatDataset (consommable par DataLoader)
+ un dict d'index par valeur d'axe pour l'agrégation aval.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import ConcatDataset

from phase1_metrologie.ssg.structure_mnist import StructureMNISTConfig, StructureMNISTDataset


@dataclass
class SweepIndex:
    """Index permettant de retrouver les exemples d'un régime donné."""
    axis_name: str
    axis_value: float
    start: int       # premier index dans le ConcatDataset
    stop: int        # exclusif


def monovariate_sweep(
    *,
    axis: str,
    values: list[float],
    base: StructureMNISTConfig,
) -> tuple[ConcatDataset, list[SweepIndex]]:
    """Balayage monovarié : un seul axe varie, autres figés à leur référence."""
    datasets: list[StructureMNISTDataset] = []
    indices: list[SweepIndex] = []
    cursor = 0
    for v in values:
        kwargs = base.__dict__.copy()
        kwargs[axis] = type(getattr(base, axis))(v)  # cast au bon type
        cfg = StructureMNISTConfig(**kwargs)
        ds = StructureMNISTDataset(cfg)
        datasets.append(ds)
        indices.append(SweepIndex(axis_name=axis, axis_value=float(v),
                                   start=cursor, stop=cursor + len(ds)))
        cursor += len(ds)
    return ConcatDataset(datasets), indices


def crossed_sweep(
    *,
    axis1: str, values1: list[float],
    axis2: str, values2: list[float],
    base: StructureMNISTConfig,
) -> tuple[ConcatDataset, list[SweepIndex], list[SweepIndex]]:
    """Balayage croisé : deux axes varient en grille, autres figés.

    Retourne (dataset, indices_axis1, indices_axis2). Pour identifier un
    régime (a, b), prendre l'intersection des index a (axis1) et b (axis2).
    """
    datasets: list[StructureMNISTDataset] = []
    indices1: list[SweepIndex] = []
    indices2: list[SweepIndex] = []
    cursor = 0
    for v1 in values1:
        for v2 in values2:
            kwargs = base.__dict__.copy()
            kwargs[axis1] = type(getattr(base, axis1))(v1)
            kwargs[axis2] = type(getattr(base, axis2))(v2)
            cfg = StructureMNISTConfig(**kwargs)
            ds = StructureMNISTDataset(cfg)
            datasets.append(ds)
            indices1.append(SweepIndex(axis_name=axis1, axis_value=float(v1),
                                        start=cursor, stop=cursor + len(ds)))
            indices2.append(SweepIndex(axis_name=axis2, axis_value=float(v2),
                                        start=cursor, stop=cursor + len(ds)))
            cursor += len(ds)
    return ConcatDataset(datasets), indices1, indices2
