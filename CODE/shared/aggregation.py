"""
aggregation.py — statistiques par régime du SSG.

Spec : DOC/01 §5 (E[rang_Hankel]/N par régime), DOC/02 (SCH comme distribution :
médiane, IQR, p10, p90, queue).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeStats:
    n: int
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    iqr: float
    tail_99: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n": self.n, "mean": self.mean, "median": self.median, "std": self.std,
            "p10": self.p10, "p25": self.p25, "p75": self.p75, "p90": self.p90,
            "iqr": self.iqr, "tail_99": self.tail_99,
        }


def regime_stats(values: np.ndarray) -> RegimeStats:
    """Statistiques distributionnelles d'une métrique sur un régime.

    Conforme à DOC/02 §V3.5 (SCH comme distribution P(r_eff | stress, domaine)).
    """
    if values.size == 0:
        return RegimeStats(n=0, mean=float("nan"), median=float("nan"), std=float("nan"),
                           p10=float("nan"), p25=float("nan"), p75=float("nan"), p90=float("nan"),
                           iqr=float("nan"), tail_99=float("nan"))
    p10, p25, p50, p75, p90, p99 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    return RegimeStats(
        n=int(values.size),
        mean=float(values.mean()),
        median=float(p50),
        std=float(values.std()),
        p10=float(p10), p25=float(p25), p75=float(p75), p90=float(p90),
        iqr=float(p75 - p25),
        tail_99=float(p99),
    )


def aggregate_by_regime(
    values: np.ndarray,
    *,
    omega: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
) -> dict[tuple[str, float], RegimeStats]:
    """Pour chaque axe fourni, agrège `values` par valeur de l'axe.

    Retourne un dict (axis, value) -> RegimeStats. Permet de produire des
    courbes monovariées sans coupler les axes (DOC/01 §2 — principe d'isolation).
    """
    out: dict[tuple[str, float], RegimeStats] = {}
    for axis_name, axis_vals in (("omega", omega), ("delta", delta), ("entropy", entropy)):
        if axis_vals is None:
            continue
        for v in np.unique(axis_vals):
            mask = axis_vals == v
            out[(axis_name, float(v))] = regime_stats(values[mask])
    return out


def aggregate_by_regime_2d(
    values: np.ndarray,
    *,
    axis1_name: str, axis1: np.ndarray,
    axis2_name: str, axis2: np.ndarray,
) -> dict[tuple[float, float], RegimeStats]:
    """Cross-sweep : agrège par paire (axis1_value, axis2_value)."""
    out: dict[tuple[float, float], RegimeStats] = {}
    pairs = list(zip(axis1, axis2, strict=True))
    seen = {(float(a), float(b)) for a, b in pairs}
    for a, b in sorted(seen):
        mask = (axis1 == a) & (axis2 == b)
        out[(a, b)] = regime_stats(values[mask])
    return out
