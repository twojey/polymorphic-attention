"""
pareto.py — Test 5d (Pareto).

Spec : DOC/05 §4, ROADMAP 5.6.

Construction de la frontière Pareto sur (qualité, FLOPs) ou (qualité, mémoire)
ou (qualité, latence). Critère : ASP sur la frontière, dominance stricte sur
au moins une famille de tâches.

Comparateurs domain-aware (DOC/05 §4) :
- sequence-domain : Transformer plein, Mamba2, Linear Attention, Hyena/RWKV, MoD
- vision-domain   : ViT, ConvNet, MLP-Mixer
- code-domain     : Transformer plein, MoD

Pas de comparaison cross-domain.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEvaluation:
    name: str
    domain: str
    quality: float          # accuracy ou métrique pertinente
    flops_per_token: float
    memory_peak_mb: float
    latency_s: float


def is_dominated(p: ModelEvaluation, q: ModelEvaluation) -> bool:
    """p est dominé par q si q ≥ p sur quality ET q ≤ p sur tous les coûts,
    avec au moins une inégalité stricte.
    """
    if p.domain != q.domain:
        return False
    quality_ok = q.quality >= p.quality
    flops_ok = q.flops_per_token <= p.flops_per_token
    memory_ok = q.memory_peak_mb <= p.memory_peak_mb
    latency_ok = q.latency_s <= p.latency_s
    strict = (
        q.quality > p.quality
        or q.flops_per_token < p.flops_per_token
        or q.memory_peak_mb < p.memory_peak_mb
        or q.latency_s < p.latency_s
    )
    return quality_ok and flops_ok and memory_ok and latency_ok and strict


def pareto_frontier(evaluations: list[ModelEvaluation]) -> list[ModelEvaluation]:
    """Filtre les points dominés. Conserve par domaine."""
    frontier: list[ModelEvaluation] = []
    for p in evaluations:
        dominated = any(is_dominated(p, q) for q in evaluations if q is not p)
        if not dominated:
            frontier.append(p)
    return frontier


def asp_on_frontier(
    evaluations: list[ModelEvaluation],
    asp_name: str = "ASP",
) -> dict[str, bool]:
    """Vérifie : ASP est sur la frontière dans chaque domaine évalué."""
    frontier = pareto_frontier(evaluations)
    domains = {e.domain for e in evaluations}
    out: dict[str, bool] = {}
    for d in domains:
        on_frontier = any(p.name == asp_name and p.domain == d for p in frontier)
        out[d] = on_frontier
    return out
