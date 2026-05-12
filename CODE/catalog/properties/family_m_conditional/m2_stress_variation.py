"""
m2_stress_variation.py — Property M2 : variation de A vs (ω, Δ, ℋ).

Spec : DOC/CATALOGUE §M2 "corrélations Spearman globales".

Scope cross_regime : reçoit dict {regime_key → AttentionDump}.

Pour chaque régime, calcule plusieurs summary statistics (par défaut :
r_eff médian, entropy spectrale, sparse fraction). Puis corrèle ces
résumés avec les paramètres de stress (ω, Δ, ℋ) via Spearman.

Sortie : un rho_spearman_{stat}_{stress} pour chaque combinaison utile.
"""

from __future__ import annotations

from typing import Any

import torch

from catalog.oracles.base import AttentionDump
from catalog.properties.base import Property, PropertyContext
from catalog.properties.registry import register_property


def _spearman_rho(x: torch.Tensor, y: torch.Tensor) -> float:
    """Coefficient de corrélation de Spearman entre deux séries 1D."""
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    # Ranks (1-indexed)
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    n = float(x.numel())
    rx_centered = rx - rx.mean()
    ry_centered = ry - ry.mean()
    num = (rx_centered * ry_centered).sum()
    den = (rx_centered.pow(2).sum() * ry_centered.pow(2).sum()).sqrt()
    if den.item() < 1e-30:
        return float("nan")
    return float((num / den).item())


@register_property
class M2StressVariation(Property):
    """M2 — Spearman summary stats × (ω, Δ, ℋ)."""

    name = "M2_stress_variation"
    family = "M"
    cost_class = 3
    requires_fp64 = False
    scope = "cross_regime"

    def __init__(self, theta_cumulative: float = 0.99, eps_floor: float = 1e-30) -> None:
        self.theta = theta_cumulative
        self.eps_floor = eps_floor

    def _r_eff_median(self, A: torch.Tensor, ctx: PropertyContext) -> float:
        A = A.to(device=ctx.device, dtype=ctx.dtype)
        s = torch.linalg.svdvals(A)
        s2 = s.pow(2)
        cumsum = s2.cumsum(dim=-1)
        total = cumsum[..., -1:].clamp_min(self.eps_floor)
        ratio = cumsum / total
        r_eff = (ratio >= self.theta).float().argmax(dim=-1) + 1
        return float(r_eff.float().median().item())

    def _spectral_entropy_norm(self, A: torch.Tensor, ctx: PropertyContext) -> float:
        A = A.to(device=ctx.device, dtype=ctx.dtype)
        s = torch.linalg.svdvals(A)
        s2 = s.pow(2)
        total = s2.sum(dim=-1, keepdim=True).clamp_min(self.eps_floor)
        p = s2 / total
        p_safe = p.clamp_min(self.eps_floor)
        H = -(p_safe * p_safe.log()).sum(dim=-1)
        K = s.shape[-1]
        return float((H / torch.tensor(K, dtype=A.dtype).log()).float().median().item())

    def _sparse_fraction(self, A: torch.Tensor, ctx: PropertyContext) -> float:
        A = A.to(device=ctx.device, dtype=ctx.dtype).abs()
        flat = A.flatten(start_dim=-2)
        max_per_mat = flat.max(dim=-1, keepdim=True).values.clamp_min(self.eps_floor)
        return float(((flat < 0.05 * max_per_mat).float().mean(dim=-1)).float().median().item())

    def compute(
        self, dumps: dict[Any, AttentionDump], ctx: PropertyContext
    ) -> dict[str, float | int | str | bool]:
        if not isinstance(dumps, dict) or not dumps:
            raise ValueError("M2 requiert un dict non-vide {regime_key → AttentionDump}")

        # Pour chaque régime, calcul des summary stats sur la layer médiane
        omegas: list[float] = []
        deltas: list[float] = []
        entropies: list[float] = []
        r_eff_vals: list[float] = []
        H_spec_vals: list[float] = []
        sparse_vals: list[float] = []

        for regime_key, dump in dumps.items():
            # Concat toutes les couches en un mega-batch pour stats globales
            attn_all = torch.cat([A for A in dump.attn], dim=0)  # ((L·B), H, N, N)
            r_eff_vals.append(self._r_eff_median(attn_all, ctx))
            H_spec_vals.append(self._spectral_entropy_norm(attn_all, ctx))
            sparse_vals.append(self._sparse_fraction(attn_all, ctx))

            # Stress params depuis metadata du dump
            omega = dump.metadata.get("regime", regime_key)
            # On regarde aussi omegas tensor du dump (par-exemple)
            if dump.omegas.numel() > 0:
                omegas.append(float(dump.omegas.float().mean().item()))
            else:
                omegas.append(0.0)
            if dump.deltas.numel() > 0:
                deltas.append(float(dump.deltas.float().mean().item()))
            else:
                deltas.append(0.0)
            if dump.entropies.numel() > 0:
                entropies.append(float(dump.entropies.float().mean().item()))
            else:
                entropies.append(0.0)

        # Spearman pour chaque (summary stat, stress param)
        x_omega = torch.tensor(omegas)
        x_delta = torch.tensor(deltas)
        x_entropy = torch.tensor(entropies)
        results: dict[str, float | int | str | bool] = {
            "n_regimes": len(dumps),
        }
        for stat_name, stat_vals in [
            ("r_eff", r_eff_vals),
            ("H_spec_norm", H_spec_vals),
            ("sparse_frac", sparse_vals),
        ]:
            y = torch.tensor(stat_vals)
            results[f"rho_spearman_{stat_name}_vs_omega"] = _spearman_rho(x_omega, y)
            results[f"rho_spearman_{stat_name}_vs_delta"] = _spearman_rho(x_delta, y)
            results[f"rho_spearman_{stat_name}_vs_entropy"] = _spearman_rho(x_entropy, y)

        # Range / variance des stats
        results["r_eff_range_across_regimes"] = max(r_eff_vals) - min(r_eff_vals)
        results["H_spec_range_across_regimes"] = max(H_spec_vals) - min(H_spec_vals)
        results["sparse_frac_range_across_regimes"] = (
            max(sparse_vals) - min(sparse_vals)
        )
        return results
