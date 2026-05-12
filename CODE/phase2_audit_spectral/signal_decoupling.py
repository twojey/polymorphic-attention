"""
signal_decoupling.py — diagnostic Spearman S_Spectral ↔ r_eff (DOC/carnet 2026-05-12).

Garde-fou méthodologique pour phase 2 : si le signal local S_Spectral (fenêtre
glissante K=64, mesuré en phase 1.5) est découplé du vrai rang effectif (SVD
batchée FP64 full, mesuré en phase 2), alors le verdict GO de phase 1.5 sur
S_Spectral mesurait un artefact statistique du choix de K, pas le rang
structurel. Conséquence : H2 (allocation guidée par signal local) est
invalidée même si SCH (H3-H4) est validée.

Le diagnostic doit tourner :
- sur les **mêmes** matrices d'attention que la SVD batchée (cohérence
  numérique), donc on réutilise les dumps déjà chargés ;
- avec un **sous-échantillonnage stratifié** par régime (ω, Δ, ℋ) pour
  contenir le coût (compute_s_spectral coûte O(B·L·H·N·K²log K) ; sur le
  set audit complet ce serait ~5h CPU, c'est inutile pour un diagnostic) ;
- en produisant Spearman global + par axe avec IC bootstrap.

Sortie :
- `rho_global` ± IC95 (verdict principal) ;
- `rho_per_axis` pour ω, Δ, ℋ ;
- `verdict` : "ok" si ρ_global > seuil_decouplage, "decoupled" sinon.

Le seuil de découplage par défaut est 0.60 (cf. DOC/carnet 2026-05-12 §
"Combinaison cauchemar"). Si ρ < 0.60 → drapeau rouge majeur, on doit revoir
l'argument H2 avant phase 3.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from phase1b_calibration_signal.bench.spearman import (
    SpearmanResult,
    bootstrap_spearman_ci,
)
from phase1b_calibration_signal.signals.s_spectral import compute_s_spectral


@dataclass
class DecouplingDiagnostic:
    """Résultat du diagnostic découplage S_Spectral / r_eff."""
    rho_global: SpearmanResult
    rho_per_axis: dict[str, SpearmanResult] = field(default_factory=dict)
    n_examples_used: int = 0
    K: int = 64
    tau: float = 1e-3
    threshold_decoupling: float = 0.60
    verdict: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "rho_global": self.rho_global.rho,
            "rho_global_ci_low": self.rho_global.ci_low,
            "rho_global_ci_high": self.rho_global.ci_high,
            "rho_per_axis": {
                k: {"rho": v.rho, "ci_low": v.ci_low, "ci_high": v.ci_high}
                for k, v in self.rho_per_axis.items()
            },
            "n_examples_used": self.n_examples_used,
            "K": self.K, "tau": self.tau,
            "threshold_decoupling": self.threshold_decoupling,
            "verdict": self.verdict,
        }


def stratified_subsample_indices(
    *,
    omegas: np.ndarray,
    deltas: np.ndarray,
    entropies: np.ndarray,
    max_total: int,
    seed: int = 0,
) -> np.ndarray:
    """Sous-échantillonnage stratifié par régime (ω, Δ, ℋ).

    Garantit que chaque régime distinct contribue ⌊max_total / n_regimes⌋
    exemples (au moins 1, si le régime est représenté). Le reste est tiré
    uniformément. Préserve la diversité cross-régime du diagnostic.

    Retourne : array d'indices triés, taille ≤ max_total.
    """
    rng = np.random.default_rng(seed)
    keys = list(zip(omegas.tolist(), deltas.tolist(), entropies.tolist(), strict=True))
    # Buckets de régime
    buckets: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)

    n_regimes = len(buckets)
    if n_regimes == 0:
        return np.array([], dtype=np.int64)

    per_regime = max(1, max_total // n_regimes)
    selected: list[int] = []
    for regime_indices in buckets.values():
        if len(regime_indices) <= per_regime:
            selected.extend(regime_indices)
        else:
            chosen = rng.choice(regime_indices, size=per_regime, replace=False)
            selected.extend(chosen.tolist())

    # Si on a du quota restant, tirer uniformément dans les régimes encore non
    # saturés.
    remaining_budget = max_total - len(selected)
    if remaining_budget > 0:
        selected_set = set(selected)
        unused = [i for i in range(len(keys)) if i not in selected_set]
        if unused:
            extra = rng.choice(
                unused,
                size=min(remaining_budget, len(unused)),
                replace=False,
            )
            selected.extend(extra.tolist())

    return np.array(sorted(selected), dtype=np.int64)


def _aggregate_s_spectral_per_example(
    sspec: torch.Tensor,
    *,
    K: int,
) -> np.ndarray:
    """Agrège (L, B, H, N) → (B,) par moyenne sur (L, H, tokens valides).

    Tokens valides = t ≥ K-1 (warmup zone exclue, où r_eff est sous-estimé par
    construction — cf. signals/s_spectral.py).
    """
    L, B, H, N = sspec.shape
    if N <= K - 1:
        # Pas assez de tokens pour exclure le warmup → on prend tout (rare).
        return sspec.mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)
    valid = sspec[..., K - 1:]  # (L, B, H, N - K + 1)
    return valid.mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float64)


def diagnose_s_spectral_decoupling(
    *,
    dumps: list[dict],
    r_eff_per_layer_head_example: np.ndarray,   # (L, B, H) global, B = total examples
    K: int = 64,
    tau: float = 1e-3,
    max_examples: int = 200,
    seed: int = 0,
    n_boot: int = 1000,
    threshold_decoupling: float = 0.60,
) -> DecouplingDiagnostic:
    """Calcule ρ_Spearman(r_eff_full, S_Spectral_K) par exemple.

    Pipeline :
    1. Concatène tous les axes scalaires des dumps (ω, Δ, ℋ).
    2. Sous-échantillonnage stratifié → indices à inclure.
    3. Pour chaque dump, calcule S_Spectral sur les exemples sélectionnés
       de ce dump (regroupés par seq_len) puis agrège par exemple.
    4. Compute r_eff_per_example = moyenne sur (L, H) du tenseur global.
    5. Spearman global + par axe avec bootstrap IC95 (réutilise phase 1.5).

    Args
    ----
    dumps : liste des dumps phase 1 V2 (un par bucket seq_len).
    r_eff_per_layer_head_example : tenseur (L, B, H) déjà calculé par le
        pipeline SVD batché du driver phase 2.
    K, tau : paramètres S_Spectral (figés à ceux de la phase 1.5).
    max_examples : plafond global du sous-échantillon stratifié.
    seed, n_boot : bootstrap IC95.
    threshold_decoupling : seuil de verdict.

    Returns
    -------
    DecouplingDiagnostic avec rho_global, rho_per_axis, verdict.
    """
    # Concaténation des axes scalaires (ordre cohérent avec le driver phase 2,
    # qui concatène dumps dans l'ordre de chargement).
    omegas = np.concatenate([d["omegas"].cpu().numpy() for d in dumps])
    deltas = np.concatenate([d["deltas"].cpu().numpy() for d in dumps])
    entropies = np.concatenate([d["entropies"].cpu().numpy() for d in dumps])
    B_total = len(omegas)
    if r_eff_per_layer_head_example.shape[1] != B_total:
        raise ValueError(
            f"r_eff a {r_eff_per_layer_head_example.shape[1]} exemples, dumps "
            f"en ont {B_total}. Mismatch concaténation/calcul."
        )

    selected = stratified_subsample_indices(
        omegas=omegas, deltas=deltas, entropies=entropies,
        max_total=max_examples, seed=seed,
    )
    if selected.size < 4:
        # Bootstrap requiert au moins 4 points pour être informatif.
        raise ValueError(
            f"Sous-échantillonnage trop petit ({selected.size}) ; "
            f"max_examples={max_examples} insuffisant."
        )

    # Mapping (index global) → (index dump local) pour calculer S_Spectral
    # seulement sur les exemples sélectionnés de chaque dump.
    s_spectral_per_example = np.zeros(selected.size, dtype=np.float64)
    selected_set = set(selected.tolist())
    selected_to_position = {idx: pos for pos, idx in enumerate(selected.tolist())}

    b_offset = 0
    for dump_idx, d in enumerate(dumps):
        B_d = d["attn"][0].size(0)
        # indices globaux du dump
        global_indices = range(b_offset, b_offset + B_d)
        # indices locaux du dump (sous-ensemble sélectionné)
        local_indices = [g - b_offset for g in global_indices if g in selected_set]
        if not local_indices:
            b_offset += B_d
            continue

        # Construire l'attention sous-échantillonnée
        attn_subset = [
            d["attn"][ell][local_indices].contiguous()
            for ell in range(len(d["attn"]))
        ]
        # compute_s_spectral exige les axes BLAS=1 ; la garde y est dans le module.
        try:
            sspec = compute_s_spectral(attn_subset, K=K, tau=tau)   # (L, B_sub, H, N)
        except RuntimeError as exc:
            # BLAS multi-thread → propage clairement vers stderr puis ré-élève.
            print(
                f"[diagnostic_decoupling] compute_s_spectral a échoué sur dump "
                f"{dump_idx} (seq_len={d['attn'][0].size(2)}) : {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise

        sspec_per_ex = _aggregate_s_spectral_per_example(sspec, K=K)
        # Placer aux positions correspondantes du vecteur global.
        for local_pos, local_idx in enumerate(local_indices):
            global_idx = b_offset + local_idx
            s_spectral_per_example[selected_to_position[global_idx]] = sspec_per_ex[local_pos]

        b_offset += B_d

    # r_eff agrégé par exemple : moyenne sur (L, H).
    L, _, H = r_eff_per_layer_head_example.shape
    r_eff_per_example_all = r_eff_per_layer_head_example.astype(np.float64).mean(axis=(0, 2))  # (B,)
    r_eff_per_example = r_eff_per_example_all[selected]

    # Spearman global
    rho_global = bootstrap_spearman_ci(
        r_eff_per_example, s_spectral_per_example, n_boot=n_boot, seed=seed,
    )

    # Spearman par axe : on rapporte Spearman(signal, axis) et Spearman(r_eff,
    # axis) pour permettre une analyse fine du découplage. Les axes constants
    # (un seul ω/Δ/ℋ sur le sous-échantillon) sont sautés explicitement —
    # sinon scipy retourne NaN avec un ConstantInputWarning sans dire au caller
    # ce qui s'est passé.
    omegas_sel = omegas[selected].astype(np.float64)
    deltas_sel = deltas[selected].astype(np.float64)
    entropies_sel = entropies[selected].astype(np.float64)
    rho_per_axis: dict[str, SpearmanResult] = {}
    for axis_name, axis_vals in (("omega", omegas_sel), ("delta", deltas_sel), ("entropy", entropies_sel)):
        if np.unique(axis_vals).size < 2:
            # Axe constant sur ce sous-échantillon → corrélation non définie ;
            # on log un avertissement clair plutôt que NaN silencieux.
            print(
                f"[diagnostic_decoupling] axe '{axis_name}' constant sur le "
                f"sous-échantillon ({axis_vals[0] if axis_vals.size else 'vide'}) "
                f"→ Spearman non calculée pour cet axe.",
                file=sys.stderr,
                flush=True,
            )
            continue
        rho_per_axis[f"s_spectral_vs_{axis_name}"] = bootstrap_spearman_ci(
            s_spectral_per_example, axis_vals, n_boot=n_boot, seed=seed + 1,
        )
        rho_per_axis[f"r_eff_vs_{axis_name}"] = bootstrap_spearman_ci(
            r_eff_per_example, axis_vals, n_boot=n_boot, seed=seed + 2,
        )

    verdict = "ok" if rho_global.rho >= threshold_decoupling else "decoupled"
    return DecouplingDiagnostic(
        rho_global=rho_global,
        rho_per_axis=rho_per_axis,
        n_examples_used=int(selected.size),
        K=K, tau=tau,
        threshold_decoupling=threshold_decoupling,
        verdict=verdict,
    )


def log_diagnostic_to_mlflow(diag: DecouplingDiagnostic, *, mlflow_module) -> None:
    """Envoie les métriques principales du diagnostic à MLflow.

    Séparé du calcul pour permettre tests sans MLflow actif.
    `mlflow_module` est `mlflow` (importé par le caller).
    """
    mlflow_module.log_metric("decoupling_rho_global", float(diag.rho_global.rho))
    mlflow_module.log_metric("decoupling_rho_global_ci_low", float(diag.rho_global.ci_low))
    mlflow_module.log_metric("decoupling_rho_global_ci_high", float(diag.rho_global.ci_high))
    mlflow_module.log_metric("decoupling_n_examples", int(diag.n_examples_used))
    mlflow_module.log_metric("decoupling_verdict_ok", 1.0 if diag.verdict == "ok" else 0.0)
    for axis_metric, res in diag.rho_per_axis.items():
        mlflow_module.log_metric(f"decoupling_{axis_metric}_rho", float(res.rho))
