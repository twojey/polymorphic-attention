"""
distillation.py — Phase 4a (warm-up avec distillation) avec V3.5 :
percentile p75 + loss asymétrique.

Spec : DOC/04 §3 (4a/4b), spec V3.5 (DOC/00 + project_asp_overview) :
"distillation sur percentile p75, pas la moyenne ; loss asymétrique pénalisant
la sous-allocation plus fortement que la sur-allocation (γ ∈ [0.1, 0.3])".

Cible de distillation : R_target_p75 = p75 du r_eff Oracle pour le régime
correspondant. La cible est pré-calculée à partir des données phase 1/2.

Loss asymétrique :
    L_distill(R_pred, R_target) =
        (1 + γ) · max(R_target - R_pred, 0)²    si R_pred < R_target  (sous-allocation)
        (1 - γ) · max(R_pred - R_target, 0)²    si R_pred > R_target  (sur-allocation)

avec γ ∈ [0.1, 0.3] (pré-enregistré).
"""

from __future__ import annotations

import torch


def asymmetric_distillation_loss(
    R_pred: torch.Tensor,
    R_target: torch.Tensor,
    *,
    gamma: float = 0.2,
) -> torch.Tensor:
    """Loss asymétrique : sous-allocation pénalisée plus fort que sur-allocation.

    R_pred, R_target : (..., ) tensors. γ ∈ [0.1, 0.3].
    """
    assert 0.0 <= gamma <= 1.0, f"gamma={gamma} hors [0, 1]"
    diff = R_pred - R_target
    sub_alloc = (-diff).clamp(min=0)        # > 0 si R_pred < R_target
    over_alloc = diff.clamp(min=0)          # > 0 si R_pred > R_target
    loss = (1 + gamma) * sub_alloc.pow(2) + (1 - gamma) * over_alloc.pow(2)
    return loss.mean()


def compute_p75_targets(
    r_eff_per_regime: dict[tuple, torch.Tensor],
) -> dict[tuple, float]:
    """Pour chaque régime, calcule le p75 du r_eff Oracle. Cible de
    distillation phase 4a (V3.5).
    """
    out: dict[tuple, float] = {}
    for regime, vals in r_eff_per_regime.items():
        if vals.numel() == 0:
            out[regime] = 0.0
            continue
        out[regime] = float(torch.quantile(vals.to(torch.float64), 0.75).item())
    return out


class TransitionMonitor:
    """Vérifie le critère de transition phase 4a → 4b (DOC/04 §3.4).

    Critère pré-enregistré V1 :
    - convergence de L_distillation : décroissance < tolerance sur N étapes
    - corrélation Spearman R_target ↔ r_target_théorique > 0.80
    - absence de plafond artificiel (R_pred n'est pas saturé sur 1 valeur)
    """

    def __init__(
        self,
        *,
        loss_window: int = 50,
        loss_tolerance: float = 1e-3,
        rho_threshold: float = 0.80,
        plateau_var_min: float = 0.5,
    ) -> None:
        self.loss_window = loss_window
        self.loss_tolerance = loss_tolerance
        self.rho_threshold = rho_threshold
        self.plateau_var_min = plateau_var_min
        self.loss_history: list[float] = []

    def update(self, loss: float) -> None:
        self.loss_history.append(loss)

    def check_transition(
        self,
        *,
        R_pred_recent: torch.Tensor,
        R_target_recent: torch.Tensor,
    ) -> tuple[bool, dict[str, float]]:
        diagnostics: dict[str, float] = {}
        # 1. Convergence loss
        if len(self.loss_history) < self.loss_window:
            return False, {"reason": "not_enough_history"}  # type: ignore[dict-item]
        recent = self.loss_history[-self.loss_window :]
        loss_change = abs(recent[-1] - recent[0]) / max(abs(recent[0]), 1e-9)
        diagnostics["loss_change"] = loss_change
        loss_converged = loss_change < self.loss_tolerance

        # 2. Spearman ρ
        from scipy import stats

        rho = float(stats.spearmanr(
            R_pred_recent.cpu().numpy().reshape(-1),
            R_target_recent.cpu().numpy().reshape(-1),
        ).statistic)
        diagnostics["rho"] = rho
        rho_ok = rho > self.rho_threshold

        # 3. Variance non plate (pas de plafond artificiel)
        var = float(R_pred_recent.var().item())
        diagnostics["pred_var"] = var
        var_ok = var > self.plateau_var_min

        passed = loss_converged and rho_ok and var_ok
        diagnostics["loss_converged"] = float(loss_converged)
        diagnostics["rho_ok"] = float(rho_ok)
        diagnostics["var_ok"] = float(var_ok)
        return passed, diagnostics
