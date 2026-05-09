"""Tests phase 4 structurel."""

from __future__ import annotations

import numpy as np
import torch

from phase4_routage_budget.curriculum import CurriculumScheduler, default_curriculum
from phase4_routage_budget.diagram_phase import (
    ParetoPoint, build_pareto_curve, build_phase_diagram, is_phase_diagram_increasing,
)
from phase4_routage_budget.distillation import (
    TransitionMonitor, asymmetric_distillation_loss, compute_p75_targets,
)
from phase4_routage_budget.sparsity_loss import loss_sparsity, sparsity_weights
from phase4_routage_budget.spectrometer import (
    FrozenAlphaSpectrometer, Spectrometer, SpectrometerConfig,
)


# ----------------------------------------------------------------
# Spectromètre
# ----------------------------------------------------------------


def test_spectrometer_output_alpha_range() -> None:
    cfg = SpectrometerConfig(input_dim=12)
    spec = Spectrometer(cfg)
    signals = torch.randn(2, 8, 12)
    alpha = spec(signals)
    assert alpha.shape == (2, 8)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_frozen_alpha_returns_ones() -> None:
    spec = FrozenAlphaSpectrometer()
    signals = torch.randn(3, 5, 10)
    out = spec(signals)
    assert torch.allclose(out, torch.ones(3, 5))


# ----------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------


def test_curriculum_starts_at_easy() -> None:
    sch = CurriculumScheduler(default_curriculum())
    assert sch.current_stage().name == "easy"


def test_curriculum_advances_when_acc_high_enough() -> None:
    sch = CurriculumScheduler(default_curriculum())
    for _ in range(300):
        sch.step(val_acc=0.99)  # acc très haut → transition
    # Devrait être en intermediate ou hard maintenant
    assert sch.current_stage().name != "easy"


def test_curriculum_does_not_advance_with_low_acc() -> None:
    sch = CurriculumScheduler(default_curriculum())
    for _ in range(500):
        sch.step(val_acc=0.5)  # acc bas
    assert sch.current_stage().name == "easy"


def test_curriculum_does_not_advance_before_min_steps() -> None:
    sch = CurriculumScheduler(default_curriculum())
    for _ in range(50):
        sch.step(val_acc=0.99)
    assert sch.current_stage().name == "easy"


# ----------------------------------------------------------------
# Distillation
# ----------------------------------------------------------------


def test_asymmetric_loss_sub_alloc_penalized_more() -> None:
    # R_pred < R_target → sous-allocation
    R_pred = torch.tensor([1.0])
    R_target = torch.tensor([3.0])
    loss_sub = asymmetric_distillation_loss(R_pred, R_target, gamma=0.2)
    # R_pred > R_target → sur-allocation, même écart absolu
    R_pred_over = torch.tensor([5.0])
    R_target_low = torch.tensor([3.0])
    loss_over = asymmetric_distillation_loss(R_pred_over, R_target_low, gamma=0.2)
    # Sous-allocation doit coûter plus cher
    assert loss_sub > loss_over


def test_asymmetric_loss_zero_when_exact() -> None:
    R = torch.tensor([3.0, 5.0, 7.0])
    loss = asymmetric_distillation_loss(R, R, gamma=0.2)
    assert loss.item() < 1e-12


def test_compute_p75_targets() -> None:
    r_eff = {
        ("regime_a",): torch.tensor([1.0, 2.0, 3.0, 4.0]),  # p75 = 3.25
        ("regime_b",): torch.tensor([10.0, 10.0]),          # p75 = 10
    }
    targets = compute_p75_targets(r_eff)
    assert abs(targets[("regime_a",)] - 3.25) < 0.1
    assert abs(targets[("regime_b",)] - 10.0) < 1e-6


def test_transition_monitor_no_history() -> None:
    mon = TransitionMonitor(loss_window=10)
    passed, diags = mon.check_transition(
        R_pred_recent=torch.randn(50), R_target_recent=torch.randn(50),
    )
    assert not passed


def test_transition_monitor_passes_when_converged_and_correlated() -> None:
    mon = TransitionMonitor(loss_window=10, loss_tolerance=0.1, rho_threshold=0.5,
                             plateau_var_min=0.5)
    # Loss converge
    for _ in range(20):
        mon.update(0.5)
    # R_pred ≈ R_target avec variance suffisante
    rng = torch.Generator().manual_seed(0)
    R_target = torch.randn(100, generator=rng) * 2.0
    R_pred = R_target + torch.randn(100, generator=rng) * 0.1
    passed, diags = mon.check_transition(R_pred_recent=R_pred, R_target_recent=R_target)
    assert passed


# ----------------------------------------------------------------
# Sparsity
# ----------------------------------------------------------------


def test_sparsity_weights_linear() -> None:
    w = sparsity_weights(R_max=4, strategy="linear")
    assert torch.allclose(w, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_loss_sparsity_zero_when_mask_zero() -> None:
    mask = torch.zeros(2, 5, 4)
    w = sparsity_weights(R_max=4, strategy="linear")
    assert loss_sparsity(mask=mask, weights=w).item() == 0


def test_loss_sparsity_full_mask_max() -> None:
    mask = torch.ones(2, 5, 4)
    w = sparsity_weights(R_max=4, strategy="linear")
    loss = loss_sparsity(mask=mask, weights=w)
    # Σ w_i = 1+2+3+4 = 10
    assert abs(loss.item() - 10.0) < 1e-6


# ----------------------------------------------------------------
# Diagramme de Phase + Pareto
# ----------------------------------------------------------------


def test_phase_diagram_increasing_with_omega() -> None:
    rng = np.random.default_rng(0)
    n = 200
    omega = rng.choice([1, 2, 4, 8], size=n).astype(float)
    delta = np.zeros(n)
    entropy = np.zeros(n)
    R_target = omega + rng.normal(0, 0.5, size=n)
    diagram = build_phase_diagram(
        R_target=R_target, omega=omega, delta=delta, entropy=entropy,
    )
    assert is_phase_diagram_increasing(diagram, axis="omega")


def test_pareto_curve_keeps_non_dominated() -> None:
    points = [
        ParetoPoint(lambda_budget=0.1, quality=0.95, avg_rank=8.0),
        ParetoPoint(lambda_budget=0.5, quality=0.90, avg_rank=4.0),
        ParetoPoint(lambda_budget=1.0, quality=0.80, avg_rank=2.0),
        ParetoPoint(lambda_budget=2.0, quality=0.70, avg_rank=1.0),
        ParetoPoint(lambda_budget=10.0, quality=0.50, avg_rank=0.5),
        # Point dominé : qualité 0.85 mais rang 6 (dominé par (0.90, 4.0))
        ParetoPoint(lambda_budget=0.3, quality=0.85, avg_rank=6.0),
    ]
    pareto = build_pareto_curve(points)
    qualities = [p.quality for p in pareto]
    # Le point dominé (0.85, 6.0) ne doit pas être dans la frontière
    assert 0.85 not in qualities
    # La frontière croît en qualité avec le rang
    sorted_pareto = sorted(pareto, key=lambda p: p.avg_rank)
    qs = [p.quality for p in sorted_pareto]
    assert qs == sorted(qs)
