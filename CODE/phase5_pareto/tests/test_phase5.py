"""Tests phase 5 structurels (sur ASPLayer factice consommant l'interface)."""

from __future__ import annotations

import numpy as np
import torch

from phase5_pareto.pareto import ModelEvaluation, asp_on_frontier, is_dominated, pareto_frontier
from phase5_pareto.test_5a_identifiability import (
    run_anti_fraud,
    run_differential_activation,
)
from phase5_pareto.test_5b_elasticity import detect_lag, run_elasticity_test
from phase5_pareto.test_5c_se_ht import compute_se_ht, measure_inference_time, passes_se_ht_targets
from phase5_pareto.test_5e_ood import run_ood_test
from phase5_pareto.test_6c_rmax_half import compute_r_med_oracle, evaluate_rmax_half


class FakeASPLayer:
    """ASPLayer factice : R_target proportionnel au nombre de tokens non-zéro
    (proxy de "non-bruit"). Permet de tester le harnais phase 5 sans vraie
    implémentation.
    """

    R_max = 16

    def __init__(self, *, base_R_target: float = 1.0, sensitivity: float = 8.0) -> None:
        self.base_R_target = base_R_target
        self.sensitivity = sensitivity

    def forward_eval(
        self, tokens: torch.Tensor, query_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = tokens.shape
        non_zero_ratio = (tokens != 0).float().mean(dim=-1, keepdim=True)
        # R_target par token : croît avec le ratio de tokens non-zéro
        R_target = self.base_R_target + self.sensitivity * non_zero_ratio.expand(-1, N)
        # Logits factices
        logits = torch.zeros(B, 10)
        return logits, R_target


# ----------------------------------------------------------------
# Test 5a anti-fraude
# ----------------------------------------------------------------


def test_anti_fraud_passes_on_zero_tokens() -> None:
    layer = FakeASPLayer(base_R_target=0.5, sensitivity=8.0)
    noise = torch.zeros(4, 32, dtype=torch.int64)
    qpos = torch.full((4,), 31)
    result = run_anti_fraud(asp_layer=layer, noise_tokens=noise, query_pos=qpos, floor_threshold=1.0)
    assert result.passed
    assert result.R_target_floor_observed <= 1.0


def test_anti_fraud_fails_when_layer_fires_on_noise() -> None:
    # Couche pathologique : R_target élevé même sur zéros
    class FraudLayer:
        R_max = 16
        def forward_eval(self, tokens: torch.Tensor, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            B, N = tokens.shape
            return torch.zeros(B, 10), torch.full((B, N), 8.0)

    layer = FraudLayer()
    noise = torch.zeros(4, 32, dtype=torch.int64)
    qpos = torch.full((4,), 31)
    result = run_anti_fraud(asp_layer=layer, noise_tokens=noise, query_pos=qpos, floor_threshold=1.0)
    assert not result.passed


# ----------------------------------------------------------------
# Test 5a.ii différentiel
# ----------------------------------------------------------------


def test_differential_activation_smoke() -> None:
    layer = FakeASPLayer(base_R_target=0.5, sensitivity=8.0)
    B, N = 4, 32
    qpos = torch.full((B,), N - 1)
    conditions = {
        "noise": (torch.zeros(B, N, dtype=torch.int64), qpos),
        "null": (torch.zeros(B, N, dtype=torch.int64), qpos),
        "trivial": (torch.full((B, N), 5, dtype=torch.int64), qpos),
        "structured": (torch.randint(1, 20, (B, N), dtype=torch.int64), qpos),
    }
    result = run_differential_activation(
        asp_layer=layer, conditions=conditions,
        threshold_active=0.01,  # seuil bas pour le test
        threshold_silent=0.001,
    )
    # structured a non_zero_ratio = 1, trivial = 1 aussi → similaires
    # noise a non_zero_ratio = 0
    # Donc structured devrait être > noise
    assert result.R_target_per_condition["structured"] > result.R_target_per_condition["noise"]


# ----------------------------------------------------------------
# Test 5b élasticité
# ----------------------------------------------------------------


def test_detect_lag_immediate() -> None:
    R = np.array([0.5, 0.5, 5.0, 5.0, 5.0, 0.5, 0.5])
    # Structure occupe positions 2-4
    lag_up = detect_lag(R, structure_start=2, structure_end=4, threshold=2.0, direction="up")
    assert lag_up == 0  # réponse immédiate


def test_detect_lag_delayed() -> None:
    R = np.array([0.5, 0.5, 0.5, 5.0, 5.0, 0.5, 0.5])
    # Structure annoncée commencer en 2, mais R ne monte qu'en 3
    lag_up = detect_lag(R, structure_start=2, structure_end=4, threshold=2.0, direction="up")
    assert lag_up == 1


# ----------------------------------------------------------------
# Test 5c SE/HT
# ----------------------------------------------------------------


def test_compute_se_ht_basic() -> None:
    result = compute_se_ht(accuracy=0.8, avg_rank=4.0, inference_time_s=0.01)
    assert abs(result.se - 0.2) < 1e-9
    assert abs(result.ht - 80.0) < 1e-9


def test_passes_se_ht_targets_strict() -> None:
    asp = compute_se_ht(accuracy=0.8, avg_rank=2.0, inference_time_s=0.01)         # SE = 0.4
    transformer = compute_se_ht(accuracy=0.8, avg_rank=8.0, inference_time_s=0.05) # SE = 0.1
    ssm = compute_se_ht(accuracy=0.7, avg_rank=4.0, inference_time_s=0.02)         # HT = 35
    verdict = passes_se_ht_targets(
        asp_result=asp, transformer_result=transformer, ssm_result=ssm,
        se_target_factor=2.0,
    )
    assert verdict["se_passed"]   # 0.4 ≥ 2 × 0.1


def test_measure_inference_time_smoke() -> None:
    def fwd():
        torch.randn(64, 64)

    t = measure_inference_time(forward_callable=fwd, n_warmup=2, n_repeat=3, sync_cuda=False)
    assert t >= 0


# ----------------------------------------------------------------
# Test 5e OOD
# ----------------------------------------------------------------


def test_ood_detects_elevation() -> None:
    layer = FakeASPLayer(base_R_target=0.5, sensitivity=8.0)
    # Train axis = beaucoup de zéros (faible R_target)
    train_tokens = torch.zeros(4, 32, dtype=torch.int64)
    train_tokens[:, :8] = torch.randint(1, 10, (4, 8))
    # Eval axis = beaucoup de tokens non-zéro (R_target élevé)
    eval_tokens = torch.randint(1, 20, (4, 32), dtype=torch.int64)
    qpos = torch.full((4,), 31)
    result = run_ood_test(
        asp_layer=layer,
        train_axis_tokens=train_tokens, train_axis_query_pos=qpos,
        eval_axis_tokens=eval_tokens, eval_axis_query_pos=qpos,
        elevation_threshold=1.2,
    )
    assert result.passed
    assert result.eval_axis_higher


# ----------------------------------------------------------------
# Test 6c R_max/2
# ----------------------------------------------------------------


def test_r_med_oracle_global_median() -> None:
    r_eff = {("a",): np.array([1, 2, 3, 4, 5]), ("b",): np.array([10, 11, 12])}
    med = compute_r_med_oracle(r_eff)
    # Médiane de [1,2,3,4,5,10,11,12] = (4+5)/2 = 4.5
    assert abs(med - 4.5) < 1e-9


def test_evaluate_rmax_half_strict() -> None:
    result = evaluate_rmax_half(
        quality_asp=0.96, quality_oracle=1.0, R_max_used=4, r_med_oracle=8.0,
    )
    assert result.verdict == "strict"


def test_evaluate_rmax_half_partial() -> None:
    result = evaluate_rmax_half(
        quality_asp=0.85, quality_oracle=1.0, R_max_used=4, r_med_oracle=8.0,
    )
    assert result.verdict == "partial"


def test_evaluate_rmax_half_fail() -> None:
    result = evaluate_rmax_half(
        quality_asp=0.5, quality_oracle=1.0, R_max_used=4, r_med_oracle=8.0,
    )
    assert result.verdict == "fail"


# ----------------------------------------------------------------
# Pareto
# ----------------------------------------------------------------


def test_is_dominated_basic() -> None:
    a = ModelEvaluation(name="a", domain="seq", quality=0.7, flops_per_token=10, memory_peak_mb=1, latency_s=0.1)
    b = ModelEvaluation(name="b", domain="seq", quality=0.8, flops_per_token=10, memory_peak_mb=1, latency_s=0.1)
    assert is_dominated(a, b)
    assert not is_dominated(b, a)


def test_pareto_frontier_filters_dominated() -> None:
    evals = [
        ModelEvaluation(name="dominated", domain="seq", quality=0.7, flops_per_token=20, memory_peak_mb=2, latency_s=0.1),
        ModelEvaluation(name="champion", domain="seq", quality=0.9, flops_per_token=10, memory_peak_mb=1, latency_s=0.05),
        ModelEvaluation(name="cheap", domain="seq", quality=0.5, flops_per_token=2, memory_peak_mb=0.5, latency_s=0.01),
    ]
    frontier = pareto_frontier(evals)
    names = {p.name for p in frontier}
    assert "champion" in names
    assert "cheap" in names
    assert "dominated" not in names
