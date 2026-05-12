"""Tests pour les properties V2 finales : F1, M1, G6-G8, O4-O5, P3-P6, Q4-Q5, R3, S3, V1, W3, K2, C6, N1-N3."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_c_token_stats.c6_s_grad import C6SGrad
from catalog.properties.family_f_dynamic.f1_lipschitz import F1Lipschitz
from catalog.properties.family_g_algebraic.g6_bernstein_sato_proxy import (
    G6BernsteinSatoProxy,
)
from catalog.properties.family_g_algebraic.g7_d_module_proxy import G7DModuleProxy
from catalog.properties.family_g_algebraic.g8_syzygies_proxy import G8SyzygiesProxy
from catalog.properties.family_k_graph.k2_persistent_homology import K2PersistentHomology
from catalog.properties.family_m_conditional.m1_token_type_sensitivity import (
    M1TokenTypeSensitivity,
)
from catalog.properties.family_n_comparative.n1_f_divergence import N1FDivergence
from catalog.properties.family_n_comparative.n2_preservation import N2Preservation
from catalog.properties.family_n_comparative.n3_lipschitz_diff import N3LipschitzDiff
from catalog.properties.family_o_displacement.o4_kailath_generators import (
    O4KailathGenerators,
)
from catalog.properties.family_o_displacement.o5_pan_recurrence import O5PanRecurrence
from catalog.properties.family_p_realization.p3_hsv_decay import P3HSVDecay
from catalog.properties.family_p_realization.p4_minimal_order import P4MinimalOrder
from catalog.properties.family_p_realization.p5_hierarchical_blocks import (
    P5HierarchicalBlocks,
)
from catalog.properties.family_p_realization.p6_gramians import P6Gramians
from catalog.properties.family_q_hierarchical.q4_nestedness_extended import (
    Q4NestednessExtended,
)
from catalog.properties.family_q_hierarchical.q5_hmatrix_distribution import (
    Q5HMatrixDistribution,
)
from catalog.properties.family_r_rkhs.r3_bochner_stationarity import (
    R3BochnerStationarity,
)
from catalog.properties.family_s_tensors.s3_hierarchical_tucker import (
    S3HierarchicalTucker,
)
from catalog.properties.family_v_operators.v1_pseudodifferential import (
    V1Pseudodifferential,
)
from catalog.properties.family_w_logic.w3_nip_dependence import W3NIPDependence
from catalog.properties.registry import REGISTRY


def _make_A(seed: int = 0, N: int = 16, B: int = 4, H: int = 2) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(B, H, N, N, dtype=torch.float64), dim=-1)


def _ctx(**meta) -> PropertyContext:
    return PropertyContext(device="cpu", dtype=torch.float64, metadata=meta)


# F1 Lipschitz
def test_f1_with_tokens() -> None:
    A = _make_A()
    tokens = torch.randint(0, 8, (4, 16))
    out = F1Lipschitz().compute(A, _ctx(tokens=tokens))
    assert "lipschitz_max" in out
    assert out["input_kind"] == "hamming"


def test_f1_without_tokens() -> None:
    A = _make_A()
    out = F1Lipschitz().compute(A, _ctx())
    assert "lipschitz_max" in out or "skip_reason" in out


def test_f1_batch_too_small() -> None:
    A = torch.softmax(torch.randn(1, 2, 8, 8, dtype=torch.float64), dim=-1)
    out = F1Lipschitz().compute(A, _ctx(tokens=torch.zeros(1, 8, dtype=torch.long)))
    assert out["skip_reason"] == "batch<2"


def test_f1_registered() -> None:
    assert REGISTRY.get("F1_lipschitz") is F1Lipschitz


# M1 Token-type sensitivity
def test_m1_basic() -> None:
    A = _make_A()
    tokens = torch.randint(0, 4, (4, 16))
    out = M1TokenTypeSensitivity().compute(A, _ctx(tokens=tokens))
    assert "query_sensitivity_std" in out or "skip_reason" in out


def test_m1_no_tokens() -> None:
    A = _make_A()
    out = M1TokenTypeSensitivity().compute(A, _ctx())
    assert "skip_reason" in out


def test_m1_registered() -> None:
    assert REGISTRY.get("M1_token_type_sensitivity") is M1TokenTypeSensitivity


# G6 Bernstein-Sato
def test_g6_basic() -> None:
    out = G6BernsteinSatoProxy().compute(_make_A(N=12), _ctx())
    assert "bs_complexity_ratio_median" in out


def test_g6_registered() -> None:
    assert REGISTRY.get("G6_bernstein_sato_proxy") is G6BernsteinSatoProxy


# G7 D-module
def test_g7_basic() -> None:
    out = G7DModuleProxy().compute(_make_A(), _ctx())
    assert "d_module_decay_alpha" in out
    assert out["d_module_class"] in ("holonomic_low_order", "regular", "irregular_frontier")


def test_g7_registered() -> None:
    assert REGISTRY.get("G7_d_module_proxy") is G7DModuleProxy


# G8 Syzygies
def test_g8_basic() -> None:
    out = G8SyzygiesProxy().compute(_make_A(), _ctx())
    assert "syzygies_nullity_median" in out
    assert 0.0 <= out["fraction_full_rank"] <= 1.0


def test_g8_rank1_high_nullity() -> None:
    """Rank-1 matrix : nullity = N-1."""
    N = 8
    u = torch.randn(N, 1, dtype=torch.float64)
    A_rank1 = u @ u.T
    A_rank1 = A_rank1 / A_rank1.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    A = A_rank1.unsqueeze(0).unsqueeze(0)
    out = G8SyzygiesProxy().compute(A, _ctx())
    assert out["syzygies_nullity_median"] >= N - 2  # tolerance


def test_g8_registered() -> None:
    assert REGISTRY.get("G8_syzygies_proxy") is G8SyzygiesProxy


# O4 Kailath generators
def test_o4_basic() -> None:
    out = O4KailathGenerators().compute(_make_A(), _ctx())
    assert "epsilon_r2_median" in out
    assert "min_r_for_eps_0p01_median" in out


def test_o4_registered() -> None:
    assert REGISTRY.get("O4_kailath_generators") is O4KailathGenerators


# O5 Pan recurrence
def test_o5_basic() -> None:
    out = O5PanRecurrence().compute(_make_A(), _ctx())
    assert "pan_fft_lr_eps_median" in out
    assert "rank_E_residual_median" in out


def test_o5_registered() -> None:
    assert REGISTRY.get("O5_pan_recurrence") is O5PanRecurrence


# P3 HSV decay
def test_p3_basic() -> None:
    out = P3HSVDecay().compute(_make_A(), _ctx())
    assert "hsv_decay_slope_exp_median" in out
    assert "fraction_exp_dominant" in out


def test_p3_registered() -> None:
    assert REGISTRY.get("P3_hsv_decay") is P3HSVDecay


# P4 Minimal order
def test_p4_basic() -> None:
    out = P4MinimalOrder().compute(_make_A(), _ctx())
    assert "minimal_order_aic_median" in out
    assert "minimal_order_bic_median" in out


def test_p4_registered() -> None:
    assert REGISTRY.get("P4_minimal_order") is P4MinimalOrder


# P5 Hierarchical blocks
def test_p5_basic() -> None:
    out = P5HierarchicalBlocks().compute(_make_A(N=16), _ctx())
    assert "local_rank_p2_median" in out or "skip_reason" in out


def test_p5_registered() -> None:
    assert REGISTRY.get("P5_hierarchical_blocks") is P5HierarchicalBlocks


# P6 Gramians
def test_p6_basic() -> None:
    out = P6Gramians().compute(_make_A(N=12), _ctx())
    assert "log10_kappa_controllability_median" in out
    assert "hsv_proxy_top_median" in out


def test_p6_registered() -> None:
    assert REGISTRY.get("P6_gramians") is P6Gramians


# Q4 Nestedness extended
def test_q4_basic() -> None:
    out = Q4NestednessExtended().compute(_make_A(N=16), _ctx())
    assert "n_matrices" in out
    # Au moins une métrique nestedness OU skip
    has_metric = any(k.startswith("avg_rank_level") for k in out)
    assert has_metric or "skip_reason" in out


def test_q4_registered() -> None:
    assert REGISTRY.get("Q4_nestedness_extended") is Q4NestednessExtended


# Q5 H-matrix distribution
def test_q5_basic() -> None:
    out = Q5HMatrixDistribution().compute(_make_A(N=16), _ctx())
    assert "hmatrix_rank_max" in out or "skip_reason" in out


def test_q5_registered() -> None:
    assert REGISTRY.get("Q5_hmatrix_distribution") is Q5HMatrixDistribution


# R3 Bochner
def test_r3_basic() -> None:
    out = R3BochnerStationarity().compute(_make_A(), _ctx())
    assert "bochner_recon_err_median" in out
    assert "fraction_pass_bochner" in out


def test_r3_toeplitz_passes() -> None:
    """Vraie Toeplitz : erreur recon ≈ 0."""
    N = 12
    c = torch.randn(N, dtype=torch.float64)
    rows = torch.arange(N).unsqueeze(0)
    cols = torch.arange(N).unsqueeze(1)
    idx = (cols - rows) % N
    A_toep = c[idx]
    A_toep = torch.softmax(A_toep, dim=-1)
    A = A_toep.unsqueeze(0).unsqueeze(0)
    out = R3BochnerStationarity().compute(A, _ctx())
    # Toeplitz pur → erreur recon faible
    assert out["bochner_recon_err_median"] < 0.10


def test_r3_registered() -> None:
    assert REGISTRY.get("R3_bochner_stationarity") is R3BochnerStationarity


# S3 Hierarchical Tucker
def test_s3_basic() -> None:
    out = S3HierarchicalTucker().compute(_make_A(B=4, H=2, N=8), _ctx())
    assert "ht_rank_root_BH_NN" in out
    assert "ht_rank_max" in out


def test_s3_registered() -> None:
    assert REGISTRY.get("S3_hierarchical_tucker") is S3HierarchicalTucker


# V1 Pseudo-differential
def test_v1_basic() -> None:
    out = V1Pseudodifferential().compute(_make_A(), _ctx())
    assert "pseudodiff_order_m_median" in out


def test_v1_registered() -> None:
    assert REGISTRY.get("V1_pseudodifferential") is V1Pseudodifferential


# W3 NIP dependence
def test_w3_basic() -> None:
    out = W3NIPDependence().compute(_make_A(), _ctx())
    assert "vc_shatter_ratio_n3_mean" in out or "vc_shatter_ratio_n4_mean" in out


def test_w3_registered() -> None:
    assert REGISTRY.get("W3_nip_dependence") is W3NIPDependence


# K2 Persistent homology
def test_k2_basic() -> None:
    out = K2PersistentHomology().compute(_make_A(N=12, B=2, H=2), _ctx())
    assert "tda_beta_0_median" in out
    assert "tda_beta_1_median" in out


def test_k2_registered() -> None:
    assert REGISTRY.get("K2_persistent_homology") is K2PersistentHomology


# C6 S_Grad squelette
def test_c6_skips_without_metadata() -> None:
    out = C6SGrad().compute(_make_A(), _ctx())
    assert out["s_grad_available"] is False
    assert "skip_reason" in out


def test_c6_uses_injected_grad() -> None:
    A = _make_A()
    s_grad = torch.randn(4, 16, dtype=torch.float32)
    out = C6SGrad().compute(A, _ctx(s_grad=s_grad))
    assert out["s_grad_available"] is True
    assert "s_grad_per_seq_median" in out


def test_c6_registered() -> None:
    assert REGISTRY.get("C6_s_grad") is C6SGrad


# N1 F-divergence
def test_n1_skips_without_student() -> None:
    out = N1FDivergence().compute(_make_A(), _ctx())
    assert out["student_available"] is False


def test_n1_uses_student() -> None:
    A = _make_A(seed=0)
    S = _make_A(seed=1)  # different student
    out = N1FDivergence().compute(A, _ctx(student_attn=S))
    assert out["student_available"] is True
    assert "kl_oracle_student_median" in out


def test_n1_identical_oracle_student() -> None:
    """A = student → KL = 0."""
    A = _make_A()
    out = N1FDivergence().compute(A, _ctx(student_attn=A))
    assert out["kl_oracle_student_median"] < 1e-6


def test_n1_registered() -> None:
    assert REGISTRY.get("N1_f_divergence") is N1FDivergence


# N2 Preservation
def test_n2_identical_full_preservation() -> None:
    A = _make_A()
    out = N2Preservation().compute(A, _ctx(student_attn=A))
    assert out["preservation_overall"] > 0.9


def test_n2_skips_without_student() -> None:
    out = N2Preservation().compute(_make_A(), _ctx())
    assert out["student_available"] is False


def test_n2_registered() -> None:
    assert REGISTRY.get("N2_preservation") is N2Preservation


# N3 Lipschitz diff
def test_n3_identical_zero_diff() -> None:
    A = _make_A()
    tokens = torch.randint(0, 8, (4, 16))
    out = N3LipschitzDiff().compute(A, _ctx(student_attn=A, tokens=tokens))
    assert out["lipschitz_diff_max"] < 1e-6


def test_n3_skips_without_student() -> None:
    out = N3LipschitzDiff().compute(_make_A(), _ctx())
    assert out["student_available"] is False


def test_n3_registered() -> None:
    assert REGISTRY.get("N3_lipschitz_diff") is N3LipschitzDiff
