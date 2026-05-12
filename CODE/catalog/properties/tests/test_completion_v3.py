"""Tests pour les ~33 Properties V3 (catalogue final 131 props).

Couvre A2/A7, D4-D6, E3-E5, F3/F4, H5/H6, I4/I5, J5, K5, L4/L5, M3/M4, N4,
P7, Q6, R5/R6, S4/S5, T3/T4, V4/V5, W4/W5.
"""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.registry import REGISTRY


def _make_A(seed: int = 0, N: int = 16, B: int = 4, H: int = 2,
            dtype: torch.dtype = torch.float64) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(B, H, N, N, dtype=dtype), dim=-1)


def _ctx(**meta) -> PropertyContext:
    return PropertyContext(device="cpu", dtype=torch.float64, metadata=meta)


# ===== A2 stable rank =====
def test_a2_stable_rank() -> None:
    from catalog.properties.family_a_spectral.a2_stable_rank import A2StableRank
    A = _make_A()
    out = A2StableRank().compute(A, _ctx())
    assert "stable_rank_median" in out
    assert out["stable_rank_median"] >= 1.0  # always ≥ 1


def test_a2_rank1_matrix() -> None:
    from catalog.properties.family_a_spectral.a2_stable_rank import A2StableRank
    # Rank-1 outer product softmax → stable_rank ≈ 1
    u = torch.randn(4, 2, 16, 1, dtype=torch.float64)
    v = torch.randn(4, 2, 1, 16, dtype=torch.float64)
    A = torch.softmax((u @ v).abs(), dim=-1)
    out = A2StableRank().compute(A, _ctx())
    assert out["stable_rank_median"] < 5.0  # very low rank


# ===== A7 effective dimension =====
def test_a7_effective_dimension() -> None:
    from catalog.properties.family_a_spectral.a7_effective_dimension import (
        A7EffectiveDimension,
    )
    A = _make_A()
    out = A7EffectiveDimension().compute(A, _ctx())
    assert "d_eff_renyi2_median" in out
    assert 1.0 <= out["d_eff_renyi2_median"] <= 16.0


# ===== D4 frobenius baseline =====
def test_d4_frobenius_baseline() -> None:
    from catalog.properties.family_d_geometric.d4_frobenius_baseline import (
        D4FrobeniusBaseline,
    )
    A = _make_A()
    out = D4FrobeniusBaseline().compute(A, _ctx(seed=42))
    assert "distance_random_softmax_median" in out
    assert out["distance_random_softmax_median"] >= 0.0


# ===== D5 nuclear norm =====
def test_d5_nuclear_norm() -> None:
    from catalog.properties.family_d_geometric.d5_nuclear_norm import D5NuclearNorm
    A = _make_A()
    out = D5NuclearNorm().compute(A, _ctx())
    assert "nuclear_norm_median" in out
    assert "nuclear_over_spectral_median" in out
    assert out["nuclear_over_spectral_median"] >= 1.0


# ===== D6 Grassmann distance =====
def test_d6_grassmann_distance() -> None:
    from catalog.properties.family_d_geometric.d6_grassmann_distance import (
        D6GrassmannDistance,
    )
    A_list = [_make_A(seed=i) for i in range(3)]
    out = D6GrassmannDistance(top_r=3).compute(A_list, _ctx())
    assert "grassmann_distance_median" in out
    assert out["n_layers"] == 3


def test_d6_skip_single_layer() -> None:
    from catalog.properties.family_d_geometric.d6_grassmann_distance import (
        D6GrassmannDistance,
    )
    out = D6GrassmannDistance().compute([_make_A()], _ctx())
    assert "skip_reason" in out


# ===== E3 redundancy =====
def test_e3_redundancy() -> None:
    from catalog.properties.family_e_information.e3_redundancy import E3Redundancy
    A = _make_A()
    out = E3Redundancy().compute(A, _ctx())
    assert 0.0 <= out["redundancy_median"] <= 1.0


def test_e3_uniform_zero_redundancy() -> None:
    from catalog.properties.family_e_information.e3_redundancy import E3Redundancy
    # uniform A = 1/N → H = log N → redundancy = 0
    A = torch.full((2, 2, 8, 8), 1.0 / 8.0, dtype=torch.float64)
    out = E3Redundancy().compute(A, _ctx())
    assert out["redundancy_median"] < 1e-6


# ===== E4 MDL proxy =====
def test_e4_mdl_proxy() -> None:
    from catalog.properties.family_e_information.e4_mdl_proxy import E4MdlProxy
    A = _make_A()
    out = E4MdlProxy().compute(A, _ctx())
    assert "mdl_bits_median" in out
    assert out["mdl_bits_median"] > 0.0


# ===== E5 entropy rate =====
def test_e5_entropy_rate() -> None:
    from catalog.properties.family_e_information.e5_entropy_rate import E5EntropyRate
    A = _make_A()
    out = E5EntropyRate().compute(A, _ctx())
    assert "entropy_rate_nats_median" in out
    assert 0.0 <= out["entropy_rate_normalized_median"] <= 1.05


# ===== F3 jacobian =====
def test_f3_jacobian_proxy() -> None:
    from catalog.properties.family_f_dynamic.f3_jacobian_proxy import F3JacobianProxy
    A = _make_A()
    out = F3JacobianProxy().compute(A, _ctx())
    assert "jacobian_max_per_row_median" in out
    assert 0.0 <= out["jacobian_max_per_row_median"] <= 0.25  # p(1-p) max=0.25


# ===== F4 Lyapunov =====
def test_f4_lyapunov_proxy() -> None:
    from catalog.properties.family_f_dynamic.f4_lyapunov_proxy import F4LyapunovProxy
    A = _make_A()
    out = F4LyapunovProxy().compute(A, _ctx())
    assert "lyapunov_log_smax_median" in out


# ===== H5 deep residual =====
def test_h5_deep_residual() -> None:
    from catalog.properties.family_h_cross_layer.h5_deep_residual_norm import (
        H5DeepResidualNorm,
    )
    A_list = [_make_A(seed=i) for i in range(4)]
    out = H5DeepResidualNorm().compute(A_list, _ctx())
    assert "deep_residual_median" in out
    assert out["n_layers"] == 4


def test_h5_identity_zero_residual() -> None:
    from catalog.properties.family_h_cross_layer.h5_deep_residual_norm import (
        H5DeepResidualNorm,
    )
    A = _make_A()
    out = H5DeepResidualNorm().compute([A, A.clone(), A.clone()], _ctx())
    assert out["deep_residual_median"] < 1e-10


# ===== H6 attention sink =====
def test_h6_attention_sink() -> None:
    from catalog.properties.family_h_cross_layer.h6_attention_sink_score import (
        H6AttentionSinkScore,
    )
    A_list = [_make_A(seed=i) for i in range(3)]
    out = H6AttentionSinkScore().compute(A_list, _ctx())
    assert "sink_score_layer_0" in out
    assert "sink_score_mean_across_layers" in out


# ===== I4 head agreement =====
def test_i4_head_agreement() -> None:
    from catalog.properties.family_i_cross_head.i4_head_agreement import (
        I4HeadAgreement,
    )
    A = _make_A(H=4)
    out = I4HeadAgreement().compute(A, _ctx())
    assert "head_agreement_median" in out
    assert 0.0 <= out["head_agreement_median"] <= 1.0


def test_i4_single_head_skip() -> None:
    from catalog.properties.family_i_cross_head.i4_head_agreement import (
        I4HeadAgreement,
    )
    A = _make_A(H=1)
    out = I4HeadAgreement().compute(A, _ctx())
    assert "skip_reason" in out


# ===== I5 head redundancy =====
def test_i5_head_redundancy() -> None:
    from catalog.properties.family_i_cross_head.i5_head_redundancy import (
        I5HeadRedundancy,
    )
    A = _make_A(H=4)
    out = I5HeadRedundancy().compute(A, _ctx())
    assert "head_corr_median" in out
    assert -1.0 <= out["head_corr_median"] <= 1.0


# ===== J5 spectral gap =====
def test_j5_spectral_gap() -> None:
    from catalog.properties.family_j_markov.j5_spectral_gap import J5SpectralGap
    A = _make_A()
    out = J5SpectralGap().compute(A, _ctx())
    assert "spectral_gap_median" in out
    assert 0.0 <= out["spectral_gap_median"] <= 1.0
    assert out["lambda1_abs_median"] >= out["lambda2_abs_median"]


# ===== K5 Cheeger =====
def test_k5_cheeger() -> None:
    from catalog.properties.family_k_graph.k5_cheeger_constant import (
        K5CheegerConstant,
    )
    A = _make_A()
    out = K5CheegerConstant().compute(A, _ctx())
    assert "cheeger_upper_median" in out


# ===== L4 DCT =====
def test_l4_dct_energy() -> None:
    from catalog.properties.family_l_frequency.l4_dct_energy import L4DctEnergy
    A = _make_A()
    out = L4DctEnergy().compute(A, _ctx())
    assert "dct_energy_low_frac_0p10_median" in out
    assert "dct_entropy_normalized_median" in out


# ===== L5 spectral peaks =====
def test_l5_spectral_peaks() -> None:
    from catalog.properties.family_l_frequency.l5_spectral_peaks import (
        L5SpectralPeaks,
    )
    A = _make_A()
    out = L5SpectralPeaks().compute(A, _ctx())
    assert "n_peaks_per_row_median" in out


# ===== M3 input dependence =====
def test_m3_input_dependence() -> None:
    from catalog.properties.family_m_conditional.m3_input_dependence import (
        M3InputDependence,
    )
    A = _make_A(B=8)
    out = M3InputDependence().compute(A, _ctx())
    assert "input_variance_per_head_median" in out


def test_m3_single_example_skip() -> None:
    from catalog.properties.family_m_conditional.m3_input_dependence import (
        M3InputDependence,
    )
    A = _make_A(B=1)
    out = M3InputDependence().compute(A, _ctx())
    assert "skip_reason" in out


# ===== M4 token class sensitivity =====
def test_m4_token_class_sensitivity() -> None:
    from catalog.properties.family_m_conditional.m4_token_class_sensitivity import (
        M4TokenClassSensitivity,
    )
    A = _make_A(B=4, N=16)
    tokens = torch.randint(0, 4, (4, 16))
    out = M4TokenClassSensitivity(min_count_per_class=2).compute(
        A, _ctx(tokens=tokens)
    )
    # Soit on a une mesure, soit skip cleanly
    assert "inter_class_entropy_variance" in out or "skip_reason" in out


def test_m4_no_tokens_skip() -> None:
    from catalog.properties.family_m_conditional.m4_token_class_sensitivity import (
        M4TokenClassSensitivity,
    )
    A = _make_A()
    out = M4TokenClassSensitivity().compute(A, _ctx())
    assert "skip_reason" in out


# ===== N4 prediction agreement =====
def test_n4_prediction_agreement_self() -> None:
    from catalog.properties.family_n_comparative.n4_prediction_agreement import (
        N4PredictionAgreement,
    )
    A = _make_A()
    out = N4PredictionAgreement().compute(A, _ctx(student_attn=A.clone()))
    # Identique → agreement = 1
    assert out["top1_agreement_mean"] > 0.99


def test_n4_no_student_skip() -> None:
    from catalog.properties.family_n_comparative.n4_prediction_agreement import (
        N4PredictionAgreement,
    )
    A = _make_A()
    out = N4PredictionAgreement().compute(A, _ctx())
    assert "skip_reason" in out


# ===== P7 markov realization =====
def test_p7_markov_realization() -> None:
    from catalog.properties.family_p_realization.p7_markov_realization_test import (
        P7MarkovRealizationTest,
    )
    A = _make_A()
    out = P7MarkovRealizationTest().compute(A, _ctx())
    assert "markov_realization_rank_median" in out


# ===== Q6 HSS off-diagonal rank =====
def test_q6_hss_off_diag() -> None:
    from catalog.properties.family_q_hierarchical.q6_hss_off_diagonal_rank import (
        Q6HssOffDiagonalRank,
    )
    A = _make_A()
    out = Q6HssOffDiagonalRank().compute(A, _ctx())
    assert "off_diag_rank_top_right_median" in out
    assert out["block_size_each"] == 8


# ===== R5 Gaussian kernel =====
def test_r5_gaussian_kernel_test() -> None:
    from catalog.properties.family_r_rkhs.r5_gaussian_kernel_test import (
        R5GaussianKernelTest,
    )
    A = _make_A()
    out = R5GaussianKernelTest().compute(A, _ctx())
    assert "symmetry_ratio_median" in out
    assert "diagonal_dominance_fraction_median" in out


# ===== R6 Nyström spectrum =====
def test_r6_spectral_kernel_proxy() -> None:
    from catalog.properties.family_r_rkhs.r6_spectral_kernel_proxy import (
        R6SpectralKernelProxy,
    )
    A = _make_A(N=32)
    out = R6SpectralKernelProxy(m_sample=8).compute(A, _ctx())
    assert "nystrom_spectrum_corr_median" in out


# ===== S4 CP rank =====
def test_s4_cp_rank_proxy() -> None:
    from catalog.properties.family_s_tensors.s4_cp_rank_proxy import S4CpRankProxy
    A = _make_A(N=8, B=2, H=2)
    out = S4CpRankProxy(ranks=(1, 2), n_iter=5).compute(A, _ctx())
    assert "cp_residual_R1" in out
    assert "cp_residual_R2" in out


# ===== S5 unfolding rank =====
def test_s5_unfolding_rank() -> None:
    from catalog.properties.family_s_tensors.s5_unfolding_rank import (
        S5UnfoldingRank,
    )
    A = _make_A(N=8)
    out = S5UnfoldingRank().compute(A, _ctx())
    assert "rank_unfold_batch_head" in out


# ===== T3 cyclic equivariance =====
def test_t3_cyclic_equivariance() -> None:
    from catalog.properties.family_t_equivariance.t3_cyclic_equivariance import (
        T3CyclicEquivariance,
    )
    A = _make_A()
    out = T3CyclicEquivariance().compute(A, _ctx())
    assert "epsilon_cyclic_k1_median" in out


# ===== T4 reflection equivariance =====
def test_t4_reflection_equivariance() -> None:
    from catalog.properties.family_t_equivariance.t4_reflection_equivariance import (
        T4ReflectionEquivariance,
    )
    A = _make_A()
    out = T4ReflectionEquivariance().compute(A, _ctx())
    assert "epsilon_reflection_median" in out


def test_t4_symmetric_zero_epsilon() -> None:
    from catalog.properties.family_t_equivariance.t4_reflection_equivariance import (
        T4ReflectionEquivariance,
    )
    # Build a J-symmetric matrix : A = J A J ssi A[i,j] = A[N-1-i, N-1-j]
    N = 8
    Q = torch.randn(2, 2, N, N, dtype=torch.float64)
    A_sym = 0.5 * (Q + Q.flip(dims=(-2, -1)))
    A_sym = torch.softmax(A_sym, dim=-1)
    # Re-symétriser après softmax (approximation)
    A_sym = 0.5 * (A_sym + A_sym.flip(dims=(-2, -1)))
    A_sym = A_sym / A_sym.sum(-1, keepdim=True)
    out = T4ReflectionEquivariance().compute(A_sym, _ctx())
    # Devrait être assez petit
    assert out["epsilon_reflection_median"] < 0.30


# ===== V4 commutator =====
def test_v4_commutator_norm() -> None:
    from catalog.properties.family_v_operators.v4_commutator_norm import (
        V4CommutatorNorm,
    )
    A = _make_A(H=4)
    out = V4CommutatorNorm().compute(A, _ctx())
    assert "commutator_norm_median" in out


def test_v4_single_head_skip() -> None:
    from catalog.properties.family_v_operators.v4_commutator_norm import (
        V4CommutatorNorm,
    )
    A = _make_A(H=1)
    out = V4CommutatorNorm().compute(A, _ctx())
    assert "skip_reason" in out


# ===== V5 Schatten =====
def test_v5_schatten() -> None:
    from catalog.properties.family_v_operators.v5_schatten_p_norm import (
        V5SchattenPNorm,
    )
    A = _make_A()
    out = V5SchattenPNorm().compute(A, _ctx())
    assert "schatten_S1_nuclear_median" in out
    assert "schatten_S2_frobenius_median" in out
    assert "schatten_Sinf_spectral_median" in out
    # S_∞ ≤ S_2 ≤ S_1
    assert (out["schatten_Sinf_spectral_median"]
            <= out["schatten_S2_frobenius_median"] + 1e-6)
    assert (out["schatten_S2_frobenius_median"]
            <= out["schatten_S1_nuclear_median"] + 1e-6)


# ===== W4 NIP score =====
def test_w4_nip_score() -> None:
    from catalog.properties.family_w_logic.w4_nip_score import W4NipScore
    A = _make_A(N=16)
    out = W4NipScore(n_samples=8, ns_to_test=(3, 4)).compute(A, _ctx())
    assert "nip_score" in out or "skip_reason" in out


# ===== W5 VC proxy =====
def test_w5_vc_proxy() -> None:
    from catalog.properties.family_w_logic.w5_vc_proxy import W5VcProxy
    A = _make_A(N=16)
    out = W5VcProxy(n_samples=8, n_max=4).compute(A, _ctx())
    assert "vc_proxy_max_shattered_n" in out
    assert out["vc_proxy_max_shattered_n"] >= 0


# ===== Registry global : 131 Properties =====
def test_registry_has_131_properties() -> None:
    all_props = REGISTRY.all()
    assert len(all_props) == 131, f"Expected 131, got {len(all_props)}"


def test_registry_all_families_covered() -> None:
    by_fam: dict[str, int] = {}
    for cls in REGISTRY.all():
        by_fam[cls.family] = by_fam.get(cls.family, 0) + 1
    # Toutes les familles A-W et N représentées
    expected = set("ABCDEFGHIJKLMNOPQRSTUVW")
    assert set(by_fam.keys()) == expected
    # Chaque famille a au moins 4 Properties
    for f, n in by_fam.items():
        assert n >= 4, f"family {f} only has {n} properties"
