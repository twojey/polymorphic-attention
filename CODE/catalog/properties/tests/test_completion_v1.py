"""Tests pour les properties V1 finales : B8/B9/G4/G5/C8/O3/P2/Q3/R2/R4/S2/T2/U4/V3/W2/K4."""

from __future__ import annotations

import torch

from catalog.properties.base import PropertyContext
from catalog.properties.family_b_structural.b8_sylvester_rank import B8SylvesterRank
from catalog.properties.family_b_structural.b9_quasiseparable import B9Quasiseparable
from catalog.properties.family_c_token_stats.c8_fisher_condition import C8FisherCondition
from catalog.properties.family_g_algebraic.g4_charpoly_coefficients import (
    G4CharpolyCoefficients,
)
from catalog.properties.family_g_algebraic.g5_additivity_test import G5AdditivityTest
from catalog.properties.family_k_graph.k4_modularity import K4Modularity
from catalog.properties.family_o_displacement.o3_vandermonde_displacement_rank import (
    O3VandermondeDisplacementRank,
)
from catalog.properties.family_p_realization.p2_hsv_details import P2HSVDetails
from catalog.properties.family_q_hierarchical.q3_nestedness import Q3Nestedness
from catalog.properties.family_r_rkhs.r2_rff_approximation import R2RFFApproximation
from catalog.properties.family_r_rkhs.r4_truncated_energy import R4TruncatedEnergy
from catalog.properties.family_s_tensors.s2_tensor_train_rank import S2TensorTrainRank
from catalog.properties.family_t_equivariance.t2_subgroup_equivariance import (
    T2SubgroupEquivariance,
)
from catalog.properties.family_u_sparse_structured.u4_pixelfly_distance import (
    U4PixelflyDistance,
)
from catalog.properties.family_v_operators.v3_compactness import V3Compactness
from catalog.properties.family_w_logic.w2_dependence_proxy import W2DependenceProxy
from catalog.properties.registry import REGISTRY


def _make_A(seed: int = 0, N: int = 16) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(1, 1, N, N, dtype=torch.float64), dim=-1)


def _ctx() -> PropertyContext:
    return PropertyContext(device="cpu", dtype=torch.float64)


def test_b8_basic() -> None:
    out = B8SylvesterRank().compute(_make_A(), _ctx())
    assert "sylvester_rank_eff_median" in out


def test_b8_registered() -> None:
    assert REGISTRY.get("B8_sylvester_rank") is B8SylvesterRank


def test_b9_basic() -> None:
    out = B9Quasiseparable().compute(_make_A(), _ctx())
    assert "qs_combined_max" in out


def test_b9_registered() -> None:
    assert REGISTRY.get("B9_quasiseparable") is B9Quasiseparable


def test_c8_basic() -> None:
    out = C8FisherCondition().compute(_make_A(), _ctx())
    assert "log10_kappa_fisher_median" in out


def test_c8_one_hot_high_kappa() -> None:
    """One-hot → max p = 1, min p ≈ eps_floor → log10 κ très grand."""
    n = 8
    A = torch.zeros(1, 1, n, n, dtype=torch.float64)
    for t in range(n):
        A[0, 0, t, t] = 1.0
    out = C8FisherCondition().compute(A, _ctx())
    # κ = 1/eps_floor = 1e30 → log10 κ = 30
    assert out["log10_kappa_fisher_median"] > 20


def test_c8_registered() -> None:
    assert REGISTRY.get("C8_fisher_condition") is C8FisherCondition


def test_g4_basic() -> None:
    out = G4CharpolyCoefficients(k_max=3).compute(_make_A(), _ctx())
    assert "e_1_median" in out
    assert "e_2_median" in out
    assert "e_3_median" in out


def test_g4_identity_charpoly_coefficients() -> None:
    """I_n : charpoly = (λ−1)^n → e_k = C(n, k)."""
    n = 4
    A = torch.eye(n, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    out = G4CharpolyCoefficients(k_max=2).compute(A, _ctx())
    # e_1 = tr(I) = 4 ; e_2 = C(4,2) = 6
    assert abs(out["e_1_median"] - n) < 1e-6
    assert abs(out["e_2_median"] - 6) < 1e-6


def test_g4_registered() -> None:
    assert REGISTRY.get("G4_charpoly_coefficients") is G4CharpolyCoefficients


def test_g5_linear_zero_eps() -> None:
    """Matrice linéaire pure : ε_add = 0."""
    out = G5AdditivityTest().compute(_make_A(), _ctx())
    assert out["additivity_eps_median"] < 1e-10


def test_g5_registered() -> None:
    assert REGISTRY.get("G5_additivity_test") is G5AdditivityTest


def test_k4_basic() -> None:
    A = _make_A(N=16)
    out = K4Modularity().compute(A, _ctx())
    assert "modularity_median" in out


def test_k4_block_diag_high_modularity() -> None:
    """Matrice quasi block-diag avec petit lien inter-bloc → modularité positive."""
    torch.manual_seed(42)
    n = 16
    A_dense = torch.zeros(n, n, dtype=torch.float64)
    A_dense[:8, :8] = torch.rand(8, 8, dtype=torch.float64) + 0.5  # bloc 1 dense
    A_dense[8:, 8:] = torch.rand(8, 8, dtype=torch.float64) + 0.5  # bloc 2 dense
    # Lien inter-bloc faible (pour casser dégénérescence eigvalsh)
    A_dense[:8, 8:] = torch.rand(8, 8, dtype=torch.float64) * 0.01
    A_dense[8:, :8] = A_dense[:8, 8:].T
    A_dense = (A_dense + A_dense.T) / 2
    A = A_dense.unsqueeze(0).unsqueeze(0)
    out = K4Modularity().compute(A, _ctx())
    # Avec deux blocs denses séparés, Q doit être > 0.1 (positif net)
    assert out["modularity_median"] > 0.1


def test_k4_registered() -> None:
    assert REGISTRY.get("K4_modularity") is K4Modularity


def test_o3_basic() -> None:
    out = O3VandermondeDisplacementRank().compute(_make_A(), _ctx())
    assert "best_r_eff_median" in out


def test_o3_registered() -> None:
    assert REGISTRY.get("O3_vandermonde_displacement_rank") is O3VandermondeDisplacementRank


def test_p2_basic() -> None:
    out = P2HSVDetails().compute(_make_A(), _ctx())
    assert "balanced_order_tau_0p01_median" in out
    assert "balanced_order_tau_0p10_median" in out


def test_p2_registered() -> None:
    assert REGISTRY.get("P2_hsv_details") is P2HSVDetails


def test_q3_basic() -> None:
    out = Q3Nestedness().compute(_make_A(N=16), _ctx())
    assert "nestedness_ratio_median" in out


def test_q3_registered() -> None:
    assert REGISTRY.get("Q3_nestedness") is Q3Nestedness


def test_r2_basic() -> None:
    out = R2RFFApproximation(n_features=16).compute(_make_A(), _ctx())
    assert "rff_epsilon_median" in out


def test_r2_registered() -> None:
    assert REGISTRY.get("R2_rff_approximation") is R2RFFApproximation


def test_r4_basic() -> None:
    out = R4TruncatedEnergy().compute(_make_A(), _ctx())
    assert "truncated_energy_top_0p10_median" in out


def test_r4_full_fraction_one() -> None:
    """Top-100% des σ capture 100% énergie."""
    out = R4TruncatedEnergy(k_fractions=(1.0,)).compute(_make_A(), _ctx())
    assert abs(out["truncated_energy_top_1p00_median"] - 1.0) < 1e-6


def test_r4_registered() -> None:
    assert REGISTRY.get("R4_truncated_energy") is R4TruncatedEnergy


def test_s2_basic() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    out = S2TensorTrainRank().compute(A, _ctx())
    assert "tt_rank_1" in out
    assert "tt_rank_2" in out
    assert "tt_rank_3" in out


def test_s2_registered() -> None:
    assert REGISTRY.get("S2_tensor_train_rank") is S2TensorTrainRank


def test_t2_basic() -> None:
    out = T2SubgroupEquivariance().compute(_make_A(), _ctx())
    assert "eps_reverse_median" in out
    assert "eps_best_subgroup_median" in out


def test_t2_constant_zero_eps() -> None:
    """A constante → toutes permutations identiques → ε=0."""
    A = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float64)
    out = T2SubgroupEquivariance().compute(A, _ctx())
    assert out["eps_reverse_median"] < 1e-10


def test_t2_registered() -> None:
    assert REGISTRY.get("T2_subgroup_equivariance") is T2SubgroupEquivariance


def test_u4_basic() -> None:
    out = U4PixelflyDistance().compute(_make_A(), _ctx())
    assert "epsilon_pixelfly_median" in out
    assert "mask_density" in out


def test_u4_registered() -> None:
    assert REGISTRY.get("U4_pixelfly_distance") is U4PixelflyDistance


def test_v3_basic() -> None:
    out = V3Compactness().compute(_make_A(), _ctx())
    assert "tail_ratio_sigmaK_over_sigma1_median" in out


def test_v3_registered() -> None:
    assert REGISTRY.get("V3_compactness") is V3Compactness


def test_w2_basic() -> None:
    out = W2DependenceProxy().compute(_make_A(), _ctx())
    assert "tree_likeness_ratio_median" in out


def test_w2_registered() -> None:
    assert REGISTRY.get("W2_dependence_proxy") is W2DependenceProxy
