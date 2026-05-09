"""Tests phase 2 sur matrices synthétiques de structure connue."""

from __future__ import annotations

import numpy as np
import torch

from phase2_audit_spectral.batteries.battery_a import (
    PROJECTORS_V1,
    fit_additive_composition,
    fit_class,
    project_hankel,
    project_identity,
    project_toeplitz,
)
from phase2_audit_spectral.batteries.battery_b import analyze_residual_svd, residual_analysis
from phase2_audit_spectral.batteries.battery_d import detect_orphan_regimes, eigen_svd_asymmetry
from phase2_audit_spectral.head_specialization import diagnose_heads, top_specialized_heads
from phase2_audit_spectral.stress_rank_map import build_monovariate_srm
from phase2_audit_spectral.svd_pipeline import r_eff_from_singular_values, svd_attention
from phase2_audit_spectral.transfer_law import fit_transfer_law


# ----------------------------------------------------------------
# SVD pipeline
# ----------------------------------------------------------------


def test_r_eff_rank_one() -> None:
    s = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert int(r_eff_from_singular_values(s, 0.95).item()) == 1
    assert int(r_eff_from_singular_values(s, 0.99).item()) == 1


def test_r_eff_uniform_distribution() -> None:
    s = torch.ones(10, dtype=torch.float64)
    # 95% nécessite 10 valeurs (tout)
    assert int(r_eff_from_singular_values(s, 0.95).item()) >= 9


def test_svd_attention_shapes() -> None:
    A = torch.softmax(torch.randn(2, 3, 8, 8, dtype=torch.float64), dim=-1)
    out = svd_attention(A, theta_values=(0.95,))
    assert out["s"].shape == (2, 3, 8)
    assert out["r_eff_95"].shape == (2, 3)


# ----------------------------------------------------------------
# Battery A — projecteurs
# ----------------------------------------------------------------


def test_project_toeplitz_identity_on_toeplitz() -> None:
    # Matrice Toeplitz exacte
    n = 6
    c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = c[abs(i - j)]
    # Toeplitz symétrique = projection sur Toeplitz doit reproduire (à l'erreur de moyenne près)
    proj = project_toeplitz(A)
    eps = fit_class(A, "toeplitz")
    assert eps < 0.01, f"Toeplitz exacte mais ε={eps}"


def test_project_hankel_identity_on_hankel() -> None:
    n = 6
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            A[i, j] = float(i + j)
    eps = fit_class(A, "hankel")
    assert eps < 0.01, f"Hankel exacte mais ε={eps}"


def test_project_identity_keeps_diagonal() -> None:
    A = torch.randn(5, 5, dtype=torch.float64)
    proj = project_identity(A)
    assert torch.allclose(torch.diagonal(proj), torch.diagonal(A))
    # off-diagonal = 0
    off = proj - torch.diag(torch.diagonal(proj))
    assert off.abs().max().item() < 1e-12


def test_fit_class_random_matrix_high_eps() -> None:
    torch.manual_seed(0)
    A = torch.randn(8, 8, dtype=torch.float64)
    # Une matrice aléatoire ne projette mal sur aucune des 3 classes
    eps_t = fit_class(A, "toeplitz")
    eps_h = fit_class(A, "hankel")
    eps_i = fit_class(A, "identity")
    assert min(eps_t, eps_h, eps_i) > 0.5


def test_additive_composition_better_than_single() -> None:
    # Toeplitz + Hankel : la composition doit faire mieux que chaque seule
    n = 8
    rng = torch.manual_seed(42)
    c = torch.randn(n, dtype=torch.float64)
    T = torch.stack([torch.roll(c, k) for k in range(n)], dim=0)
    h = torch.randn(2 * n - 1, dtype=torch.float64)
    H = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            H[i, j] = h[i + j]
    A = T + 0.5 * H

    eps_solo_t = fit_class(A, "toeplitz")
    eps_compo, _, _ = fit_additive_composition(A, class1="toeplitz", class2="hankel")
    assert eps_compo < eps_solo_t * 0.8


# ----------------------------------------------------------------
# Battery B — résidu
# ----------------------------------------------------------------


def test_residual_svd_low_rank() -> None:
    # Résidu = uv^T pur → top-1 contribue 100% de la variance
    u = torch.randn(8, dtype=torch.float64)
    v = torch.randn(8, dtype=torch.float64)
    residual = u.unsqueeze(1) @ v.unsqueeze(0)
    ratio, top_s = analyze_residual_svd(residual, top_k=2)
    # σ_1 contient toute la variance, σ_2 = 0
    assert ratio > 0.99


def test_residual_analysis_smoke() -> None:
    r = torch.randn(8, 8, dtype=torch.float64)
    result = residual_analysis(r)
    assert result.norm_residual > 0
    assert 0 <= result.svd_top_k_ratio <= 1


# ----------------------------------------------------------------
# Battery D — orphelins, asymétrie
# ----------------------------------------------------------------


def test_detect_orphan_regimes() -> None:
    epsilons = {
        ("good",): {"toeplitz": 0.05, "hankel": 0.10, "identity": 0.50},
        ("orphan",): {"toeplitz": 0.50, "hankel": 0.55, "identity": 0.60},
    }
    orphans = detect_orphan_regimes(epsilons, threshold=0.30)
    assert ("orphan",) in orphans
    assert ("good",) not in orphans


def test_eigen_svd_asymmetry_symmetric() -> None:
    A = torch.randn(6, 6, dtype=torch.float64)
    A_sym = (A + A.T) / 2
    assert eigen_svd_asymmetry(A_sym) < 1e-10


def test_eigen_svd_asymmetry_triangular() -> None:
    A = torch.tril(torch.ones(5, 5, dtype=torch.float64))
    asym = eigen_svd_asymmetry(A)
    assert asym > 0.3


# ----------------------------------------------------------------
# Stress-Rank Map
# ----------------------------------------------------------------


def test_srm_monovariate_groups_by_axis() -> None:
    n = 200
    rng = np.random.default_rng(0)
    omega = rng.choice([1, 2, 4], size=n)
    r_eff = omega + rng.normal(0, 0.1, size=n)
    srm = build_monovariate_srm(
        r_eff_values=r_eff, omega=omega.astype(float),
        delta=np.zeros(n), entropy=np.zeros(n),
    )
    assert 1.0 in srm["omega"]
    assert 4.0 in srm["omega"]
    assert srm["omega"][1.0].median < srm["omega"][4.0].median


# ----------------------------------------------------------------
# Loi de transfert
# ----------------------------------------------------------------


def test_fit_transfer_law_recovers_exponents() -> None:
    rng = np.random.default_rng(0)
    n = 500
    omega = rng.integers(0, 12, size=n)
    delta = rng.integers(0, 4096, size=n)
    entropy = rng.uniform(0, 1, size=n)
    # Vraie loi : r = 2.0 · (1+ω)^0.5 · (1+Δ)^0.3 · exp(0.2·ℋ) + bruit
    r_true = 2.0 * (1 + omega) ** 0.5 * (1 + delta) ** 0.3 * np.exp(0.2 * entropy)
    r_obs = r_true * np.exp(rng.normal(0, 0.05, size=n))
    fit = fit_transfer_law(r_target=r_obs, omega=omega.astype(float),
                            delta=delta.astype(float), entropy=entropy)
    assert fit.r2 > 0.95
    assert abs(fit.alpha - 0.5) < 0.05
    assert abs(fit.beta - 0.3) < 0.05
    assert abs(fit.gamma - 0.2) < 0.05


# ----------------------------------------------------------------
# Spécialisation des têtes
# ----------------------------------------------------------------


def test_diagnose_heads_detects_dormant() -> None:
    rng = np.random.default_rng(0)
    L, H, R = 2, 4, 100
    r_eff = rng.uniform(5, 10, size=(L, H, R))
    # Tête (0, 0) dormante : faible r et faible variance
    r_eff[0, 0] = 0.1 + rng.normal(0, 0.05, size=R)
    diags = diagnose_heads(r_eff=r_eff, dormant_threshold=0.5, spec_dormant_threshold=0.1)
    dormant = [d for d in diags if d.is_dormant]
    assert any(d.layer == 0 and d.head == 0 for d in dormant)


def test_top_specialized_heads_orders_by_var() -> None:
    rng = np.random.default_rng(0)
    L, H, R = 1, 3, 50
    r_eff = np.zeros((L, H, R))
    r_eff[0, 0] = rng.uniform(1, 2, size=R)        # faible variance
    r_eff[0, 1] = rng.uniform(1, 10, size=R)       # haute variance
    r_eff[0, 2] = rng.uniform(2, 5, size=R)        # moyenne
    diags = diagnose_heads(r_eff=r_eff)
    top = top_specialized_heads(diags, k=1)
    assert top[0].head == 1
