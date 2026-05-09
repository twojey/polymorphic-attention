"""Tests phase 3 : Soft-Mask, Matriochka, ASPLayer, sanity checks."""

from __future__ import annotations

import torch

from phase3_kernel_asp.asp_layer import ASPLayer, ASPLayerConfig
from phase3_kernel_asp.backbone import IdentityBackbone, LinearBackbone
from phase3_kernel_asp.losses import (
    loss_consistency,
    loss_matriochka,
    matriochka_rank_schedule,
    matriochka_weights,
)
from phase3_kernel_asp.matriochka import MatriochkaBases, MatriochkaInitConfig
from phase3_kernel_asp.sanity import (
    sanity_check_collapse,
    sanity_check_monotone_quality,
    sanity_check_smoothness,
)
from phase3_kernel_asp.soft_mask import hard_threshold_ste, soft_mask


# ----------------------------------------------------------------
# Soft-Mask
# ----------------------------------------------------------------


def test_soft_mask_alpha_one_saturates() -> None:
    alpha = torch.tensor(1.0)
    m = soft_mask(alpha=alpha, R_max=8, beta=4.0)
    # tous les m proches de 1
    assert (m > 0.5).all()


def test_soft_mask_alpha_zero_collapses() -> None:
    alpha = torch.tensor(0.0)
    m = soft_mask(alpha=alpha, R_max=8, beta=4.0)
    # tous les m proches de 0
    assert (m < 0.5).all()


def test_soft_mask_monotone_decreasing() -> None:
    # Pour alpha intermédiaire, m doit décroître monotonement avec i
    alpha = torch.tensor(0.5)
    m = soft_mask(alpha=alpha, R_max=16, beta=4.0).squeeze()
    diffs = m[1:] - m[:-1]
    assert (diffs <= 1e-6).all()


def test_soft_mask_batch_shape() -> None:
    alpha = torch.rand(3, 5)
    m = soft_mask(alpha=alpha, R_max=8)
    assert m.shape == (3, 5, 8)


def test_hard_threshold_ste_forward_hard() -> None:
    soft = torch.tensor([0.3, 0.7, 0.5, 0.9])
    out = hard_threshold_ste(soft)
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.allclose(out, expected)


# ----------------------------------------------------------------
# Matriochka
# ----------------------------------------------------------------


def test_matriochka_correction_rank_zero() -> None:
    bases = MatriochkaBases(d_model=16, R_max=8)
    x = torch.randn(2, 5, 16)
    out = bases.correction(x, r=0)
    assert torch.allclose(out, torch.zeros_like(x))


def test_matriochka_correction_grows_with_rank() -> None:
    bases = MatriochkaBases(d_model=16, R_max=8)
    x = torch.randn(2, 5, 16)
    norms = [bases.correction(x, r=r).norm().item() for r in [0, 1, 4, 8]]
    # Les normes doivent globalement augmenter (avec r)
    assert norms[0] == 0
    assert norms[3] >= norms[1]


def test_matriochka_smart_init_preserves_columns() -> None:
    d, R, K = 16, 8, 4
    smart_vectors = torch.randn(d, K)
    init = MatriochkaInitConfig(strategy="smart", smart_init_vectors=smart_vectors)
    bases = MatriochkaBases(d_model=d, R_max=R, init=init)
    assert torch.allclose(bases.U_max[:, :K], smart_vectors)


# ----------------------------------------------------------------
# ASPLayer + sanity checks
# ----------------------------------------------------------------


def test_asplayer_forward_shape() -> None:
    cfg = ASPLayerConfig(d_model=16, R_max=8)
    layer = ASPLayer(cfg, backbone=LinearBackbone(16))
    x = torch.randn(2, 5, 16)
    out = layer.forward_with_rank(x, r=4)
    assert out.shape == (2, 5, 16)


def test_asplayer_collapse_at_rank_zero() -> None:
    cfg = ASPLayerConfig(d_model=16, R_max=8, layernorm=True)
    layer = ASPLayer(cfg, backbone=LinearBackbone(16))
    layer.eval()
    x = torch.randn(2, 5, 16)
    passed, diff = sanity_check_collapse(layer, x)
    assert passed, f"Effondrement échoué : diff={diff}"


def test_asplayer_alpha_zero_equivalent_to_rank_zero() -> None:
    cfg = ASPLayerConfig(d_model=16, R_max=8, soft_mask_beta=10.0)
    layer = ASPLayer(cfg, backbone=IdentityBackbone())
    layer.eval()
    x = torch.randn(2, 5, 16)
    out_alpha = layer.forward_with_alpha(x, alpha=torch.zeros(2, 5))
    out_r0 = layer.forward_with_rank(x, r=0)
    assert (out_alpha - out_r0).abs().max() < 0.05  # soft mask n'est pas exactement 0


# ----------------------------------------------------------------
# Losses
# ----------------------------------------------------------------


def test_matriochka_rank_schedule_includes_R_max() -> None:
    ranks = matriochka_rank_schedule(R_max=16, n_samples=4, seed=0)
    assert 16 in ranks


def test_matriochka_weights_uniform_sums_to_one() -> None:
    ranks = [1, 2, 4, 8]
    weights = matriochka_weights(ranks, strategy="uniform")
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9


def test_loss_matriochka_smoke() -> None:
    cfg = ASPLayerConfig(d_model=8, R_max=4)
    layer = ASPLayer(cfg, backbone=LinearBackbone(8))
    x = torch.randn(2, 5, 8)
    y = torch.randn(2, 5, 8)

    def task_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (out - target).pow(2).mean()

    ranks = [1, 2, 4]
    weights = matriochka_weights(ranks)
    loss = loss_matriochka(
        asp_layer_forward=lambda xx, r: layer.forward_with_rank(xx, r),
        x=x, y_target=y, task_loss=task_loss, ranks=ranks, weights=weights,
    )
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_loss_consistency_smoke() -> None:
    cfg = ASPLayerConfig(d_model=8, R_max=4)
    layer = ASPLayer(cfg, backbone=LinearBackbone(8))
    x = torch.randn(2, 5, 8)
    loss = loss_consistency(
        asp_layer_forward=lambda xx, r: layer.forward_with_rank(xx, r),
        x=x, R_max=4, n_samples=3, delta=1, seed=0,
    )
    assert loss.item() >= 0
    assert torch.isfinite(loss)


def test_smoothness_check() -> None:
    qualities = [0.1, 0.3, 0.5, 0.7, 0.9]
    smooth, max_d = sanity_check_smoothness(qualities, max_jump=0.5)
    assert smooth
    qualities_bad = [0.1, 0.9, 0.1, 0.9]
    smooth_bad, _ = sanity_check_smoothness(qualities_bad, max_jump=0.5)
    assert not smooth_bad
