"""Tests pour ASPTransformer et backbones concrets phase 3."""

from __future__ import annotations

import pytest
import torch

from phase3_kernel_asp.backbone_concrete import (
    CompositeBackbone,
    HankelSSMBackbone,
    ToeplitzConvBackbone,
    build_backbone_from_class,
)
from phase3_kernel_asp.transformer import ASPTransformer, ASPTransformerConfig


# -----------------------------------------------------------------------------
# Backbones concrets
# -----------------------------------------------------------------------------


def test_toeplitz_backbone_shape() -> None:
    bb = ToeplitzConvBackbone(d_model=16, kernel_len=8, n_heads=1)
    x = torch.randn(3, 12, 16)
    y = bb(x)
    assert y.shape == x.shape
    assert bb.class_name == "toeplitz"


def test_toeplitz_backbone_multihead() -> None:
    bb = ToeplitzConvBackbone(d_model=16, kernel_len=8, n_heads=4)
    x = torch.randn(2, 10, 16)
    y = bb(x)
    assert y.shape == x.shape


def test_toeplitz_backbone_causality() -> None:
    """Output à t doit dépendre seulement de x[:t+1] (causalité)."""
    bb = ToeplitzConvBackbone(d_model=8, kernel_len=4, n_heads=1)
    x1 = torch.randn(1, 16, 8)
    x2 = x1.clone()
    x2[:, 8:] = torch.randn_like(x2[:, 8:])  # change future
    y1 = bb(x1)
    y2 = bb(x2)
    # Le passé (t < 8) doit être inchangé
    assert torch.allclose(y1[:, :8], y2[:, :8], atol=1e-5), \
        "ToeplitzConvBackbone n'est pas causal"


def test_hankel_backbone_shape() -> None:
    bb = HankelSSMBackbone(d_model=16, state_size=4)
    x = torch.randn(2, 12, 16)
    y = bb(x)
    assert y.shape == x.shape
    assert bb.class_name == "hankel"


def test_hankel_backbone_stability() -> None:
    """A = sigmoid(A_log) ∈ (0, 1) → la récurrence reste bornée."""
    bb = HankelSSMBackbone(d_model=8, state_size=4)
    # init A à valeurs hautes : A_log = +5 → sigmoid → ~1, encore stable
    with torch.no_grad():
        bb.A_log.fill_(5.0)
    x = torch.ones(1, 100, 8)
    y = bb(x)
    assert not torch.isnan(y).any()
    assert torch.isfinite(y).all()


def test_composite_backbone() -> None:
    bb1 = ToeplitzConvBackbone(d_model=8, kernel_len=4)
    bb2 = HankelSSMBackbone(d_model=8, state_size=2)
    comp = CompositeBackbone([bb1, bb2])
    x = torch.randn(2, 8, 8)
    y = comp(x)
    assert y.shape == x.shape
    assert "toeplitz" in comp.class_name and "hankel" in comp.class_name


def test_build_backbone_from_class_known() -> None:
    for cls in ("toeplitz", "hankel", "identity", "linear"):
        bb = build_backbone_from_class(cls, d_model=8)
        x = torch.randn(2, 4, 8)
        y = bb(x)
        assert y.shape == x.shape


def test_build_backbone_unknown_raises() -> None:
    with pytest.raises(ValueError, match="class_name inconnu"):
        build_backbone_from_class("unknown_class", d_model=8)


def test_build_backbone_composite() -> None:
    bb = build_backbone_from_class("toeplitz+hankel", d_model=8)
    assert isinstance(bb, CompositeBackbone)
    x = torch.randn(1, 4, 8)
    y = bb(x)
    assert y.shape == x.shape


# -----------------------------------------------------------------------------
# ASPTransformer
# -----------------------------------------------------------------------------


def _small_cfg() -> ASPTransformerConfig:
    return ASPTransformerConfig(
        vocab_size=24, d_model=32, n_heads=4, n_layers=2, d_ff=64,
        max_seq_len=128, R_max=8,
    )


def test_asp_transformer_forward_full_rank() -> None:
    cfg = _small_cfg()
    model = ASPTransformer(cfg)
    tokens = torch.randint(1, 20, (3, 16), dtype=torch.long)
    qpos = torch.zeros(3, dtype=torch.long)
    out = model(tokens, qpos)
    assert out.shape == (3, cfg.n_classes)


def test_asp_transformer_forward_at_rank_zero() -> None:
    """À r=0, l'ASPLayer ≡ Backbone (Identity ici), donc ≠ NaN."""
    cfg = _small_cfg()
    model = ASPTransformer(cfg)
    tokens = torch.randint(1, 20, (2, 10), dtype=torch.long)
    qpos = torch.zeros(2, dtype=torch.long)
    out = model.forward_at_rank(tokens, qpos, r=0)
    assert out.shape == (2, cfg.n_classes)
    assert not torch.isnan(out).any()


def test_asp_transformer_monotonic_quality_no_train() -> None:
    """Sans entraînement, le forward à r croissant n'a pas de raison d'être
    monotone. On vérifie juste que ça ne plante pas et que les sorties
    diffèrent.
    """
    cfg = _small_cfg()
    model = ASPTransformer(cfg)
    tokens = torch.randint(1, 20, (2, 12), dtype=torch.long)
    qpos = torch.zeros(2, dtype=torch.long)
    outs = [model.forward_at_rank(tokens, qpos, r=r) for r in (0, 2, 4, 8)]
    for o in outs:
        assert o.shape == (2, cfg.n_classes)
    # Les sorties à r différents doivent différer (sinon ΔAttn ne fait rien)
    diff_max = (outs[3] - outs[0]).abs().max().item()
    assert diff_max > 1e-6


def test_asp_transformer_with_toeplitz_backbone() -> None:
    cfg = _small_cfg()
    cfg.backbone_class = "toeplitz"
    cfg.backbone_params = {"kernel_len": 8, "n_heads": 1}
    model = ASPTransformer(cfg)
    tokens = torch.randint(1, 20, (2, 16), dtype=torch.long)
    qpos = torch.zeros(2, dtype=torch.long)
    out = model(tokens, qpos)
    assert out.shape == (2, cfg.n_classes)


def test_asp_transformer_backward() -> None:
    """Test backward pour vérifier que les gradients passent à travers
    ASPLayer + Matriochka + Backbone."""
    cfg = _small_cfg()
    cfg.backbone_class = "linear"  # plus simple à entraîner
    model = ASPTransformer(cfg)
    tokens = torch.randint(1, 20, (2, 8), dtype=torch.long)
    qpos = torch.zeros(2, dtype=torch.long)
    out = model(tokens, qpos)
    loss = out.sum()
    loss.backward()
    # Au moins un param a un gradient non nul
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)
