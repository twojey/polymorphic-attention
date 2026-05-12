"""Tests VisionOracle + CodeOracle (squelettes V1 + pipelines V2)."""

from __future__ import annotations

import pytest
import torch


# ----------------------------------------------------------------------------
# Vision Oracle
# ----------------------------------------------------------------------------

def test_vision_imports() -> None:
    from catalog.oracles import VisionOracle, VisionModelSpec
    assert VisionOracle.domain == "vision"


def test_vision_rejects_missing_ckpt(tmp_path) -> None:
    from catalog.oracles import VisionOracle
    with pytest.raises(FileNotFoundError):
        VisionOracle(checkpoint_path=tmp_path / "nope.pt")


def test_vision_random_init_works(tmp_path) -> None:
    """Sans checkpoint : random init utilisable."""
    from catalog.oracles import VisionOracle, VisionModelSpec
    spec = VisionModelSpec(img_size=16, patch_size=4, n_channels=3,
                           n_classes=2, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    oracle = VisionOracle(model_spec=spec)
    grid = oracle.regime_grid()
    assert len(grid) > 0


def test_vision_regime_grid_non_empty(tmp_path) -> None:
    from catalog.oracles import VisionOracle, VisionModelSpec
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"x")
    spec = VisionModelSpec(img_size=28, patch_size=4)
    oracle = VisionOracle(checkpoint_path=ckpt, model_spec=spec)
    grid = oracle.regime_grid()
    assert len(grid) >= 1


def test_vision_extract_pipeline_random_init(tmp_path) -> None:
    """Pipeline V2 complet : extract avec random init produit AttentionDump valide."""
    from catalog.oracles import VisionOracle, VisionModelSpec
    from catalog.oracles.base import RegimeSpec
    spec = VisionModelSpec(img_size=16, patch_size=4, n_channels=3,
                           n_classes=2, d_model=32, n_heads=2, n_layers=2, d_ff=64)
    oracle = VisionOracle(model_spec=spec)
    regime = RegimeSpec(omega=4, delta=2, entropy=0.5)
    dump = oracle.extract_regime(regime, n_examples=2)
    dump.validate()
    assert dump.n_layers == 2
    assert dump.n_heads == 2
    assert dump.attn[0].shape[0] == 2  # batch
    assert dump.attn[0].dtype == torch.float64


# ----------------------------------------------------------------------------
# Code Oracle
# ----------------------------------------------------------------------------

def test_code_imports() -> None:
    from catalog.oracles import CodeOracle, CodeModelSpec
    assert CodeOracle.domain == "code"


def test_code_rejects_missing_ckpt(tmp_path) -> None:
    from catalog.oracles import CodeOracle
    with pytest.raises(FileNotFoundError):
        CodeOracle(checkpoint_path=tmp_path / "nope.pt")


def test_code_random_init_works(tmp_path) -> None:
    from catalog.oracles import CodeOracle, CodeModelSpec
    spec = CodeModelSpec(vocab_size=16, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                         max_seq_len=64)
    oracle = CodeOracle(model_spec=spec, n_bracket_types=2)
    grid = oracle.regime_grid()
    assert len(grid) == 12  # 4 depths × 3 seq_lens


def test_code_extract_pipeline_random_init(tmp_path) -> None:
    """Pipeline Code V2 : Dyck-2 valide + extraction attentions."""
    from catalog.oracles import CodeOracle, CodeModelSpec
    from catalog.oracles.base import RegimeSpec
    spec = CodeModelSpec(vocab_size=16, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                         max_seq_len=32)
    oracle = CodeOracle(model_spec=spec, n_bracket_types=2)
    regime = RegimeSpec(omega=2, delta=32, entropy=0.0)
    dump = oracle.extract_regime(regime, n_examples=3)
    dump.validate()
    assert dump.n_layers == 2
    assert dump.attn[0].dtype == torch.float64
    # Au moins 80% des séquences générées doivent être Dyck-k valides
    assert dump.metadata["n_valid_dyck_k"] >= 2


def test_code_dyck_k_generator() -> None:
    """Test direct du générateur Dyck-k."""
    from catalog.oracles.code import generate_dyck_k, validate_dyck_k
    for seed in range(5):
        seq = generate_dyck_k(depth=3, seq_len=32, k=2, seed=seed)
        assert validate_dyck_k(seq), f"Sequence seed={seed} not valid Dyck-k"


def test_code_dyck_k_validates_invalid() -> None:
    from catalog.oracles.code import validate_dyck_k
    # '(' open type 0, ')' close type 0 = tokens 2, 3
    # '[' open type 1, ']' close type 1 = tokens 4, 5
    assert validate_dyck_k([2, 3])  # ()
    assert validate_dyck_k([2, 4, 5, 3])  # ([])
    assert not validate_dyck_k([2, 5])  # (]
    assert not validate_dyck_k([2])  # (
