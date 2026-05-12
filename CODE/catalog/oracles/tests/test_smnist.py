"""Tests SMNISTOracle — smoke + interface.

NB : tests d'extraction réelle exigent le checkpoint Oracle local
(OPS/checkpoints/oracle_e2f0b5e.ckpt). Skipif si absent pour permettre
CI sans le ckpt.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from catalog.oracles import RegimeSpec


CKPT = Path(__file__).resolve().parents[4] / "OPS/checkpoints/oracle_e2f0b5e.ckpt"
_HAS_CKPT = CKPT.is_file()


@pytest.mark.skipif(not _HAS_CKPT, reason=f"checkpoint absent : {CKPT}")
def test_smnist_oracle_loads() -> None:
    from catalog.oracles.smnist import SMNISTOracle
    oracle = SMNISTOracle(checkpoint_path=CKPT, device="cpu")
    assert oracle.oracle_id.startswith("smnist_")
    assert oracle.domain == "smnist"
    assert oracle.n_layers == 6
    assert oracle.n_heads == 8


@pytest.mark.skipif(not _HAS_CKPT, reason=f"checkpoint absent : {CKPT}")
def test_smnist_oracle_regime_grid_non_empty() -> None:
    from catalog.oracles.smnist import SMNISTOracle
    oracle = SMNISTOracle(checkpoint_path=CKPT, device="cpu")
    grid = oracle.regime_grid()
    assert len(grid) >= 5  # 6 omega + 3 delta (Δ=16 dédupliqué)
    # Tous ont omega + delta définis
    for r in grid:
        assert r.omega is not None
        assert r.delta is not None


@pytest.mark.skipif(not _HAS_CKPT, reason=f"checkpoint absent : {CKPT}")
def test_smnist_oracle_extract_tiny_regime() -> None:
    """Smoke : extrait 2 examples pour le régime ω=0 Δ=16 (seq_len=19, rapide)."""
    from catalog.oracles.smnist import SMNISTOracle
    oracle = SMNISTOracle(checkpoint_path=CKPT, device="cpu")
    regime = RegimeSpec(omega=0, delta=16, entropy=0.0)
    dump = oracle.extract_regime(regime, n_examples=2)
    dump.validate()
    assert dump.n_examples == 2
    assert dump.n_layers == 6
    assert dump.n_heads == 8
    # ω=0 Δ=16 → seq_len = (2*16+2)*0 + 16 + 3 = 19
    assert dump.seq_len == 19
    # Attention softmax : lignes somment à 1
    A = dump.attn[0]
    sums = A.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_smnist_oracle_rejects_missing_checkpoint() -> None:
    from catalog.oracles.smnist import SMNISTOracle
    with pytest.raises(FileNotFoundError):
        SMNISTOracle(checkpoint_path="/nonexistent/oracle.ckpt")


def test_smnist_oracle_rejects_incomplete_regime() -> None:
    if not _HAS_CKPT:
        pytest.skip("checkpoint absent")
    from catalog.oracles.smnist import SMNISTOracle
    oracle = SMNISTOracle(checkpoint_path=CKPT, device="cpu")
    with pytest.raises(ValueError, match="omega et regime.delta"):
        oracle.extract_regime(RegimeSpec(omega=None, delta=None), n_examples=2)
