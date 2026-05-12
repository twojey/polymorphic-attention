"""Tests pour SyntheticOracle + interface AbstractOracle."""

from __future__ import annotations

import pytest
import torch

from catalog.oracles import AbstractOracle, AttentionDump, RegimeSpec, SyntheticOracle


# -----------------------------------------------------------------------------
# RegimeSpec
# -----------------------------------------------------------------------------


def test_regime_spec_hashable() -> None:
    r1 = RegimeSpec(omega=2, delta=16, entropy=0.0)
    r2 = RegimeSpec(omega=2, delta=16, entropy=0.0)
    r3 = RegimeSpec(omega=4, delta=16, entropy=0.0)
    s = {r1, r2, r3}
    assert len(s) == 2


def test_regime_spec_custom_field() -> None:
    r = RegimeSpec(omega=0, delta=0, custom={"bucket": "seq_512"})
    assert hash(r)  # ne plante pas


# -----------------------------------------------------------------------------
# AttentionDump
# -----------------------------------------------------------------------------


def test_attention_dump_basic_shape_props() -> None:
    attn = [torch.randn(4, 2, 8, 8, dtype=torch.float64) for _ in range(3)]
    omegas = torch.zeros(4, dtype=torch.int64)
    deltas = torch.full((4,), 16, dtype=torch.int64)
    entropies = torch.zeros(4, dtype=torch.float32)
    dump = AttentionDump(attn=attn, omegas=omegas, deltas=deltas, entropies=entropies)
    assert dump.n_layers == 3
    assert dump.n_heads == 2
    assert dump.n_examples == 4
    assert dump.seq_len == 8


def test_attention_dump_validate_passes_on_consistent() -> None:
    attn = [torch.randn(4, 2, 8, 8) for _ in range(3)]
    dump = AttentionDump(
        attn=attn,
        omegas=torch.zeros(4), deltas=torch.zeros(4), entropies=torch.zeros(4),
    )
    dump.validate()  # ne lève pas


def test_attention_dump_validate_rejects_inconsistent_layer_shape() -> None:
    attn = [torch.randn(4, 2, 8, 8), torch.randn(4, 2, 9, 9)]
    dump = AttentionDump(
        attn=attn,
        omegas=torch.zeros(4), deltas=torch.zeros(4), entropies=torch.zeros(4),
    )
    with pytest.raises(ValueError, match="layer"):
        dump.validate()


def test_attention_dump_validate_rejects_size_mismatch() -> None:
    attn = [torch.randn(4, 2, 8, 8)]
    dump = AttentionDump(
        attn=attn,
        omegas=torch.zeros(5), deltas=torch.zeros(4), entropies=torch.zeros(4),
    )
    with pytest.raises(ValueError, match="omegas"):
        dump.validate()


# -----------------------------------------------------------------------------
# SyntheticOracle
# -----------------------------------------------------------------------------


def test_synthetic_oracle_random_structure() -> None:
    oracle = SyntheticOracle(structure="random", seq_len=16, n_layers=2, n_heads=4, seed=42)
    dump = oracle.extract_regime(RegimeSpec(omega=2, delta=16, entropy=0.0), n_examples=8)
    dump.validate()
    assert dump.n_examples == 8
    assert dump.n_layers == 2
    assert dump.n_heads == 4
    assert dump.seq_len == 16
    # Softmax → lignes somment à 1
    assert torch.allclose(dump.attn[0].sum(dim=-1), torch.ones_like(dump.attn[0].sum(dim=-1)), atol=1e-5)


def test_synthetic_oracle_low_rank_structure() -> None:
    """Structure low_rank produit attention de rang ≤ target_rank (approx softmax)."""
    oracle = SyntheticOracle(structure="low_rank", target_rank=2, seq_len=16, seed=0)
    dump = oracle.extract_regime(RegimeSpec(omega=2, delta=16), n_examples=4)
    # SVD : on devrait observer un rang effectif faible
    A = dump.attn[0][0, 0]  # (N, N)
    s = torch.linalg.svdvals(A.double())
    # Énergie concentrée sur les premières valeurs singulières
    cumsum_ratio = (s ** 2).cumsum(0) / (s ** 2).sum()
    # Au rang 8 (= target_rank * 4), on capture déjà > 99 %
    assert cumsum_ratio[7] > 0.99


def test_synthetic_oracle_toeplitz_structure() -> None:
    """Structure toeplitz produit attention quasi-Toeplitz (ε_T < ε_T(random))."""
    from catalog.projectors import Toeplitz
    proj = Toeplitz()

    rand_oracle = SyntheticOracle(structure="random", seq_len=12, seed=0)
    rand_dump = rand_oracle.extract_regime(RegimeSpec(omega=2, delta=16), n_examples=2)
    rand_eps = proj.epsilon(rand_dump.attn[0]).mean().item()

    toep_oracle = SyntheticOracle(structure="toeplitz", seq_len=12, seed=0)
    toep_dump = toep_oracle.extract_regime(RegimeSpec(omega=2, delta=16), n_examples=2)
    toep_eps = proj.epsilon(toep_dump.attn[0]).mean().item()

    assert toep_eps < rand_eps, f"toeplitz ε={toep_eps} should be < random ε={rand_eps}"


def test_synthetic_oracle_deterministic_seed() -> None:
    o1 = SyntheticOracle(structure="random", seq_len=8, seed=42)
    o2 = SyntheticOracle(structure="random", seq_len=8, seed=42)
    d1 = o1.extract_regime(RegimeSpec(omega=2, delta=16), n_examples=4)
    d2 = o2.extract_regime(RegimeSpec(omega=2, delta=16), n_examples=4)
    assert torch.allclose(d1.attn[0], d2.attn[0])


def test_synthetic_oracle_regime_grid_non_empty() -> None:
    oracle = SyntheticOracle()
    grid = oracle.regime_grid()
    assert len(grid) >= 1
    assert all(isinstance(r, RegimeSpec) for r in grid)


def test_synthetic_oracle_unknown_structure_raises() -> None:
    oracle = SyntheticOracle(structure="not_a_thing")
    with pytest.raises(ValueError, match="structure inconnue"):
        oracle.extract_regime(RegimeSpec(omega=0, delta=0), n_examples=2)
