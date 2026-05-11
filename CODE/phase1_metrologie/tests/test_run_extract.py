"""Tests pour run_extract.py — driver V2 d'extraction des matrices A FP64.

Couvre les helpers purs et le pipeline end-to-end (mini Oracle, mini dataset)
sans dépendre de Hydra ni de MLflow réels.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.utils.data import Subset

from phase1_metrologie.oracle.extract import AttentionExtractor, ExtractorConfig
from phase1_metrologie.oracle.transformer import OracleConfig, OracleTransformer
from phase1_metrologie.run_extract import (
    _adaptive_batch_size,
    _expected_seq_len,
    _extract_bucket,
    _group_by_seq_len,
    _load_oracle,
    _log_bucket_metrics,
)
from phase1_metrologie.ssg.structure_mnist import (
    StructureMNISTConfig,
    StructureMNISTDataset,
    Vocab,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def vocab() -> Vocab:
    return Vocab(n_ops=2, n_noise=2)


@pytest.fixture
def mini_dataset(vocab: Vocab) -> StructureMNISTDataset:
    cfg = StructureMNISTConfig(
        omega=1, delta=2, entropy=0.0,
        n_examples=8, n_ops=2, n_noise=2, seed=0,
    )
    return StructureMNISTDataset(cfg)


@pytest.fixture
def mini_oracle(vocab: Vocab) -> tuple[OracleTransformer, OracleConfig]:
    model_cfg = OracleConfig(
        vocab_size=vocab.size, d_model=16, n_heads=2, n_layers=2,
        d_ff=32, max_seq_len=64, dropout=0.0, n_classes=10, pad_id=vocab.PAD,
    )
    model = OracleTransformer(model_cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model_cfg


# -----------------------------------------------------------------------------
# Helpers purs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("omega,delta", [(1, 2), (2, 16), (0, 0), (4, 64), (8, 16)])
def test_expected_seq_len_matches_ssg(omega: int, delta: int) -> None:
    """_expected_seq_len doit coller à StructureMNISTConfig.expected_seq_len()."""
    cfg = StructureMNISTConfig(
        omega=omega, delta=delta, entropy=0.0,
        n_examples=1, n_ops=2, n_noise=2, seed=0,
    )
    assert _expected_seq_len(omega=omega, delta=delta) == cfg.expected_seq_len()


def test_adaptive_batch_size_respects_cap() -> None:
    # Petite seq_len : on doit atteindre le cap
    b = _adaptive_batch_size(seq_len=64, L=6, H=8, target_gb=15.0, cap=8)
    assert b == 8

    # Grande seq_len : on doit descendre à 1 (clamp inférieur)
    b_huge = _adaptive_batch_size(seq_len=8000, L=6, H=8, target_gb=15.0, cap=8)
    assert b_huge == 1

    # seq_len=0 → 1 (garde-fou)
    assert _adaptive_batch_size(seq_len=0, L=6, H=8, target_gb=15.0, cap=8) == 1


# -----------------------------------------------------------------------------
# _group_by_seq_len
# -----------------------------------------------------------------------------


def test_group_by_seq_len_uniform(mini_dataset: StructureMNISTDataset) -> None:
    """Dataset uniforme (un seul ω/Δ) → un seul bucket."""
    groups = _group_by_seq_len(list(range(len(mini_dataset))), mini_dataset)
    assert len(groups) == 1
    seq_len = next(iter(groups.keys()))
    assert seq_len == mini_dataset.seq_len
    assert len(groups[seq_len]) == len(mini_dataset)


def test_group_by_seq_len_mixed(vocab: Vocab) -> None:
    """ConcatDataset de deux régimes différents → deux buckets."""
    from torch.utils.data import ConcatDataset
    cfg_a = StructureMNISTConfig(omega=1, delta=2, entropy=0.0,
                                  n_examples=4, n_ops=2, n_noise=2, seed=0)
    cfg_b = StructureMNISTConfig(omega=2, delta=4, entropy=0.0,
                                  n_examples=4, n_ops=2, n_noise=2, seed=1)
    ds_a = StructureMNISTDataset(cfg_a)
    ds_b = StructureMNISTDataset(cfg_b)
    concat = ConcatDataset([ds_a, ds_b])
    groups = _group_by_seq_len(list(range(len(concat))), concat)
    assert len(groups) == 2
    assert sum(len(v) for v in groups.values()) == len(concat)


# -----------------------------------------------------------------------------
# _load_oracle (strip prefix Fabric / torch.compile)
# -----------------------------------------------------------------------------


def test_load_oracle_strips_fabric_prefix(
    mini_oracle: tuple[OracleTransformer, OracleConfig],
    vocab: Vocab,
    tmp_path: Path,
) -> None:
    """Le checkpoint Fabric a `_forward_module.` en prefix — doit être strippé."""
    model, model_cfg = mini_oracle
    ckpt = tmp_path / "oracle.ckpt"
    sd = {f"_forward_module.{k}": v for k, v in model.state_dict().items()}
    torch.save({"model": sd}, ckpt)

    loaded = _load_oracle(str(ckpt), vocab, model_cfg, device="cpu")
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], loaded.state_dict()[k]), k


def test_load_oracle_strips_compile_prefix(
    mini_oracle: tuple[OracleTransformer, OracleConfig],
    vocab: Vocab,
    tmp_path: Path,
) -> None:
    """Prefix `_orig_mod.` (torch.compile) doit aussi être strippé."""
    model, model_cfg = mini_oracle
    ckpt = tmp_path / "oracle.ckpt"
    sd = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    torch.save(sd, ckpt)

    loaded = _load_oracle(str(ckpt), vocab, model_cfg, device="cpu")
    for k in model.state_dict():
        assert torch.equal(model.state_dict()[k], loaded.state_dict()[k]), k


# -----------------------------------------------------------------------------
# _extract_bucket end-to-end
# -----------------------------------------------------------------------------


def test_extract_bucket_shapes_dtypes(
    mini_oracle: tuple[OracleTransformer, OracleConfig],
    mini_dataset: StructureMNISTDataset,
    vocab: Vocab,
) -> None:
    """Vérifie shape (B,H,N,N) FP64 par couche + cohérence omegas/deltas/entropies."""
    model, model_cfg = mini_oracle
    extractor = AttentionExtractor(model, config=ExtractorConfig(fp64=True))
    subset = Subset(mini_dataset, list(range(len(mini_dataset))))
    seq_len = mini_dataset.seq_len

    dump = _extract_bucket(
        extractor, subset, pad_id=vocab.PAD, query_id=vocab.QUERY,
        batch_size=2, n_layers=model_cfg.n_layers,
    )
    assert len(dump["attn"]) == model_cfg.n_layers
    for A in dump["attn"]:
        assert A.shape == (len(mini_dataset), model_cfg.n_heads, seq_len, seq_len)
        assert A.dtype == torch.float64
    for key in ("omegas", "deltas", "entropies", "tokens"):
        assert dump[key].shape[0] == len(mini_dataset), key


# -----------------------------------------------------------------------------
# _log_bucket_metrics
# -----------------------------------------------------------------------------


def test_log_bucket_metrics_logs_two_per_layer() -> None:
    """L couches → 2L appels mlflow.log_metric (hankel + entropie)."""
    fake_attn = torch.eye(8).expand(2, 4, 8, 8).contiguous().to(torch.float64)
    dump = {
        "attn": [fake_attn, fake_attn, fake_attn],  # 3 couches
        "omegas": torch.tensor([2, 4]),
        "deltas": torch.tensor([16, 16]),
        "entropies": torch.tensor([0.0, 0.5]),
        "tokens": torch.zeros(2, 8, dtype=torch.long),
    }
    with mock.patch("phase1_metrologie.run_extract.mlflow.log_metric") as log:
        _log_bucket_metrics(dump, seq_len=8, hankel_tau=1e-3)
        assert log.call_count == 6, log.call_args_list
        # vérifie le préfixe des noms de métriques
        names = [call.args[0] for call in log.call_args_list]
        assert any("hankel_rank_seq8_layer0" in n for n in names)
        assert any("spectral_entropy_seq8_layer2" in n for n in names)
