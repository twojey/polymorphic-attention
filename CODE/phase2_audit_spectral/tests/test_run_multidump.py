"""Tests pour le driver phase 2 (run.py) — support multi-dump.

Couvre les helpers `_load_dumps` et `_validate_dumps` qui gèrent l'interface
entre les dumps multi-bucket produits par run_extract.py (phase 1 V2) et le
pipeline phase 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from phase2_audit_spectral.run import _load_dumps, _validate_dumps


def _fake_dump(*, B: int, L: int, H: int, N: int, omega: int, delta: int) -> dict:
    """Dump minimal au format attendu : attn list[L]((B,H,N,N)) + omegas/deltas/entropies."""
    return {
        "attn": [torch.randn(B, H, N, N).double() for _ in range(L)],
        "omegas": torch.full((B,), omega, dtype=torch.int64),
        "deltas": torch.full((B,), delta, dtype=torch.int64),
        "entropies": torch.zeros(B, dtype=torch.float32),
        "tokens": torch.zeros(B, N, dtype=torch.long),
    }


# -----------------------------------------------------------------------------
# _load_dumps : validation des args
# -----------------------------------------------------------------------------


def test_load_dumps_requires_one_source() -> None:
    with pytest.raises(SystemExit, match="Override requis"):
        _load_dumps(None, None)


def test_load_dumps_rejects_both_sources() -> None:
    with pytest.raises(SystemExit, match="Conflit"):
        _load_dumps("/some/path.pt", "/some/dir")


# -----------------------------------------------------------------------------
# _load_dumps : mode A (single file)
# -----------------------------------------------------------------------------


def test_load_dumps_single_file(tmp_path: Path) -> None:
    p = tmp_path / "dump.pt"
    torch.save(_fake_dump(B=4, L=3, H=2, N=8, omega=1, delta=16), p)
    dumps = _load_dumps(str(p), None)
    assert len(dumps) == 1
    assert dumps[0]["attn"][0].shape == (4, 2, 8, 8)


# -----------------------------------------------------------------------------
# _load_dumps : mode B (dossier multi-bucket)
# -----------------------------------------------------------------------------


def test_load_dumps_directory_numeric_sort(tmp_path: Path) -> None:
    """Le tri doit être numérique sur seq_len, pas lex (sinon 11 < 291 < 87)."""
    for n in [11, 87, 291]:
        torch.save(
            _fake_dump(B=2, L=3, H=2, N=n, omega=1, delta=16),
            tmp_path / f"audit_dump_seq{n}.pt",
        )
    dumps = _load_dumps(None, str(tmp_path))
    seq_lens = [d["attn"][0].size(2) for d in dumps]
    assert seq_lens == [11, 87, 291], (
        f"Tri attendu numérique [11,87,291], reçu {seq_lens}"
    )


def test_load_dumps_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Aucun dump"):
        _load_dumps(None, str(tmp_path))


def test_load_dumps_custom_glob(tmp_path: Path) -> None:
    """Glob personnalisé : on ignore les fichiers qui ne matchent pas."""
    torch.save(_fake_dump(B=2, L=3, H=2, N=8, omega=1, delta=16),
               tmp_path / "myprefix_42.pt")
    torch.save(_fake_dump(B=2, L=3, H=2, N=8, omega=1, delta=16),
               tmp_path / "noise.pt")
    dumps = _load_dumps(None, str(tmp_path), dumps_glob="myprefix_*.pt")
    assert len(dumps) == 1


# -----------------------------------------------------------------------------
# _validate_dumps : invariants cross-bucket
# -----------------------------------------------------------------------------


def test_validate_dumps_consistent() -> None:
    dumps = [
        _fake_dump(B=2, L=3, H=4, N=8, omega=1, delta=16),
        _fake_dump(B=3, L=3, H=4, N=16, omega=2, delta=32),
    ]
    L, H = _validate_dumps(dumps)
    assert L == 3 and H == 4


def test_validate_dumps_rejects_layer_mismatch() -> None:
    dumps = [
        _fake_dump(B=2, L=3, H=4, N=8, omega=1, delta=16),
        _fake_dump(B=2, L=5, H=4, N=16, omega=2, delta=32),
    ]
    with pytest.raises(SystemExit, match="couches"):
        _validate_dumps(dumps)


def test_validate_dumps_rejects_head_mismatch() -> None:
    dumps = [
        _fake_dump(B=2, L=3, H=4, N=8, omega=1, delta=16),
        _fake_dump(B=2, L=3, H=8, N=16, omega=2, delta=32),
    ]
    with pytest.raises(SystemExit, match="têtes"):
        _validate_dumps(dumps)
