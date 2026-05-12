"""Tests pour infra/machine.py MachineProfile."""

from __future__ import annotations

import os

import pytest
import torch

from infra.machine import GpuArch, MachineProfile, _pick_precision_for_arch


def test_detect_returns_valid_profile() -> None:
    """detect() retourne toujours un MachineProfile cohérent."""
    p = MachineProfile.detect()
    assert p.device in ("cpu", "cuda")
    assert isinstance(p.gpu_arch, GpuArch)
    assert p.precision_svd in ("fp32", "fp64")
    assert p.batch_cap >= 1
    assert p.n_blas_threads >= 1


def test_fake_profile_for_tests() -> None:
    p = MachineProfile.fake(
        device="cuda",
        gpu_arch=GpuArch.BLACKWELL_CONSUMER,
        precision_svd="fp32",
        batch_cap=16,
    )
    assert p.device == "cuda"
    assert p.gpu_arch == GpuArch.BLACKWELL_CONSUMER
    assert p.precision_svd == "fp32"
    assert p.batch_cap == 16


def test_dtype_svd_mapping() -> None:
    p32 = MachineProfile.fake(precision_svd="fp32")
    p64 = MachineProfile.fake(precision_svd="fp64")
    assert p32.dtype_svd == torch.float32
    assert p64.dtype_svd == torch.float64


def test_precision_pick_blackwell_consumer_fp32() -> None:
    p = _pick_precision_for_arch(GpuArch.BLACKWELL_CONSUMER, "cuda")
    assert p == "fp32"


def test_precision_pick_ada_consumer_fp32() -> None:
    p = _pick_precision_for_arch(GpuArch.ADA, "cuda")
    assert p == "fp32"


def test_precision_pick_hopper_fp64() -> None:
    p = _pick_precision_for_arch(GpuArch.HOPPER, "cuda")
    assert p == "fp64"


def test_precision_pick_cpu_always_fp64() -> None:
    for arch in (GpuArch.CPU_ONLY, GpuArch.BLACKWELL_CONSUMER, GpuArch.HOPPER):
        p = _pick_precision_for_arch(arch, "cpu")
        assert p == "fp64"


def test_apply_blas_env_sets_all_vars() -> None:
    p = MachineProfile.fake(n_blas_threads=2)
    # Sauvegarde l'état avant
    saved = {v: os.environ.get(v) for v in
             ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    try:
        p.apply_blas_env()
        for v in saved:
            assert os.environ[v] == "2"
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def test_profile_frozen() -> None:
    """MachineProfile est frozen (immutable) — empêche modification silencieuse."""
    p = MachineProfile.fake()
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.device = "cuda"  # type: ignore[misc]


def test_summary_contains_key_info() -> None:
    p = MachineProfile.fake(
        device="cuda",
        gpu_arch=GpuArch.BLACKWELL_CONSUMER,
        precision_svd="fp32",
        batch_cap=16,
    )
    s = p.summary()
    assert "cuda" in s
    assert "blackwell_consumer" in s
    assert "fp32" in s
