"""
machine.py — MachineProfile : point d'autorité unique sur la politique perf.

Spec : DOC/CONTEXT.md §MachineProfile.

Auto-détecte :
- `device` : cuda / cpu
- `gpu_arch` : Blackwell sm_120 / Hopper / Ampere / older / cpu
- `precision_svd` : fp32 (consumer Blackwell où fp64=1/64 du fp32) ou fp64
- `n_blas_threads` : 1 si fork multiprocessing prévu, sinon all-cores
- `batch_cap` : selon VRAM / RAM disponible

Politique précision SVD :
- Blackwell consumer (sm_120) : fp64 nerfé 1/64, mieux d'utiliser fp32 GPU
  ou fp64 CPU. Par défaut → fp32 GPU si cuda dispo, sinon fp64 CPU.
- Hopper, Ampere, etc. : fp64 GPU OK.

Consommé par :
- Properties (compute(A, ctx) où ctx.device, ctx.dtype viennent du profile)
- Drivers (au lieu de `cuda.is_available()` inline)
- Setup scripts (validation cohérence machine ↔ config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import torch


class GpuArch(str, Enum):
    """Architecture GPU détectée. Détermine la politique précision SVD."""
    BLACKWELL_CONSUMER = "blackwell_consumer"   # sm_120 (RTX 5090) — fp64 ≈ 1/64 du fp32
    BLACKWELL_DATACENTER = "blackwell_datacenter"  # sm_100 (B100/B200) — fp64 OK
    HOPPER = "hopper"                            # sm_90 (H100) — fp64 OK
    ADA = "ada"                                  # sm_89 (RTX 4090) — fp64 1/64
    AMPERE = "ampere"                            # sm_80/86 — fp64 OK serveur, 1/32 consumer
    OLDER = "older"                              # sm_75 et inférieur
    CPU_ONLY = "cpu_only"
    UNKNOWN = "unknown"


PrecisionPolicy = Literal["fp32", "fp64"]
DevicePolicy = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class MachineProfile:
    """Politique perf de la machine, dérivée d'une auto-détection ou injectée.

    Frozen : créer un nouveau profile pour ré-évaluer. Évite les modifications
    silencieuses entre Properties.
    """

    device: DevicePolicy
    gpu_arch: GpuArch
    precision_svd: PrecisionPolicy
    n_blas_threads: int
    batch_cap: int
    gpu_name: str = ""
    vram_gb: float = 0.0
    ram_gb: float = 0.0

    @property
    def dtype_svd(self) -> torch.dtype:
        return torch.float32 if self.precision_svd == "fp32" else torch.float64

    @classmethod
    def detect(cls, *, force_blas_single: bool = True) -> "MachineProfile":
        """Auto-détection. force_blas_single=True quand fork multiprocessing
        est prévu (cf. compute_s_spectral qui forke un pool avec eigvalsh)."""
        device, arch, gpu_name, vram_gb = _detect_gpu()
        precision = _pick_precision_for_arch(arch, device)
        ram_gb = _detect_ram_gb()
        n_threads = 1 if force_blas_single else max(1, (os.cpu_count() or 1) - 1)
        batch_cap = _suggest_batch_cap(device, vram_gb, ram_gb)

        return cls(
            device=device,
            gpu_arch=arch,
            precision_svd=precision,
            n_blas_threads=n_threads,
            batch_cap=batch_cap,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
        )

    @classmethod
    def fake(
        cls,
        *,
        device: DevicePolicy = "cpu",
        gpu_arch: GpuArch = GpuArch.CPU_ONLY,
        precision_svd: PrecisionPolicy = "fp64",
        n_blas_threads: int = 1,
        batch_cap: int = 8,
    ) -> "MachineProfile":
        """Profile factice pour tests. Permet de simuler n'importe quelle config."""
        return cls(
            device=device,
            gpu_arch=gpu_arch,
            precision_svd=precision_svd,
            n_blas_threads=n_blas_threads,
            batch_cap=batch_cap,
            gpu_name="fake",
            vram_gb=0.0,
            ram_gb=8.0,
        )

    def apply_blas_env(self) -> None:
        """Pose les variables d'env BLAS pour cohérence multiprocessing.

        Idempotent : peut être appelé plusieurs fois. À appeler AVANT
        d'importer torch dans une boucle worker.
        """
        n = str(self.n_blas_threads)
        for var in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = n

    def summary(self) -> str:
        """Ligne lisible pour log de démarrage."""
        return (
            f"MachineProfile(device={self.device}, arch={self.gpu_arch.value}, "
            f"precision_svd={self.precision_svd}, blas_threads={self.n_blas_threads}, "
            f"batch_cap={self.batch_cap}, vram={self.vram_gb:.1f} GB, "
            f"ram={self.ram_gb:.1f} GB)"
        )


def _detect_gpu() -> tuple[DevicePolicy, GpuArch, str, float]:
    """Retourne (device, arch, gpu_name, vram_gb)."""
    if not torch.cuda.is_available():
        return "cpu", GpuArch.CPU_ONLY, "", 0.0

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    major, minor = capability

    # Mapping compute capability → architecture
    # https://developer.nvidia.com/cuda-gpus
    arch = GpuArch.UNKNOWN
    if major == 12:
        # sm_120 = Blackwell consumer (RTX 5090, 5080)
        if "5090" in name or "5080" in name or "5070" in name:
            arch = GpuArch.BLACKWELL_CONSUMER
        else:
            arch = GpuArch.BLACKWELL_DATACENTER
    elif major == 10:
        arch = GpuArch.BLACKWELL_DATACENTER  # sm_100 = B100/B200
    elif major == 9:
        arch = GpuArch.HOPPER
    elif major == 8:
        if minor == 9:
            arch = GpuArch.ADA  # sm_89 = RTX 40xx
        else:
            arch = GpuArch.AMPERE
    elif major <= 7:
        arch = GpuArch.OLDER

    return "cuda", arch, name, vram_gb


def _pick_precision_for_arch(arch: GpuArch, device: DevicePolicy) -> PrecisionPolicy:
    """Politique précision SVD : FP32 sur GPU consumer où FP64 est nerfé.

    Spec stricte DOC/01 §8.4 = FP64. Mais sur Blackwell consumer FP64 ≈ 1/64
    du FP32, donc on bascule sur FP32 GPU (r_eff ordre des σ, FP32 suffit).
    Documenter dans le rapport quand on sort de la spec stricte.
    """
    if device == "cpu":
        return "fp64"
    if arch in (GpuArch.BLACKWELL_CONSUMER, GpuArch.ADA):
        return "fp32"
    return "fp64"


def _detect_ram_gb() -> float:
    """RAM système totale en GB, via /proc/meminfo (Linux). 0.0 si indéterminable."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except FileNotFoundError:
        pass
    return 0.0


def _suggest_batch_cap(
    device: DevicePolicy, vram_gb: float, ram_gb: float
) -> int:
    """Heuristique batch_cap conservatrice selon mémoire disponible.

    Pour attention dense FP64 (B, H=8, N, N) ≈ B × 8 × N² × 8 bytes.
    On laisse marge ×3 pour activations + grad si training.
    """
    if device == "cuda":
        # GPU : 5% VRAM par exemple type seq_len=1000 batch=1 ≈ 64 MB
        if vram_gb >= 30:
            return 32
        if vram_gb >= 16:
            return 16
        if vram_gb >= 8:
            return 8
        return 4
    # CPU : RAM système
    if ram_gb >= 64:
        return 16
    if ram_gb >= 32:
        return 8
    return 4
