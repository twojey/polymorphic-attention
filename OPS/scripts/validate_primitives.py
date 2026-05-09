"""
validate_primitives.py — Stage 0.2 du projet ASP.

Exécute les checks listés dans OPS/env/STACK.md § Primitives à valider.
Sortie : table Markdown stdout + JSON sur disque dans
OPS/env/primitives_results.json (à reporter dans PRIMITIVES.md).

Usage :
    PYTHONPATH=CODE uv run python OPS/scripts/validate_primitives.py

À exécuter sur le pod RunPod RTX 5090 après setup_env.sh. Le script
détecte CUDA et adapte ses tests ; en mode CPU il valide la sémantique
mais pas les performances.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class CheckResult:
    name: str
    passed: bool
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _run(name: str, fn: Callable[[], dict[str, Any]]) -> CheckResult:
    t0 = time.perf_counter()
    try:
        details = fn()
        return CheckResult(name=name, passed=True, duration_s=time.perf_counter() - t0, details=details)
    except Exception as e:
        return CheckResult(
            name=name,
            passed=False,
            duration_s=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )


def check_svd_fp64_batched(device: torch.device) -> dict[str, Any]:
    # Phase 2.1 : SVD batchée FP64 sur matrices d'attention. Test sur N petit
    # mais représentatif (batch=8, N=256). Le pod testera N=2¹² séparément.
    B, N = 8, 256
    a = torch.randn(B, N, N, dtype=torch.float64, device=device)
    a = torch.softmax(a, dim=-1)  # ressemble à une matrice d'attention
    U, S, Vh = torch.linalg.svd(a, full_matrices=False)
    reconstructed = U @ torch.diag_embed(S) @ Vh
    err = (a - reconstructed).abs().max().item()
    return {"batch": B, "N": N, "max_recon_err": err, "S_dtype": str(S.dtype)}


def check_svd_lowrank_randomized(device: torch.device) -> dict[str, Any]:
    # Phase 1.5 : S_Spectral via SVD partielle randomisée sur fenêtre K.
    K = 128
    q = 16
    a = torch.randn(K, K, dtype=torch.float32, device=device)
    a = torch.softmax(a, dim=-1)
    U, S, V = torch.svd_lowrank(a, q=q)
    return {"K": K, "q": q, "S_top": float(S[0]), "S_bot": float(S[-1])}


def check_fft_rfft_long(device: torch.device) -> dict[str, Any]:
    # Phase 3 : convolution causale FFT-based sur longues séquences.
    N = 8192  # phase 5 visera 2¹⁶, on échelonne
    x = torch.randn(4, N, dtype=torch.float32, device=device)
    X = torch.fft.rfft(x, dim=-1)
    x_back = torch.fft.irfft(X, n=N, dim=-1)
    err = (x - x_back).abs().max().item()
    return {"N": N, "X_shape": tuple(X.shape), "max_inverse_err": err}


def check_lstsq_batched(device: torch.device) -> dict[str, Any]:
    # Phase 2.6b batterie A : fitting ε_C par classe via lstsq batché.
    B, M, N = 16, 64, 32
    A = torch.randn(B, M, N, dtype=torch.float64, device=device)
    b = torch.randn(B, M, 1, dtype=torch.float64, device=device)
    sol = torch.linalg.lstsq(A, b)
    return {"batch": B, "M": M, "N": N, "solution_shape": tuple(sol.solution.shape)}


def check_vmap(device: torch.device) -> dict[str, Any]:
    # Phase 2 batterie : vmap sur des opérations matricielles (utile pour
    # parallélisation par tête × couche).
    from torch.func import vmap

    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x)

    x = torch.randn(32, 16, 16, device=device)
    norms = vmap(fn)(x)
    return {"input_shape": tuple(x.shape), "output_shape": tuple(norms.shape)}


def check_dense_attention_export(device: torch.device) -> dict[str, Any]:
    # Phase 1 : extraction de la matrice A complète, par (couche, tête, exemple).
    # Vérifie que (a) on peut exécuter une attention dense pure, (b) la matrice
    # softmax(QK^T / √d) est récupérable, (c) le coût mémoire tient.
    B, H, N, d = 4, 8, 256, 64
    Q = torch.randn(B, H, N, d, device=device)
    K = torch.randn(B, H, N, d, device=device)
    V = torch.randn(B, H, N, d, device=device)
    scores = (Q @ K.transpose(-1, -2)) / (d**0.5)
    A = torch.softmax(scores, dim=-1)
    out = A @ V
    A_fp64 = A.to(torch.float64)  # cast pour SVD aval
    bytes_attn = A.numel() * A.element_size()
    return {
        "B": B,
        "H": H,
        "N": N,
        "d": d,
        "out_shape": tuple(out.shape),
        "A_shape": tuple(A.shape),
        "A_bytes_mb": bytes_attn / (1024**2),
        "A_fp64_bytes_mb": A_fp64.numel() * A_fp64.element_size() / (1024**2),
    }


def system_info() -> dict[str, Any]:
    import os

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
        info["device_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Détection Blackwell (sm_120 = capability 12.0)
        info["is_blackwell"] = info["device_capability"] == (12, 0)

    # ENV vars Blackwell — issues du retour Lumis sur RTX 5090
    expected_env = {
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_MODULE_LOADING": "LAZY",
        "MAX_JOBS": "4",
    }
    info["env_vars"] = {k: os.environ.get(k) for k in expected_env}
    info["env_vars_missing"] = [k for k, v in expected_env.items() if os.environ.get(k) != v]
    return info


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys = system_info()
    print("# Stage 0.2 — Validation des primitives\n")
    print("## Système\n")
    for k, v in sys.items():
        print(f"- **{k}** : `{v}`")
    print()

    if sys.get("env_vars_missing"):
        print(
            f"⚠ ENV vars Blackwell manquantes ou incorrectes : {sys['env_vars_missing']}.\n"
            f"   Lance `bash OPS/scripts/setup_env.sh` AVANT ce script pour les set.\n"
        )
    if sys.get("cuda_available") and not sys.get("is_blackwell"):
        print(
            f"⚠ Compute capability {sys.get('device_capability')} ≠ (12, 0). "
            f"Ce n'est pas une 5090 — ASP cible Blackwell pour validation finale.\n"
        )

    checks = [
        ("svd_fp64_batched", check_svd_fp64_batched),
        ("svd_lowrank_randomized", check_svd_lowrank_randomized),
        ("fft_rfft_long", check_fft_rfft_long),
        ("lstsq_batched", check_lstsq_batched),
        ("vmap", check_vmap),
        ("dense_attention_export", check_dense_attention_export),
    ]

    results: list[CheckResult] = []
    for name, fn in checks:
        print(f"## Check : {name}\n")
        result = _run(name, lambda f=fn: f(device))
        results.append(result)
        if result.passed:
            print(f"- **passed** en {result.duration_s:.3f}s")
            for k, v in result.details.items():
                print(f"- {k} : `{v}`")
        else:
            print(f"- **FAILED** en {result.duration_s:.3f}s")
            print(f"- erreur : `{result.error}`")
        print()

    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    print(f"## Verdict : {n_passed}/{n_total} checks passés.\n")

    out = {
        "system": sys,
        "results": [asdict(r) for r in results],
        "summary": {"passed": n_passed, "total": n_total},
    }
    out_path = Path(__file__).resolve().parent.parent / "env" / "primitives_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Résultats JSON écrits dans : {out_path}")

    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
