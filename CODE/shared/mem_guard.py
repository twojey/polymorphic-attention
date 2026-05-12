"""
mem_guard.py — surveillance RAM disponible avec abort gracieux.

Spec : feedback `feedback_script_robustness.md` — pas de crash silencieux.

Utilisé par Sprint B/C et Battery pour :
- Vérifier la RAM disponible avant une opération coûteuse (extract dump,
  SVD sur matrice large).
- Lever MemoryGuardAbort si la RAM tombe sous un seuil critique, avant
  que le kernel OOM-killer ne tue le process silencieusement.
- Logger l'usage mémoire à chaque check pour comprendre où la
  consommation grimpe.

Fonctionne sans dépendance externe : lit /proc/meminfo si psutil absent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


class MemoryGuardAbort(RuntimeError):
    """Levée quand la RAM disponible passe sous le seuil critique."""


def available_memory_gb() -> float:
    """RAM disponible en GB. Lit /proc/meminfo (Linux) avec fallback psutil."""
    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.is_file():
        info: dict[str, int] = {}
        with proc_meminfo.open() as f:
            for line in f:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                val = parts[1].strip().split()
                if val and val[0].isdigit():
                    info[key] = int(val[0])  # kB
        if "MemAvailable" in info:
            return info["MemAvailable"] / (1024 * 1024)  # kB → GB
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return float("inf")  # impossible à mesurer = pas de garde


def total_memory_gb() -> float:
    """RAM totale en GB."""
    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.is_file():
        with proc_meminfo.open() as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return float("inf")


def rss_memory_gb() -> float:
    """RSS du process courant en GB (taille effective allouée)."""
    proc_status = Path(f"/proc/{os.getpid()}/status")
    if proc_status.is_file():
        with proc_status.open() as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except ImportError:
        return 0.0


def check_memory(
    *,
    min_available_gb: float = 4.0,
    label: str = "",
    logger: logging.Logger | None = None,
    abort: bool = True,
) -> float:
    """Vérifie que `available_memory_gb() >= min_available_gb`.

    Args
    ----
    min_available_gb : seuil RAM disponible (GB). Sous ce seuil, abort
        (par défaut) ou log WARN.
    label : étiquette pour le log (ex "régime ω=2 Δ=256").
    logger : logger fourni (sinon root).
    abort : True → raise MemoryGuardAbort si sous le seuil.

    Returns
    -------
    Available memory in GB (informatif).

    Raises
    ------
    MemoryGuardAbort si abort=True et available < min_available_gb.
    """
    log = logger or logging.getLogger("shared.mem_guard")
    avail = available_memory_gb()
    rss = rss_memory_gb()
    total = total_memory_gb()
    tag = f"[{label}] " if label else ""
    log.info(
        "%smem: available=%.1f GB | rss=%.1f GB | total=%.1f GB",
        tag, avail, rss, total,
    )
    if avail < min_available_gb:
        msg = (
            f"{tag}RAM disponible {avail:.1f} GB < seuil {min_available_gb:.1f} GB "
            f"(rss={rss:.1f} GB, total={total:.1f} GB)"
        )
        log.error("[mem_guard] %s", msg)
        if abort:
            raise MemoryGuardAbort(msg)
    return avail
