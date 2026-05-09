"""
runner.py — utilitaires de lancement de runs.

Fournit :
- `git_status_clean()` : True si l'arbre git est clean (pas de fichier
  modifié, pas de commit en attente). Pré-requis pour status:registered.
- `git_short_hash()` : 7 chars du HEAD courant. Inclus dans le run name MLflow.
- `hardware_fingerprint()` : dict du GPU/CPU/CUDA/driver pour le manifest.
- `write_manifest()` : produit OPS/logs/manifests/<run_id>.yaml conforme au
  template OPS/configs/manifest_template.yaml.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def git_short_hash(repo: Path | None = None) -> str:
    return _run(["git", "rev-parse", "--short=7", "HEAD"], cwd=repo)


def git_full_hash(repo: Path | None = None) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo)


def git_branch(repo: Path | None = None) -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)


def git_status_clean(repo: Path | None = None) -> bool:
    out = _run(["git", "status", "--porcelain"], cwd=repo)
    return out == ""


def hardware_fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        fp["torch"] = torch.__version__
        fp["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            fp["cuda"] = torch.version.cuda
            fp["device_capability"] = list(torch.cuda.get_device_capability(0))
            fp["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except ImportError:
        fp["torch"] = None
    fp["pod_id"] = os.environ.get("RUNPOD_POD_ID")
    return fp


@dataclass
class Manifest:
    run_id: str
    status: str  # registered | exploratory | invalidated
    phase: str
    sprint: int
    domain: str
    seed: int
    config_path: str
    config_overrides: list[str] = field(default_factory=list)
    oracle_id: str | None = None
    data_hash: str | None = None
    git_commit: str = ""
    git_short_hash: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    hardware: dict[str, Any] = field(default_factory=dict)
    stack: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str | None = None
    duration_s: float | None = None
    mlflow_experiment: str = ""
    mlflow_run_id: str = ""
    mlflow_run_url: str | None = None
    artifacts: list[dict[str, str]] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=lambda: {"outcome": None, "notes": None})


def make_manifest(
    *,
    phase: str,
    sprint: int,
    domain: str,
    seed: int,
    config_path: str,
    config_overrides: list[str] | None = None,
    oracle_id: str | None = None,
    short_description: str = "run",
    repo: Path | None = None,
) -> Manifest:
    sh = git_short_hash(repo)
    full = git_full_hash(repo)
    branch = git_branch(repo)
    dirty = not git_status_clean(repo)
    run_id = f"s{sprint}_{domain}_{short_description}_{sh}"
    return Manifest(
        run_id=run_id,
        status="exploratory" if dirty else "registered",
        phase=phase,
        sprint=sprint,
        domain=domain,
        seed=seed,
        config_path=config_path,
        config_overrides=config_overrides or [],
        oracle_id=oracle_id,
        git_commit=full,
        git_short_hash=sh,
        git_branch=branch,
        git_dirty=dirty,
        hardware=hardware_fingerprint(),
        started_at=datetime.now(timezone.utc).isoformat(),
        mlflow_experiment=f"phase{phase}",
    )


def write_manifest(manifest: Manifest, repo_root: Path) -> Path:
    out_dir = repo_root / "OPS" / "logs" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.run_id}.yaml"
    payload = {k: v for k, v in manifest.__dict__.items()}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return out_path


def finalize_manifest(manifest: Manifest, *, duration_s: float, repo_root: Path) -> Path:
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.duration_s = duration_s
    return write_manifest(manifest, repo_root)
