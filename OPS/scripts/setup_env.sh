#!/usr/bin/env bash
# setup_env.sh — installation reproductible de l'environnement ASP
#
# Idempotent : peut être ré-exécuté sans effet de bord.
# Conçu pour deux topologies :
#   1) VPS sans GPU       → uv sync (torch CPU depuis PyPI)
#   2) RunPod RTX 5090    → ASP_CUDA=1 uv sync --extra cuda (torch CUDA 12.8)
#
# Usage :
#   bash OPS/scripts/setup_env.sh           # auto-détection GPU
#   ASP_CUDA=1 bash OPS/scripts/setup_env.sh    # force CUDA
#   ASP_CUDA=0 bash OPS/scripts/setup_env.sh    # force CPU

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo root : $REPO_ROOT"

# ----------------------------------------------------------------
# 1. Détection GPU
# ----------------------------------------------------------------
if [[ -z "${ASP_CUDA:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ASP_CUDA=1
    echo "==> nvidia-smi détecté → mode CUDA"
  else
    ASP_CUDA=0
    echo "==> Pas de GPU détecté → mode CPU"
  fi
fi

# ----------------------------------------------------------------
# 2. Vérif CUDA / driver si mode CUDA
# ----------------------------------------------------------------
if [[ "$ASP_CUDA" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERREUR : ASP_CUDA=1 mais nvidia-smi indisponible." >&2
    exit 1
  fi
  echo "==> GPU info :"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if (( driver_major < 555 )); then
    echo "AVERTISSEMENT : driver NVIDIA $driver_major < 555 (recommandé pour Blackwell sm_120)." >&2
  fi
fi

# ----------------------------------------------------------------
# 3. Installation de uv
# ----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv absent, installation..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi

uv_version=$(uv --version)
echo "==> $uv_version"

# ----------------------------------------------------------------
# 4. Synchronisation deps
# ----------------------------------------------------------------
if [[ "$ASP_CUDA" == "1" ]]; then
  echo "==> uv sync --extra cuda --extra dev"
  uv sync --extra cuda --extra dev
else
  echo "==> uv sync --extra cpu --extra dev"
  uv sync --extra cpu --extra dev
fi

# ----------------------------------------------------------------
# 5. Vérif post-install : import torch
# ----------------------------------------------------------------
echo "==> Vérification torch :"
uv run python - <<'PY'
import torch
print(f"torch        : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda version : {torch.version.cuda}")
    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"capability   : {torch.cuda.get_device_capability(0)}")
PY

# ----------------------------------------------------------------
# 6. W&B login (interactif si nécessaire)
# ----------------------------------------------------------------
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "==> WANDB_API_KEY présent, login non interactif"
  uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
else
  echo "==> WANDB_API_KEY absent. Login manuel requis : 'uv run wandb login'"
fi

echo "==> Setup terminé."
