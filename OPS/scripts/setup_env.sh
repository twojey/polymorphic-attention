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
# 0. Variables d'environnement Blackwell (sm_120, RTX 5090)
# ----------------------------------------------------------------
# Configuration validée par les scripts Lumis (/root/lumis/OPS).
# Ces variables doivent être set AVANT toute compilation/install qui pourrait
# tenter de produire un kernel CUDA — sinon l'arch n'est pas ciblée.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"                                 # anti-freeze sur compilation parallèle
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"                 # bénin en single-GPU, sain
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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
# 6. Vérification MLFLOW_TRACKING_URI
# ----------------------------------------------------------------
# MLflow self-hosted sur le VPS. Sur le pod, il faut que le tunnel SSH
# vers le VPS soit ouvert (cf. OPS/env/LOGGING.md § Configuration du pod)
# et que MLFLOW_TRACKING_URI pointe vers le tunnel local.
if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "==> MLFLOW_TRACKING_URI absent."
  if [[ "$ASP_CUDA" == "1" ]]; then
    echo "    Sur le pod, ouvrir le tunnel SSH puis :"
    echo "    export MLFLOW_TRACKING_URI=http://localhost:5000"
  else
    echo "    Sur le VPS, MLflow tourne via OPS/scripts/start_mlflow_server.sh"
    echo "    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000"
  fi
else
  echo "==> MLFLOW_TRACKING_URI = $MLFLOW_TRACKING_URI"
  if uv run python -c "import mlflow; mlflow.MlflowClient().search_experiments()" 2>/dev/null; then
    echo "    Connexion MLflow OK."
  else
    echo "    AVERTISSEMENT : MLflow URI set mais serveur injoignable." >&2
  fi
fi

echo "==> Setup terminé."
