#!/usr/bin/env bash
# install_python_deps.sh — synchronise les deps Python via uv
#
# Usage :
#   bash OPS/scripts/install_python_deps.sh         # auto-détection GPU
#   ASP_CUDA=1 bash OPS/scripts/install_python_deps.sh   # force CUDA
#   ASP_CUDA=0 bash OPS/scripts/install_python_deps.sh   # force CPU
#
# Pré-requis : uv installé (cf. install_uv.sh)
#              ENV vars Blackwell sourcées si CUDA (cf. blackwell_env.sh)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERREUR : uv introuvable. Lancer install_uv.sh d'abord." >&2
  exit 1
fi

# Détection GPU si ASP_CUDA pas set
if [[ -z "${ASP_CUDA:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ASP_CUDA=1
    echo "==> nvidia-smi détecté → mode CUDA"
  else
    ASP_CUDA=0
    echo "==> Pas de GPU détecté → mode CPU"
  fi
fi

if [[ "$ASP_CUDA" == "1" ]]; then
  if [[ -z "${ASP_BLACKWELL_ENV_LOADED:-}" ]]; then
    echo "AVERTISSEMENT : ENV vars Blackwell non chargées." >&2
    echo "                Lancer 'source OPS/scripts/blackwell_env.sh' avant." >&2
  fi
  echo "==> uv sync --extra cuda --extra dev"
  uv sync --extra cuda --extra dev
else
  echo "==> uv sync --extra cpu --extra dev"
  uv sync --extra cpu --extra dev
fi

echo "==> Deps Python synchronisées."
