#!/usr/bin/env bash
# setup_pod.sh — orchestrateur "from scratch" pour pod RunPod RTX 5090
#
# Lance dans l'ordre les scripts modulaires :
#   1. blackwell_env.sh  (ENV vars sm_120)        — sourcé
#   2. install_uv.sh     (gestionnaire Python)    — exécuté
#   3. install_python_deps.sh (uv sync CUDA)      — exécuté
#   4. verify_torch.sh   (smoke import + GPU)     — exécuté
#
# Idempotent : peut être ré-exécuté sans effet de bord.
#
# Usage : bash OPS/scripts/setup_pod.sh
#
# Pour un setup minimal et progressif (utile en debug), lancer chaque
# script individuellement plutôt que ce wrapper. Cf. OPS/env/SETUP.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo : $REPO_ROOT"
echo

# 1. ENV vars Blackwell — sourcé pour persistance dans la session courante
echo "──── 1/4 : Blackwell ENV vars ────"
# shellcheck disable=SC1091
source "$REPO_ROOT/OPS/scripts/blackwell_env.sh"
echo

# 2. uv
echo "──── 2/4 : uv ────"
bash "$REPO_ROOT/OPS/scripts/install_uv.sh"
# uv install met le binaire dans ~/.local/bin/ — assurer qu'il est dans
# le PATH pour les étapes suivantes
if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
echo

# 3. Deps Python (auto-détection GPU)
echo "──── 3/4 : deps Python ────"
bash "$REPO_ROOT/OPS/scripts/install_python_deps.sh"
echo

# 4. Verif torch + GPU
echo "──── 4/4 : verification torch ────"
bash "$REPO_ROOT/OPS/scripts/verify_torch.sh"
echo

# Rappel MLflow
if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "==> MLFLOW_TRACKING_URI non set."
  echo "    Étape suivante : ouvrir le tunnel SSH inverse depuis le VPS."
  echo "    Cf. OPS/scripts/start_mlflow_tunnel.sh (à lancer depuis le VPS)."
  echo "    Puis sur le pod : export MLFLOW_TRACKING_URI=http://localhost:5000"
fi

echo "==> setup_pod.sh terminé."
