#!/usr/bin/env bash
# setup_env.sh — wrapper de compatibilité, redirige vers setup_pod.sh
#
# DEPRECATED : préférer désormais les scripts modulaires :
#   - source OPS/scripts/blackwell_env.sh    (ENV vars sm_120, à sourcer)
#   - bash OPS/scripts/install_uv.sh         (uv installer)
#   - bash OPS/scripts/install_python_deps.sh (uv sync CPU/CUDA)
#   - bash OPS/scripts/verify_torch.sh       (smoke check)
#
# Ou l'orchestrateur full pod-from-scratch :
#   - bash OPS/scripts/setup_pod.sh
#
# Documentation complète : OPS/setup/SETUP.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "==> setup_env.sh (legacy) → délégation à setup_pod.sh"
exec bash "$REPO_ROOT/OPS/scripts/setup_pod.sh" "$@"
