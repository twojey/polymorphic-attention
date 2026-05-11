#!/usr/bin/env bash
# setup_pod_cpu.sh — orchestrateur setup pod RunPod CPU (sans GPU/CUDA).
#
# Variante CPU de setup_pod.sh (qui cible Blackwell sm_120).
# Pas de blackwell_env, pas de verify_torch CUDA — juste uv + deps CPU.
#
# Usage (sur le pod, après git clone du repo) :
#   bash OPS/scripts/setup_pod_cpu.sh
#
# Idempotent : peut être relancé sans dégât si une étape échoue.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/3] Install uv ==="
bash OPS/scripts/install_uv.sh

echo ""
echo "=== [2/3] Install Python deps (mode CPU forcé) ==="
ASP_CUDA=0 bash OPS/scripts/install_python_deps.sh

echo ""
echo "=== [3/3] Smoke check torch CPU ==="
export PATH="${HOME}/.local/bin:${PATH}"
uv run python -c "
import torch, os
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()} (attendu: False)')
print(f'CPU count: {os.cpu_count()}')
print(f'BLAS threads: OPENBLAS={os.environ.get(\"OPENBLAS_NUM_THREADS\", \"unset\")} '
      f'MKL={os.environ.get(\"MKL_NUM_THREADS\", \"unset\")}')
# Smoke eigvalsh single-thread (test du fix deadlock)
m = torch.randn(10, 32, 32)
m = m @ m.transpose(-2, -1) + 1e-6 * torch.eye(32)
ev = torch.linalg.eigvalsh(m)
assert ev.shape == (10, 32), 'eigvalsh shape KO'
print('eigvalsh smoke: OK')
"

echo ""
echo "=== Setup pod CPU terminé ==="
echo ""
echo "Prochaines étapes (dans l'ordre) :"
echo "  1. Côté VPS : POD_IP=... POD_PORT=... bash OPS/scripts/start_mlflow_tunnel.sh"
echo "  2. Côté pod : export MLFLOW_TRACKING_URI=http://localhost:5000"
echo "  3. Copier l'oracle checkpoint depuis le VPS via rsync (cf. POD_CPU_SETUP.md §3)"
echo "  4. Lancer phase 1.5 : ./OPS/env/launch_phase1b.sh --nohup -- bench.n_examples=2000 s_kl.enabled=false"
