#!/bin/bash
# launch_phase1b.sh — Lancer phase 1.5 avec safeguards contre deadlock BLAS.
#
# Usage:
#   ./OPS/env/launch_phase1b.sh [OPTIONS] -- [HYDRA_OVERRIDES]
#   ./OPS/env/launch_phase1b.sh --nohup -- bench.n_examples=2000 s_kl.enabled=false
#
# Options:
#   --nohup              Lancer en background (nohup + disown)
#   --watch              Afficher les logs en temps réel (tail -f)
#   --help               Afficher cette aide

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NOHUP_MODE=0
WATCH_MODE=0
LOG_FILE="/tmp/phase1b_$(date +%s).log"

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup)
            NOHUP_MODE=1
            shift
            ;;
        --watch)
            WATCH_MODE=1
            shift
            ;;
        --help)
            cat >&2 <<'EOF'
launch_phase1b.sh — Lancer phase 1.5 avec safeguards BLAS.

Options:
  --nohup    Background (nohup + disown)
  --watch    Afficher les logs (tail -f)

Exemples:
  ./OPS/env/launch_phase1b.sh -- bench.n_examples=2000
  ./OPS/env/launch_phase1b.sh --nohup -- n_examples=500 s_kl.enabled=true
EOF
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Erreur: option inconnue $1" >&2
            exit 1
            ;;
    esac
done

# Hydra overrides (rest of args)
HYDRA_OVERRIDES=("$@")

# Safeguards env vars (évite deadlock BLAS multi-thread)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# MLflow tracking (sans ça, le run crash dans start_run — cf. carnet 2026-05-11 06:31)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
# PYTHONPATH : le code n'est pas installé en editable, il faut pointer CODE/
export PYTHONPATH="$REPO_ROOT/CODE:${PYTHONPATH:-}"

cd "$REPO_ROOT"

if [[ $NOHUP_MODE -eq 1 ]]; then
    echo "Lancement en background: $LOG_FILE"
    nohup bash -c "
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export OMP_NUM_THREADS=1
        export PYTHONUNBUFFERED=1
        export MLFLOW_TRACKING_URI='${MLFLOW_TRACKING_URI}'
        export PYTHONPATH='${PYTHONPATH}'
        uv run python -u -m phase1b_calibration_signal.run \
          --config-path=../../OPS/configs/phase1b --config-name=signals \
          oracle_checkpoint=/root/polymorphic-attention/OPS/logs/mlflow/artifacts/2/7e1e859818b64b85b6c2c88433767c48/artifacts/oracle/s1_smnist_oracle_e2f0b5e_oracle.ckpt \
          ${HYDRA_OVERRIDES[@]}
    " > "$LOG_FILE" 2>&1 &
    PID=$!
    disown
    echo "PID: $PID"
    if [[ $WATCH_MODE -eq 1 ]]; then
        echo "Watching logs..."
        sleep 2
        tail -f "$LOG_FILE"
    fi
else
    # Foreground
    uv run python -u -m phase1b_calibration_signal.run \
      --config-path=../../OPS/configs/phase1b --config-name=signals \
      oracle_checkpoint=/root/polymorphic-attention/OPS/logs/mlflow/artifacts/2/7e1e859818b64b85b6c2c88433767c48/artifacts/oracle/s1_smnist_oracle_e2f0b5e_oracle.ckpt \
      "${HYDRA_OVERRIDES[@]}"
fi
