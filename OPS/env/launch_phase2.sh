#!/usr/bin/env bash
# launch_phase2.sh — Lancer phase 2 (Audit Spectral) avec safeguards BLAS,
# logs persistants, tunnel MLflow, dump dir auto.
#
# Reprise du pattern launch_phase1b.sh (carnet 2026-05-11 refactor scripts robustes).
#
# Usage:
#   bash OPS/env/launch_phase2.sh [OPTIONS] -- [HYDRA_OVERRIDES]
#
# Options:
#   --nohup              Lancer en background (nohup + disown)
#   --watch              Afficher les logs en temps réel (tail -f)
#   --dump-dir PATH      Dossier des dumps multi-bucket produits par run_extract.py
#                        (défaut : /workspace/phase1_extract)
#   --dump-path PATH     Single dump .pt (mode legacy, mutuellement exclusif avec --dump-dir)
#   --smoke              Active mode smoke (sous-échantillonnage agressif :
#                        decoupling.max_examples=30, batteries.max_examples_per_layer=4)
#   --gpu-fast           SVD sur GPU FP32 (RTX 5090 nerfé en FP64 ; r_eff(θ) OK en fp32).
#                        Override svd.device=cuda svd.precision=fp32. Spec strict reste fp64 CPU.
#   --log-prefix PFX     Préfixe du log (défaut : "phase2")
#   --help               Afficher cette aide
#
# Exemples:
#   bash OPS/env/launch_phase2.sh --smoke -- attention_dump_dir=/workspace/phase1_extract
#   bash OPS/env/launch_phase2.sh --nohup --gpu-fast
#   bash OPS/env/launch_phase2.sh --nohup --watch
#
# Codes de sortie :
#   0   succès complet
#   1   erreur de parsing options
#   2   pré-condition manquante (uv, dump dir/path, repo)
#   N   exit code Python (propagation)
#
# Logs : OPS/logs/<prefix>_<UTC_TS>.log

# shellcheck source=../scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/lib/common.sh"
asp::init_strict_mode

# --- Parse options -----------------------------------------------------------

NOHUP_MODE=0
WATCH_MODE=0
SMOKE_MODE=0
GPU_FAST=0
LOG_PREFIX="phase2"
DUMP_DIR_DEFAULT="/workspace/phase1_extract"
DUMP_DIR=""
DUMP_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup)        NOHUP_MODE=1; shift ;;
        --watch)        WATCH_MODE=1; shift ;;
        --smoke)        SMOKE_MODE=1; shift ;;
        --gpu-fast)     GPU_FAST=1; shift ;;
        --dump-dir)     DUMP_DIR="${2:?--dump-dir requiert un argument}"; shift 2 ;;
        --dump-path)    DUMP_PATH="${2:?--dump-path requiert un argument}"; shift 2 ;;
        --log-prefix)   LOG_PREFIX="${2:?--log-prefix requiert un argument}"; shift 2 ;;
        --help|-h)      sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        --)             shift; break ;;
        -*)             asp::die 1 "option inconnue : $1 (--help pour usage)" ;;
        *)              asp::die 1 "argument inattendu : $1 (utiliser -- avant les overrides Hydra)" ;;
    esac
done

HYDRA_OVERRIDES=("$@")

if [[ -n "$DUMP_DIR" && -n "$DUMP_PATH" ]]; then
    asp::die 1 "--dump-dir et --dump-path sont mutuellement exclusifs"
fi
if [[ -z "$DUMP_DIR" && -z "$DUMP_PATH" ]]; then
    DUMP_DIR="$DUMP_DIR_DEFAULT"
fi

# --- Init logging -----------------------------------------------------------

ASP_REPO_ROOT="$(asp::_resolve_repo_root)"

if [[ "$NOHUP_MODE" -eq 0 ]]; then
    asp::init_logging "$LOG_PREFIX"
    asp::log_env
fi

# --- Pré-conditions ---------------------------------------------------------

asp::section "Pré-conditions"
asp::require_cmd uv
[[ -d "$ASP_REPO_ROOT/CODE" ]] || asp::die 2 "CODE/ introuvable dans $ASP_REPO_ROOT"

if [[ -n "$DUMP_DIR" ]]; then
    [[ -d "$DUMP_DIR" ]] || asp::die 2 "dump dir introuvable : $DUMP_DIR (lancer extract d'abord)"
    # Vérifier au moins un dump présent
    DUMP_COUNT="$(find "$DUMP_DIR" -maxdepth 1 -name 'audit_dump_seq*.pt' | wc -l)"
    [[ "$DUMP_COUNT" -gt 0 ]] || asp::die 2 "aucun audit_dump_seq*.pt dans $DUMP_DIR"
    asp::log "dump_dir    : $DUMP_DIR ($DUMP_COUNT buckets)"
else
    asp::require_file "$DUMP_PATH"
    asp::log "dump_path   : $DUMP_PATH"
fi

DIRTY_COUNT="$(cd "$ASP_REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l)"
if [[ "$DIRTY_COUNT" -gt 0 ]]; then
    asp::log "[WARN] git working tree DIRTY ($DIRTY_COUNT fichiers) → manifest status=exploratory."
fi

# --- Env vars ---------------------------------------------------------------

asp::section "Env vars phase 2"

# Anti-deadlock BLAS — OBLIGATOIRE pour compute_s_spectral (diagnostic découplage)
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export PYTHONPATH="$ASP_REPO_ROOT/CODE${PYTHONPATH:+:$PYTHONPATH}"

export ASP_LOG_FILE
export ASP_LOG_PREFIX

asp::log "BLAS=1 PYTHONUNBUFFERED=1 MLFLOW=$MLFLOW_TRACKING_URI"

# --- Construction overrides Hydra ------------------------------------------

asp::section "Overrides Hydra"

OVERRIDES_AUTO=()
if [[ -n "$DUMP_DIR" ]]; then
    OVERRIDES_AUTO+=("attention_dump_dir=$DUMP_DIR")
fi
if [[ -n "$DUMP_PATH" ]]; then
    OVERRIDES_AUTO+=("attention_dump_path=$DUMP_PATH")
fi
if [[ "$SMOKE_MODE" -eq 1 ]]; then
    OVERRIDES_AUTO+=("decoupling.max_examples=30")
    OVERRIDES_AUTO+=("decoupling.n_boot=200")
    OVERRIDES_AUTO+=("batteries.max_examples_per_layer=4")
    asp::log "smoke mode activé (sous-échantillonnage agressif)"
fi
if [[ "$GPU_FAST" -eq 1 ]]; then
    OVERRIDES_AUTO+=("svd.device=cuda")
    OVERRIDES_AUTO+=("svd.precision=fp32")
    asp::log "[ATTN] gpu-fast → svd.device=cuda svd.precision=fp32"
    asp::log "       Spec stricte DOC/01 §8.4 = CPU FP64. À documenter dans rapport."
fi

asp::log "auto    : ${OVERRIDES_AUTO[*]:-<aucun>}"
asp::log "user    : ${HYDRA_OVERRIDES[*]:-<aucun>}"

# --- Exécution --------------------------------------------------------------

asp::section "Lancement phase2_audit_spectral.run"

cd "$ASP_REPO_ROOT"

PYTHON_CMD=(
    uv run python -u -m phase2_audit_spectral.run
    --config-path=../../OPS/configs/phase2 --config-name=audit
    "${OVERRIDES_AUTO[@]}"
    "${HYDRA_OVERRIDES[@]}"
)

if [[ "$NOHUP_MODE" -eq 1 ]]; then
    ASP_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
    CHILD_LOG="$ASP_REPO_ROOT/OPS/logs/${LOG_PREFIX}_${ASP_TS}.log"
    mkdir -p "$(dirname "$CHILD_LOG")"

    {
        printf '[%s] [init] log=%s pid=$$ (background)\n' "$(asp::_ts)" "$CHILD_LOG"
        printf '[%s] commit=%s\n' "$(asp::_ts)" "$(git -C "$ASP_REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo n/a)"
        printf '[%s] cmd=%s\n' "$(asp::_ts)" "${PYTHON_CMD[*]}"
        printf '[%s] BLAS=1 PYTHONPATH=%s MLFLOW=%s\n' "$(asp::_ts)" "$PYTHONPATH" "$MLFLOW_TRACKING_URI"
    } > "$CHILD_LOG"

    nohup "${PYTHON_CMD[@]}" >> "$CHILD_LOG" 2>&1 < /dev/null &
    PID=$!
    disown

    echo "Lancé en background :"
    echo "  PID      : $PID"
    echo "  Log      : $CHILD_LOG"
    echo "  Suivre   : tail -f $CHILD_LOG"
    echo "  Killer   : kill $PID"

    if [[ "$WATCH_MODE" -eq 1 ]]; then
        echo ""
        echo "[watch] tail -f démarré (Ctrl-C pour arrêter le tail, pas le process)"
        sleep 2
        tail -f "$CHILD_LOG"
    fi
else
    asp::log "cmd: ${PYTHON_CMD[*]}"
    "${PYTHON_CMD[@]}"
fi
