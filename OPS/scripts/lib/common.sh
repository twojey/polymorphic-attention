#!/usr/bin/env bash
# common.sh — helpers de robustesse pour tous les scripts shell ASP.
#
# **À sourcer en début de chaque script** :
#
#     # shellcheck source=OPS/scripts/lib/common.sh
#     source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#     asp::init_strict_mode
#     asp::init_logging "phase1b_skl"  # -> OPS/logs/phase1b_skl_<TS>.log
#     asp::log_env
#
# Fournit :
#   - asp::init_strict_mode      → set -euo pipefail + trap ERR avec contexte
#   - asp::init_logging <prefix> → fichier log horodaté dans OPS/logs/, tee stdout+stderr
#   - asp::log_env               → commit, Python, GPU, BLAS, PYTHONPATH
#   - asp::log <msg>             → ligne préfixée timestamp UTC
#   - asp::section <titre>       → marker de section visible
#   - asp::die <code> <msg>      → log erreur + exit
#   - asp::require_cmd <bin>     → check binaire disponible ou die
#   - asp::require_file <path>   → check fichier existe ou die
#
# Garanties :
#   - stderr → stdout (exec 2>&1) après asp::init_logging → tout dans le log
#   - trap ERR capture line number + commande + exit code
#   - traps EXIT logge la durée totale + status final
#
# Origine : feedback utilisateur 2026-05-11 suite Run 3 silent crash.
# Cf. DOC/carnet_de_bord.md "Refactor scripts robustes" + memory feedback.

# Guard contre double-source
if [[ -n "${ASP_COMMON_SH_LOADED:-}" ]]; then
    return 0
fi
readonly ASP_COMMON_SH_LOADED=1

# Globals (lecture après asp::init_logging)
ASP_LOG_FILE=""
ASP_LOG_PREFIX=""
ASP_START_EPOCH=""
ASP_REPO_ROOT=""

# ---------------------------------------------------------------------------
# Internal: timestamp UTC ISO-8601
# ---------------------------------------------------------------------------
asp::_ts() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

# ---------------------------------------------------------------------------
# asp::log <msg> — log horodaté
# ---------------------------------------------------------------------------
asp::log() {
    printf '[%s] %s\n' "$(asp::_ts)" "$*"
}

# ---------------------------------------------------------------------------
# asp::section <titre> — marker de section
# ---------------------------------------------------------------------------
asp::section() {
    local title="$*"
    echo ""
    echo "==================================================================="
    printf '[%s] === %s ===\n' "$(asp::_ts)" "$title"
    echo "==================================================================="
}

# ---------------------------------------------------------------------------
# asp::die <exit_code> <msg> — log erreur + exit
# ---------------------------------------------------------------------------
asp::die() {
    local code="${1:-1}"
    shift
    printf '[%s] [FATAL] %s (exit=%d)\n' "$(asp::_ts)" "$*" "$code" >&2
    exit "$code"
}

# ---------------------------------------------------------------------------
# asp::require_cmd <bin> — check binaire ou die
# ---------------------------------------------------------------------------
asp::require_cmd() {
    local bin="$1"
    if ! command -v "$bin" >/dev/null 2>&1; then
        asp::die 127 "binaire '$bin' introuvable dans PATH"
    fi
}

# ---------------------------------------------------------------------------
# asp::require_file <path> — check fichier ou die
# ---------------------------------------------------------------------------
asp::require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        asp::die 2 "fichier requis introuvable : $path"
    fi
}

# ---------------------------------------------------------------------------
# asp::_resolve_repo_root — détecte le repo root depuis BASH_SOURCE[1]
# ---------------------------------------------------------------------------
asp::_resolve_repo_root() {
    # On suppose lib/common.sh est dans OPS/scripts/lib/, donc repo root = ../../..
    local lib_dir
    lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # lib_dir = .../OPS/scripts/lib → repo = .../
    echo "$(cd "$lib_dir/../../.." && pwd)"
}

# ---------------------------------------------------------------------------
# asp::init_strict_mode — set -euo pipefail + trap ERR + trap EXIT
# ---------------------------------------------------------------------------
asp::init_strict_mode() {
    set -euo pipefail

    # trap ERR : capture ligne d'erreur + commande + exit code
    # NB : exécuté AVANT que set -e ne fasse exit, donc on peut logger.
    trap 'asp::_on_err $? $LINENO "$BASH_COMMAND"' ERR

    # trap EXIT : log durée + status final (qu'on quitte sur succès ou erreur)
    trap 'asp::_on_exit $?' EXIT
}

asp::_on_err() {
    local code="$1" lineno="$2" cmd="$3"
    printf '[%s] [ERR] ligne=%d cmd=%q exit=%d\n' \
        "$(asp::_ts)" "$lineno" "$cmd" "$code" >&2
    # Stack trace bash (depth 1+ = caller chain)
    local i=1
    while caller "$i" >/dev/null 2>&1; do
        local frame
        frame="$(caller "$i")"
        printf '[%s] [ERR]   stack: %s\n' "$(asp::_ts)" "$frame" >&2
        i=$((i + 1))
    done
}

asp::_on_exit() {
    local code="$1"
    if [[ -n "$ASP_START_EPOCH" ]]; then
        local now duration
        now="$(date -u +%s)"
        duration=$((now - ASP_START_EPOCH))
        local hms
        hms="$(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60)))"
        if [[ "$code" -eq 0 ]]; then
            printf '[%s] [DONE] OK duration=%s (%ds) log=%s\n' \
                "$(asp::_ts)" "$hms" "$duration" "$ASP_LOG_FILE"
        else
            printf '[%s] [DONE] FAIL exit=%d duration=%s (%ds) log=%s\n' \
                "$(asp::_ts)" "$code" "$hms" "$duration" "$ASP_LOG_FILE" >&2
        fi
    fi
}

# ---------------------------------------------------------------------------
# asp::init_logging <prefix> [log_dir]
#   Crée OPS/logs/<prefix>_<TS>.log, redirige stdout+stderr en tee vers ce fichier.
#   Le terminal continue à voir la sortie ET le fichier la garde.
# ---------------------------------------------------------------------------
asp::init_logging() {
    local prefix="${1:?asp::init_logging: prefix requis}"
    local log_dir="${2:-}"

    ASP_REPO_ROOT="$(asp::_resolve_repo_root)"
    if [[ -z "$log_dir" ]]; then
        log_dir="$ASP_REPO_ROOT/OPS/logs"
    fi
    mkdir -p "$log_dir"

    local ts
    ts="$(date -u +'%Y%m%dT%H%M%SZ')"
    ASP_LOG_FILE="$log_dir/${prefix}_${ts}.log"
    ASP_LOG_PREFIX="$prefix"
    ASP_START_EPOCH="$(date -u +%s)"

    # Redirige stdout+stderr en tee → écrit dans le fichier ET sur le terminal.
    # /dev/null en fin pour ne pas que tee retourne EPIPE.
    exec > >(tee -a "$ASP_LOG_FILE") 2>&1

    asp::log "[init] log=$ASP_LOG_FILE prefix=$ASP_LOG_PREFIX repo=$ASP_REPO_ROOT pid=$$"
}

# ---------------------------------------------------------------------------
# asp::log_env — capture l'environnement critique
# ---------------------------------------------------------------------------
asp::log_env() {
    asp::section "Environnement"

    # Repo info
    asp::log "repo_root=$ASP_REPO_ROOT"
    if [[ -d "$ASP_REPO_ROOT/.git" ]]; then
        local commit branch
        commit="$(cd "$ASP_REPO_ROOT" && git rev-parse --short=7 HEAD 2>/dev/null || echo 'n/a')"
        branch="$(cd "$ASP_REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
        asp::log "git: commit=$commit branch=$branch"
        # Files modified (untracked + uncommitted)
        local dirty
        dirty="$(cd "$ASP_REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l)"
        if [[ "$dirty" -gt 0 ]]; then
            asp::log "git: working tree DIRTY ($dirty files modified or untracked)"
        fi
    fi

    # OS / kernel
    asp::log "host=$(hostname) os=$(uname -srm)"

    # Python / uv
    if command -v python3 >/dev/null 2>&1; then
        asp::log "python3=$(python3 --version 2>&1 | head -1)"
    fi
    if command -v uv >/dev/null 2>&1; then
        asp::log "uv=$(uv --version 2>&1 | head -1)"
    fi

    # GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_info
        gpu_info="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
        asp::log "gpu=$gpu_info"
    else
        asp::log "gpu=none (nvidia-smi absent)"
    fi

    # BLAS env (critique : deadlock eigvalsh si multi-thread)
    asp::log "BLAS env: OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset} MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-unset}"

    # PYTHONPATH (critique : si pas set, imports échouent)
    asp::log "PYTHONPATH=${PYTHONPATH:-unset}"

    # MLflow URI (utile pour debug si run crash dans start_run)
    asp::log "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-unset}"

    # Disk space (OPS/logs/ doit avoir de la place)
    if command -v df >/dev/null 2>&1; then
        local disk
        disk="$(df -h "$ASP_REPO_ROOT/OPS/logs" 2>/dev/null | tail -1)"
        asp::log "disk: $disk"
    fi
}

# ---------------------------------------------------------------------------
# asp::run_with_status <cmd...> — exécute une cmd, logge debut/fin/duration
# ---------------------------------------------------------------------------
asp::run_with_status() {
    local label="$1"
    shift
    asp::log "[run] $label : $*"
    local t0 t1 duration code
    t0="$(date -u +%s)"
    if "$@"; then
        code=0
    else
        code=$?
    fi
    t1="$(date -u +%s)"
    duration=$((t1 - t0))
    if [[ "$code" -eq 0 ]]; then
        asp::log "[run] $label OK (${duration}s)"
    else
        asp::log "[run] $label FAIL exit=$code (${duration}s)"
    fi
    return "$code"
}
