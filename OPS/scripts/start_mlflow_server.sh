#!/usr/bin/env bash
# start_mlflow_server.sh — lance le serveur MLflow sur le VPS
#
# Bind par défaut : 127.0.0.1:5000 (jamais exposé public — l'accès distant
# se fait via tunnel SSH côté pod : ssh -L 5000:127.0.0.1:5000 user@vps).
#
# Usage :
#   bash OPS/scripts/start_mlflow_server.sh                        # foreground
#   nohup bash OPS/scripts/start_mlflow_server.sh > mlflow.log &   # background
#   (mieux : lancer dans un tmux ou via systemd unit)
#
# Variables d'environnement optionnelles :
#   MLFLOW_HOST  (défaut : 127.0.0.1)        — ne pas passer à 0.0.0.0 sans auth devant
#   MLFLOW_PORT  (défaut : 5000)
#   MLFLOW_DIR   (défaut : <repo>/OPS/logs/mlflow)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_DIR="${MLFLOW_DIR:-$REPO_ROOT/OPS/logs/mlflow}"

if [[ "$MLFLOW_HOST" != "127.0.0.1" && "$MLFLOW_HOST" != "localhost" ]]; then
  echo "AVERTISSEMENT : bind sur $MLFLOW_HOST — MLflow OSS n'a pas d'auth." >&2
  echo "                Mettre un reverse-proxy (Caddy/nginx) avec basic auth devant." >&2
fi

mkdir -p "$MLFLOW_DIR/artifacts"

echo "==> Repo root      : $REPO_ROOT"
echo "==> MLflow dir     : $MLFLOW_DIR"
echo "==> Bind           : $MLFLOW_HOST:$MLFLOW_PORT"
echo "==> Backend store  : sqlite:///$MLFLOW_DIR/mlflow.db"
echo "==> Artifact root  : $MLFLOW_DIR/artifacts"

# uv doit être disponible — si setup_env.sh n'a pas été exécuté sur le VPS,
# faire `bash OPS/scripts/setup_env.sh` d'abord.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERREUR : uv introuvable. Lancer setup_env.sh d'abord." >&2
  exit 1
fi

exec uv run mlflow server \
  --host "$MLFLOW_HOST" \
  --port "$MLFLOW_PORT" \
  --backend-store-uri "sqlite:///$MLFLOW_DIR/mlflow.db" \
  --default-artifact-root "mlflow-artifacts:/" \
  --artifacts-destination "$MLFLOW_DIR/artifacts" \
  --serve-artifacts
