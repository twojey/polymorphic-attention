#!/usr/bin/env bash
# install_uv.sh — installation idempotente de uv (gestionnaire Python)
#
# Usage : bash OPS/scripts/install_uv.sh
#
# uv est requis pour `uv sync` (deps via pyproject.toml + uv.lock).
# Réf : https://docs.astral.sh/uv/

set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "==> uv déjà installé : $(uv --version)"
  exit 0
fi

echo "==> uv absent, installation via script Astral..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv install met le binaire dans ~/.local/bin/ — assurer qu'il est dans le PATH
# pour la session courante et pour les futures.
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
else
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERREUR : uv installé mais introuvable dans le PATH." >&2
  echo "         Vérifier ${HOME}/.local/bin/." >&2
  exit 1
fi

echo "==> uv installé : $(uv --version)"
