#!/usr/bin/env bash
# verify_torch.sh — vérification rapide que torch s'importe et voit le GPU
#
# Usage : bash OPS/scripts/verify_torch.sh
#
# Sortie attendue sur RTX 5090 :
#   torch          : 2.11.0+cu128
#   cuda available : True
#   cuda version   : 12.8
#   device         : NVIDIA GeForce RTX 5090
#   capability     : (12, 0)
#
# Pour un check exhaustif des primitives mathématiques (SVD FP64, FFT,
# vmap, etc.), utiliser plutôt :
#     PYTHONPATH=CODE uv run python OPS/scripts/validate_primitives.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv run python - <<'PY'
import sys
import torch

print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda version   : {torch.version.cuda}")
    print(f"device         : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"capability     : {cap}")
    if cap == (12, 0):
        print("  → Blackwell sm_120 confirmé.")
    else:
        print(f"  ⚠ capability ({cap}) ≠ (12, 0) attendu pour RTX 5090.", file=sys.stderr)
else:
    print("  → mode CPU.")
PY
