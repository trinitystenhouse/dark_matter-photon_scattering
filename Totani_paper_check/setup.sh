#!/usr/bin/env bash

conda activate fermi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_PATH="${REPO_ROOT}"

if ! command -v python &>/dev/null; then
  echo "python not found on PATH"
  exit 1
fi

echo "Installing top-level helpers (editable)..."
python -m pip install -e "${REPO_ROOT}"

echo "Installing Totani helpers (editable)..."
python -m pip install -e "${SCRIPT_DIR}"

echo "Done. Use this same Python environment when running Totani scripts."

RPROMPT=""