#!/usr/bin/env bash

conda activate scattering

set -euo pipefail

if [[ -n "${BASH_SOURCE-}" ]]; then
  _SELF_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION-}" ]]; then
  _SELF_PATH="${(%):-%x}"
else
  _SELF_PATH="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "${_SELF_PATH}")" && pwd)"
export REPO_PATH="${SCRIPT_DIR}"

if ! command -v python &>/dev/null; then
  echo "python not found on PATH"
  exit 1
fi

echo "Installing local helpers package in editable mode..."
python -m pip install --upgrade pip
python -m pip install -e "${SCRIPT_DIR}"

echo "Done. You can now run scripts from anywhere as long as you use the same Python environment."
