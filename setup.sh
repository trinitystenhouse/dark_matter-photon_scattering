#!/usr/bin/env bash

set -uo pipefail

if [[ -n "${BASH_SOURCE-}" ]]; then
  _SELF_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION-}" ]]; then
  _SELF_PATH="${(%):-%x}"
else
  _SELF_PATH="$0"
fi

SCRIPT_DIR="$(cd "$(dirname "${_SELF_PATH}")" && pwd)"
export REPO_PATH="${SCRIPT_DIR}"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found on PATH. Please install conda/miniconda first."
  read -p "Press Enter to continue..."
  exit 1
fi

if ! conda env list | grep -q "^scattering "; then
  echo "Creating conda environment 'scattering'..."
  conda create -n scattering python=3.11 -y || {
    echo "ERROR: Failed to create conda environment"
    read -p "Press Enter to continue..."
    exit 1
  }
else
  echo "Conda environment 'scattering' already exists."
fi

echo "Activating conda environment 'scattering'..."
eval "$(conda shell.bash hook)"
conda activate scattering || {
  echo "ERROR: Failed to activate conda environment"
  read -p "Press Enter to continue..."
  exit 1
}

if ! command -v python &>/dev/null; then
  echo "ERROR: python not found on PATH after activating environment"
  read -p "Press Enter to continue..."
  exit 1
fi

if [[ -f "${SCRIPT_DIR}/requirements_analysis.txt" ]]; then
  echo "Installing dependencies from requirements_analysis.txt..."
  python -m pip install --upgrade pip || echo "WARNING: Failed to upgrade pip"
  python -m pip install -r "${SCRIPT_DIR}/requirements_analysis.txt" || echo "WARNING: Some dependencies failed to install"
fi

echo "Installing local helpers package in editable mode..."
python -m pip install -e "${SCRIPT_DIR}" || echo "WARNING: Failed to install local package"

echo "Done. You can now run scripts from anywhere as long as you use the same Python environment."
