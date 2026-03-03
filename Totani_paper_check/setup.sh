#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_PATH="${REPO_ROOT}"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found on PATH. Please install conda/miniconda first."
  read -p "Press Enter to continue..."
  exit 1
fi

if ! conda env list | grep -q "^fermi "; then
  echo "Creating conda environment 'fermi'..."
  conda create -n fermi python=3.11 -y || {
    echo "ERROR: Failed to create conda environment"
    read -p "Press Enter to continue..."
    exit 1
  }
else
  echo "Conda environment 'fermi' already exists."
fi

echo "Activating conda environment 'fermi'..."
eval "$(conda shell.bash hook)"
conda activate fermi || {
  echo "ERROR: Failed to activate conda environment"
  read -p "Press Enter to continue..."
  exit 1
}

if ! command -v python &>/dev/null; then
  echo "ERROR: python not found on PATH after activating environment"
  read -p "Press Enter to continue..."
  exit 1
fi

if [[ -f "${REPO_ROOT}/requirements_fermi.txt" ]]; then
  echo "Installing dependencies from requirements_fermi.txt..."
  python -m pip install --upgrade pip || echo "WARNING: Failed to upgrade pip"
  python -m pip install -r "${REPO_ROOT}/requirements_fermi.txt" || echo "WARNING: Some dependencies failed to install"
fi

if [[ -f "${REPO_ROOT}/requirements_analysis.txt" ]]; then
  echo "Installing dependencies from requirements_analysis.txt..."
  echo "Ensuring numpy<2.0 for astropy compatibility..."
  python -m pip install "numpy<2.0" || echo "WARNING: Failed to pin numpy<2.0"
  python -m pip install -r "${REPO_ROOT}/requirements_analysis.txt" || echo "WARNING: Some dependencies failed to install"
fi

echo "Installing top-level helpers (editable)..."
python -m pip install -e "${REPO_ROOT}" || echo "WARNING: Failed to install top-level package"

echo "Installing Totani helpers (editable)..."
python -m pip install -e "${SCRIPT_DIR}" || echo "WARNING: Failed to install Totani package"

echo "Done. Use this same Python environment when running Totani scripts."

RPROMPT=""