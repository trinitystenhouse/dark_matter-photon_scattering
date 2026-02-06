#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/Fermi-LAT_analysis_coupling.py"
DATA_FILE="${REPO_ROOT}/fermi_data/york/processed/spectrum_data.txt"

python "${PY_SCRIPT}" --is-cosmic --y_eff 1e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 2e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 3e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 4e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 5e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 6e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 7e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 8e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 9e3 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.1e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.2e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.3e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.4e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.5e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.6e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.7e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.8e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 1.9e4 --filename "${DATA_FILE}"
python "${PY_SCRIPT}" --is-cosmic --y_eff 2e4 --filename "${DATA_FILE}"
