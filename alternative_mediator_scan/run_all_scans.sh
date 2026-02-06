#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA="${REPO_ROOT}/fermi_data/york/processed/spectrum_data.txt"
RHO="1.2e-6"
LVAL="12"
LUNIT="Gpc"

# DM masses to scan (GeV)
MCHI_LIST=(
  1e-3
  1e-2
  1e-1
  1
  1e1
  1e2
  1e3
  1e4
  1e5
)

SPACING="50"
E0FORCED="175"
NCOS="1200"

# Dip criterion: require a fractional dip of at least this depth somewhere in the band.
# Example: 0.01 = 1% dip.
DIP_ANYWHERE="0.01"

# 99% CL compatibility with data: require p-value >= PMIN.
PMIN="0.01"
PMIN_MODEL="dm"

# Conservative perturbativity bound
GCHI_MAX="3.5449077"   # sqrt(4*pi)

RESULTS_BASE="${SCRIPT_DIR}/scans/results"
PLOTS_BASE="${SCRIPT_DIR}/scans/plots"

mkdir -p "${RESULTS_BASE}"
mkdir -p "${PLOTS_BASE}"

collect_plots() {
  local results_dir="$1"
  local plots_dir="$2"
  local mchi_tag="$3"

  mkdir -p "${plots_dir}"
  if [[ -d "${results_dir}/plots" ]]; then
    shopt -s nullglob
    for f in "${results_dir}/plots"/*.png; do
      local base
      base="$(basename "${f}")"
      cp -f "${f}" "${plots_dir}/${mchi_tag}_${base}"
    done
    shopt -u nullglob
  fi
}

for MCHI in "${MCHI_LIST[@]}"; do
  echo "============================================================"
  echo "Scanning mchi=${MCHI} GeV"
  echo "============================================================"

  MCHI_TAG="mchi${MCHI}"

# Scalar: mmed, gchi, cgamma
OUT_SCALAR="${RESULTS_BASE}/scalar/${MCHI_TAG}"
PLOTS_SCALAR="${PLOTS_BASE}/scalar"
python "${SCRIPT_DIR}/alternative_mediators_scan.py" \
  --filename "$DATA" \
  --mchi "$MCHI" \
  --mediator scalar \
  --rho "$RHO" --L "$LVAL" --L_unit "$LUNIT" \
  --spacing "$SPACING" --E0_forced "$E0FORCED" --ncos "$NCOS" \
  --scan mmed=log:0.01:1e4:18 \
  --scan gchi=log:1e-3:"$GCHI_MAX":16 \
  --scan cgamma=log:1e-10:1:18 \
  --gchi_max "$GCHI_MAX" \
  --dip_anywhere "$DIP_ANYWHERE" \
  --pmin "$PMIN" \
  --pmin_model "$PMIN_MODEL" \
  --plot_best_deviant \
  --outdir "${OUT_SCALAR}" \
  || echo "[warn] scalar: no dip point found (or scan failed)"
collect_plots "${OUT_SCALAR}" "${PLOTS_SCALAR}" "${MCHI_TAG}"

# Pseudoscalar
OUT_PSEUDO="${RESULTS_BASE}/pseudoscalar/${MCHI_TAG}"
PLOTS_PSEUDO="${PLOTS_BASE}/pseudoscalar"
python "${SCRIPT_DIR}/alternative_mediators_scan.py" \
  --filename "$DATA" \
  --mchi "$MCHI" \
  --mediator pseudoscalar \
  --rho "$RHO" --L "$LVAL" --L_unit "$LUNIT" \
  --spacing "$SPACING" --E0_forced "$E0FORCED" --ncos "$NCOS" \
  --scan mmed=log:0.01:1e4:18 \
  --scan gchi=log:1e-3:"$GCHI_MAX":16 \
  --scan ctilde=log:1e-10:1:18 \
  --gchi_max "$GCHI_MAX" \
  --dip_anywhere "$DIP_ANYWHERE" \
  --pmin "$PMIN" \
  --pmin_model "$PMIN_MODEL" \
  --plot_best_deviant \
  --outdir "${OUT_PSEUDO}" \
  || echo "[warn] pseudoscalar: no dip point found (or scan failed)"
collect_plots "${OUT_PSEUDO}" "${PLOTS_PSEUDO}" "${MCHI_TAG}"

# Thomson / millicharge
OUT_THOMSON="${RESULTS_BASE}/thomson/${MCHI_TAG}"
PLOTS_THOMSON="${PLOTS_BASE}/thomson"
python "${SCRIPT_DIR}/alternative_mediators_scan.py" \
  --filename "$DATA" \
  --mchi "$MCHI" \
  --mediator thomson \
  --rho "$RHO" --L "$LVAL" --L_unit "$LUNIT" \
  --spacing "$SPACING" --E0_forced "$E0FORCED" --ncos "$NCOS" \
  --scan qeff=log:1e-20:1e-2:28 \
  --dip_anywhere "$DIP_ANYWHERE" \
  --pmin "$PMIN" \
  --pmin_model "$PMIN_MODEL" \
  --plot_best_deviant \
  --outdir "${OUT_THOMSON}" \
  || echo "[warn] thomson: no dip point found (or scan failed)"
collect_plots "${OUT_THOMSON}" "${PLOTS_THOMSON}" "${MCHI_TAG}"

# Rayleigh even (EFT-ish: Lambda >= 1500 GeV)
OUT_RAY_EVEN="${RESULTS_BASE}/rayleigh_even/${MCHI_TAG}"
PLOTS_RAY_EVEN="${PLOTS_BASE}/rayleigh_even"
python "${SCRIPT_DIR}/alternative_mediators_scan.py" \
  --filename "$DATA" \
  --mchi "$MCHI" \
  --mediator rayleigh_even \
  --rho "$RHO" --L "$LVAL" --L_unit "$LUNIT" \
  --spacing "$SPACING" --E0_forced "$E0FORCED" --ncos "$NCOS" \
  --scan Lambda=log:1:1e6:28 \
  --dip_anywhere "$DIP_ANYWHERE" \
  --pmin "$PMIN" \
  --pmin_model "$PMIN_MODEL" \
  --plot_best_deviant \
  --outdir "${OUT_RAY_EVEN}" \
  || echo "[warn] rayleigh_even: no dip point found (or scan failed)"
collect_plots "${OUT_RAY_EVEN}" "${PLOTS_RAY_EVEN}" "${MCHI_TAG}"

# Rayleigh odd
OUT_RAY_ODD="${RESULTS_BASE}/rayleigh_odd/${MCHI_TAG}"
PLOTS_RAY_ODD="${PLOTS_BASE}/rayleigh_odd"
python "${SCRIPT_DIR}/alternative_mediators_scan.py" \
  --filename "$DATA" \
  --mchi "$MCHI" \
  --mediator rayleigh_odd \
  --rho "$RHO" --L "$LVAL" --L_unit "$LUNIT" \
  --spacing "$SPACING" --E0_forced "$E0FORCED" --ncos "$NCOS" \
  --scan Lambda=log:1:1e6:28 \
  --dip_anywhere "$DIP_ANYWHERE" \
  --pmin "$PMIN" \
  --pmin_model "$PMIN_MODEL" \
  --plot_best_deviant \
  --outdir "${OUT_RAY_ODD}" \
  || echo "[warn] rayleigh_odd: no dip point found (or scan failed)"
collect_plots "${OUT_RAY_ODD}" "${PLOTS_RAY_ODD}" "${MCHI_TAG}"

# Graviton: scan regulator as proxy for IR/angle resolution
OUT_GRAV="${RESULTS_BASE}/graviton/${MCHI_TAG}"
PLOTS_GRAV="${PLOTS_BASE}/graviton"
python "${SCRIPT_DIR}/alternative_mediators_scan.py" \
  --filename "$DATA" \
  --mchi "$MCHI" \
  --mediator graviton \
  --rho "$RHO" --L "$LVAL" --L_unit "$LUNIT" \
  --spacing "$SPACING" --E0_forced "$E0FORCED" --ncos "$NCOS" \
  --scan regulator=log:1e-12:1e-2:22 \
  --dip_anywhere "$DIP_ANYWHERE" \
  --pmin "$PMIN" \
  --pmin_model "$PMIN_MODEL" \
  --plot_best_deviant \
  --outdir "${OUT_GRAV}" \
  || echo "[warn] graviton: no dip point found (or scan failed)"
collect_plots "${OUT_GRAV}" "${PLOTS_GRAV}" "${MCHI_TAG}"

done

echo "Done. Check scans/*/scan_*.csv and scans/*/bestdeviantplot_*.png"
