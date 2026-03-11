#!/bin/bash
# Download Fermi LAT data files (Totani query)
# Uses curl (macOS friendly), supports resume

set -euo pipefail

# Resolve repo root (two levels above this script: Totani_paper_check/data_download -> repo root)
if [[ -z "${REPO_DIR:-}" ]]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

DATA_DIR="${REPO_DIR}/fermi_data/totani/0_30"
BASE_URL="https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries"
TAG="L2603090820193F2F87D182"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "===================================="
echo "Downloading Fermi LAT Totani dataset"
echo "Query tag: ${TAG}"
echo "Destination: ${DATA_DIR}"
echo "===================================="
echo ""

# Helper function
download() {
    local file=$1
    local url="${BASE_URL}/${file}"
    echo "Downloading ${file} ..."
    curl -L -C - -o "${file}" "${url}"
    echo ""
}

# wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2603090820193F2F87D182_SC00.fits
# wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2603090820193F2F87D182_PH00.fits
# wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2603090820193F2F87D182_PH01.fits
# wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2603090820193F2F87D182_PH02.fits
# wget https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2603090820193F2F87D182_PH03.fits

# Spacecraft file
# echo "[1/5] Spacecraft file"
# download "${TAG}_SC00.fits"

# # Photon files
# echo "[2/5] Photon file PH00"
# download "${TAG}_PH00.fits"

# echo "[3/5] Photon file PH01"
# download "${TAG}_PH01.fits"

# echo "[4/5] Photon file PH02"
# download "${TAG}_PH02.fits"

echo "[5/5] Photon file PH03"
download "${TAG}_PH03.fits"

echo ""
echo "===================================="
echo "Download complete"
echo "===================================="
echo ""

ls -lh *.fits

echo ""
echo "Next steps:"
echo "  conda activate fermitools"
echo "  bash process_totani.sh"
echo ""
