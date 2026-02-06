#!/bin/bash
# Download Fermi LAT data files (Totani query)
# Uses curl (macOS friendly), supports resume

set -e

DATA_DIR="fermi_data/totani"
BASE_URL="https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries"
TAG="L2601121856298DC5797010"

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

# Spacecraft file
# echo "[1/8] Spacecraft file"
# download "${TAG}_SC00.fits"

# # Photon files
# echo "[2/8] Photon file PH00"
# download "${TAG}_PH00.fits"

# echo "[3/8] Photon file PH01"
# download "${TAG}_PH01.fits"

# echo "[4/8] Photon file PH02"
# download "${TAG}_PH02.fits"

echo "[5/8] Photon file PH03"
download "${TAG}_PH03.fits"

echo "[6/8] Photon file PH04"
download "${TAG}_PH04.fits"

echo "[7/8] Photon file PH05"
download "${TAG}_PH05.fits"

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
