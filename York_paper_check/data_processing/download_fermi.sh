#!/bin/bash
# Download Fermi LAT data files using curl
# Works on macOS without needing wget

set -e

if [[ -n "${REPO_PATH-}" ]]; then
  REPO_ROOT="${REPO_PATH}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

RAW_DIR="${REPO_ROOT}/fermi_data/york"
mkdir -p "${RAW_DIR}"

cd "${RAW_DIR}"

echo "Downloading Fermi LAT data files..."
echo "===================================="
echo ""

# Spacecraft file (required - download first)
echo "[1/8] Downloading spacecraft file (2.4 GB - this will take a while)..."
curl -o L2511041113306F357373F85_SC00.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_SC00.fits

# Photon files
echo "[2/8] Downloading photon file PH00..."
curl -o L2511041113306F357373F85_PH00.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH00.fits

echo "[3/8] Downloading photon file PH02..."
curl -o L2511041113306F357373F85_PH02.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH02.fits

echo "[4/8] Downloading photon file PH01..."
curl -o L2511041113306F357373F85_PH01.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH01.fits

echo "[5/8] Downloading photon file PH05..."
curl -o L2511041113306F357373F85_PH05.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH05.fits

echo "[6/8] Downloading photon file PH03..."
curl -o L2511041113306F357373F85_PH03.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH03.fits

echo "[7/8] Downloading photon file PH04..."
curl -o L2511041113306F357373F85_PH04.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH04.fits

echo "[8/8] Downloading photon file PH06..."
curl -o L2511041113306F357373F85_PH06.fits https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L2511041113306F357373F85_PH06.fits

echo ""
echo "===================================="
echo "Download complete!"
echo "===================================="
echo ""
echo "Files downloaded:"
ls -lh "${RAW_DIR}"/*.fits
echo ""
echo "Next step: Process the data"
echo "  conda activate fermi"
echo "  python \"${REPO_ROOT}/York_paper_check/data_processing/batch_process_fermi.py\""
echo ""
