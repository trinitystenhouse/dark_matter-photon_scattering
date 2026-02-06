#!/bin/bash
# Re-download the spacecraft file
# It is very big and takes a long time to download
# Also very easy for it to get corrupted partway through

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

echo "Removing corrupted spacecraft file..."
rm -f L251104111306F357373F85_SC00.fits

echo "Downloading spacecraft file (1.8 GB - this will take 10-30 minutes)..."
echo "Using curl with resume capability..."

# Use curl with resume (-C -) in case download is interrupted
curl -C - -O https://fermi.gsfc.nasa.gov/FTP/fermi/data/lat/queries/L251104111306F357373F85_SC00.fits

echo ""
echo "Verifying download..."
if [ -f "L251104111306F357373F85_SC00.fits" ]; then
    size=$(ls -lh L251104111306F357373F85_SC00.fits | awk '{print $5}')
    echo "✓ File downloaded: $size"
    
    # Check if it's a valid FITS file
    if head -c 6 L251104111306F357373F85_SC00.fits | grep -q "SIMPLE"; then
        echo "✓ Valid FITS file detected"
    else
        echo "⚠ WARNING: File may be corrupted (doesn't start with FITS header)"
        echo "  Try downloading again or check the Fermi server"
    fi
else
    echo "✗ Download failed"
    exit 1
fi

cd ..

echo ""
echo "Ready to process! Run:"
echo "  python \"${REPO_ROOT}/York_paper_check/data_processing/batch_process_fermi.py\""
