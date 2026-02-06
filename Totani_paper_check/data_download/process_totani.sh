#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Paths
# -------------------------
DATA_DIR="../fermi_data/totani"
OUT_DIR="${DATA_DIR}/../processed"
mkdir -p "${OUT_DIR}"

# Find spacecraft + photon files
SCFILE="$(ls -1 ${DATA_DIR}/*_SC*.fits | head -n 1)"
PHLIST="${OUT_DIR}/photon_list.txt"
ls -1 ${DATA_DIR}/*_PH*.fits > "${PHLIST}"

echo "SCFILE: ${SCFILE}"
echo "Photon list: ${PHLIST}"
echo "N photon files: $(wc -l < ${PHLIST})"
echo

# -------------------------
# Totani-style selections
# -------------------------
# ROI geometry for Fig.1 maps: |l|,|b|<=60 deg, CAR, 0.125 deg/pix => 960x960
NX=960
NY=960
BINSZ=0.125
XREF=0
YREF=0
COORDSYS="GAL"
PROJ="CAR"
AXISROT=0

# Energy range: for Fig.1 only, 1–10 GeV keeps it smaller.
# For full Totani dataset, set EMAX=1000000 (1 TeV) and ENUMBINS=13 (log bins).
EMIN=1000
EMAX=1000000
ENUMBINS=13

# Event selection
EVCLASS=1024   # ULTRACLEANVETO (Pass 8)
EVTYPE=3       # FRONT+BACK
ZMAX=100

# GTI filter
GTI_FILTER="(DATA_QUAL>0)&&(LAT_CONFIG==1)"

# IRFs must match evclass choice
# If EVCLASS=1024 => use ULTRACLEANVETO IRFs
IRFS="P8R3_ULTRACLEANVETO_V3"

# -------------------------
# 1) Apply GTIs + selection per file (outputs: *_gti_sel.fits)
# -------------------------
SEL_LIST="${OUT_DIR}/sel_gti_list.txt"
: > "${SEL_LIST}"

echo "==> Filtering weekly/query PH files with gtmktime+gtselect"
i=0
while read -r PH; do
  i=$((i+1))
  base="$(basename "${PH}" .fits)"
  gti="${OUT_DIR}/${base}_gti.fits"
  sel="${OUT_DIR}/${base}_gti_sel.fits"

  echo
  echo "[${i}] gtmktime -> ${gti}"
  gtmktime \
    "scfile=${SCFILE}" \
    "evfile=${PH}" \
    "outfile=${gti}" \
    "roicut=no" \
    "filter=${GTI_FILTER}"

  echo "[${i}] gtselect -> ${sel}"
  # Use a big cone so gtselect doesn't clip your square; we bin into square later.
  gtselect \
    "infile=${gti}" \
    "outfile=${sel}" \
    "ra=266.4051" "dec=-28.936175" "rad=60" \
    "tmin=INDEF" "tmax=INDEF" \
    "emin=${EMIN}" "emax=${EMAX}" \
    "zmax=${ZMAX}" \
    "evclass=${EVCLASS}" \
    "evtype=${EVTYPE}"

  rm -f "${gti}"
  echo "${sel}" >> "${SEL_LIST}"
done < "${PHLIST}"

echo
echo "✓ Wrote selected filelist: ${SEL_LIST}"
echo "✓ N selected files: $(wc -l < ${SEL_LIST})"

# -------------------------
# 2) Counts cube (CCUBE)
# -------------------------
COUNTS_CCUBE="${OUT_DIR}/counts_ccube_${EMIN}to${EMAX}.fits"

echo
echo "==> gtbin CCUBE -> ${COUNTS_CCUBE}"
gtbin \
  "evfile=@${SEL_LIST}" \
  "scfile=${SCFILE}" \
  "outfile=${COUNTS_CCUBE}" \
  "algorithm=CCUBE" \
  "coordsys=${COORDSYS}" \
  "proj=${PROJ}" \
  "axisrot=${AXISROT}" \
  "xref=${XREF}" "yref=${YREF}" \
  "nxpix=${NX}" "nypix=${NY}" \
  "binsz=${BINSZ}" \
  "ebinalg=LOG" \
  "emin=${EMIN}" "emax=${EMAX}" \
  "enumbins=${ENUMBINS}"

# Optional integrated counts image
COUNTS_CMAP="${OUT_DIR}/counts_cmap_${EMIN}to${EMAX}.fits"
echo
echo "==> gtbin CMAP -> ${COUNTS_CMAP}"
gtbin \
  "evfile=@${SEL_LIST}" \
  "scfile=${SCFILE}" \
  "outfile=${COUNTS_CMAP}" \
  "algorithm=CMAP" \
  "coordsys=${COORDSYS}" \
  "proj=${PROJ}" \
  "axisrot=${AXISROT}" \
  "xref=${XREF}" "yref=${YREF}" \
  "nxpix=${NX}" "nypix=${NY}" \
  "binsz=${BINSZ}"

# -------------------------
# 3) Livetime cube (gtltcube)
# -------------------------
LTCUBE="${OUT_DIR}/ltcube_${EMIN}to${EMAX}.fits"

echo
echo "==> gtltcube -> ${LTCUBE}"
gtltcube \
  "evfile=@${SEL_LIST}" \
  "scfile=${SCFILE}" \
  "outfile=${LTCUBE}" \
  "zmax=${ZMAX}" \
  "dcostheta=0.025" \
  "binsz=1"

# -------------------------
# 4) Exposure cube (gtexpcube2)
# -------------------------
EXPCUBE="${OUT_DIR}/expcube_${EMIN}to${EMAX}.fits"

echo
echo "==> gtexpcube2 -> ${EXPCUBE}"
gtexpcube2 \
  "infile=${LTCUBE}" \
  "cmap=none" \
  "outfile=${EXPCUBE}" \
  "irfs=${IRFS}" \
  "coordsys=${COORDSYS}" \
  "proj=${PROJ}" \
  "axisrot=${AXISROT}" \
  "xref=${XREF}" "yref=${YREF}" \
  "nxpix=${NX}" "nypix=${NY}" \
  "binsz=${BINSZ}" \
  "ebinalg=LOG" \
  "emin=${EMIN}" "emax=${EMAX}" \
  "enumbins=${ENUMBINS}"

echo
echo "===================================="
echo "DONE"
echo "Counts CCUBE: ${COUNTS_CCUBE}"
echo "Counts CMAP : ${COUNTS_CMAP}"
echo "LTCUBE      : ${LTCUBE}"
echo "EXPCUBE     : ${EXPCUBE}"
echo "Filelist    : ${SEL_LIST}"
echo "===================================="
