#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Build Totani-style Fermi-LAT products from 4 downloaded
# LAT query folders, merging and de-duplicating overlapping
# event files before running Fermitools.
#
# Usage:
#   bash process_totani_full.sh
#
# Optional overrides:
#   bash process_totani_full.sh QUERY_DIR1 QUERY_DIR2 QUERY_DIR3 QUERY_DIR4 [OUT_DIR]
#
# Notes:
# - Assumes each query folder contains:
#     * one photon event FITS file
#     * one spacecraft FITS file
# - Overlapping query regions create duplicate events.
#   We remove duplicates using EVENT_ID + RUN_ID if present.
# ============================================================

# Resolve repo root from script location if REPO_DIR is not set
if [[ -z "${REPO_DIR:-}" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

DATA_ROOT="${REPO_DIR}/fermi_data/totani"

if [[ "$#" -ge 4 ]]; then
  QUERY_DIRS=("$1" "$2" "$3" "$4")
else
  QUERY_DIRS=(
    "${DATA_ROOT}/300_0"
    "${DATA_ROOT}/60_0"
    "${DATA_ROOT}/0_-30"
    "${DATA_ROOT}/0_30"
  )
fi

if [[ "$#" -ge 5 ]]; then
  OUT_DIR="$5"
else
  OUT_DIR="${DATA_ROOT}/processed"
fi

mkdir -p "${OUT_DIR}"

# Auto-enable a disk-safe pipeline on low free space.
# gtmktime can create a temp output nearly as large as the input evfile.
PER_FILE_FILTER="${PER_FILE_FILTER:-0}"
if [[ "${PER_FILE_FILTER}" == "0" ]]; then
  if df_out=$(df -Pk "${OUT_DIR}" 2>/dev/null | tail -n 1); then
    avail_kb=$(echo "${df_out}" | awk '{print $4}')
    if [[ -n "${avail_kb}" ]]; then
      avail_gb=$((avail_kb / 1024 / 1024))
      if [[ "${avail_gb}" -lt 40 ]]; then
        PER_FILE_FILTER="1"
      fi
    fi
  fi
fi

echo "==> Using query folders:"
for qdir in "${QUERY_DIRS[@]}"; do
  echo "  - ${qdir}"
  if [[ ! -d "${qdir}" ]]; then
    echo "ERROR: Missing query folder: ${qdir}" >&2
    exit 1
  fi
done
echo "==> Output directory: ${OUT_DIR}"

need_cmd() {
  local c="$1"
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: ${c}" >&2
    echo "  PATH=${PATH}" >&2
    echo "  If this is a Fermitools command, make sure your Fermitools environment is activated (e.g. conda env) before running this script." >&2
    return 1
  fi
  return 0
}

need_any() {
  local c1="$1"
  local c2="$2"
  if command -v "${c1}" >/dev/null 2>&1; then
    echo "${c1}"
    return 0
  fi
  if command -v "${c2}" >/dev/null 2>&1; then
    echo "${c2}"
    return 0
  fi
  return 1
}

FITS_MERGE_TOOL=""
if command -v ftmerge >/dev/null 2>&1; then
  FITS_MERGE_TOOL="ftmerge"
else
  FITS_MERGE_TOOL="python"
fi

PYTHON_BIN=""

need_cmd gtmktime
need_cmd gtselect
need_cmd gtbin
need_cmd gtltcube
need_cmd gtexpcube2

# Always pick a python interpreter for the deduplication step (and merge fallback)
if command -v python3 >/dev/null 2>&1 && python3 -c "import astropy" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1 && python -c "import astropy" >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  if [[ "${FITS_MERGE_TOOL}" == "python" ]]; then
    echo "ERROR: ftmerge not found, and no available python interpreter can import astropy for FITS merging." >&2
    echo "Tried: python3, python" >&2
    echo "Fix options:" >&2
    echo "  - Activate the environment where astropy is installed, then re-run this script." >&2
    echo "  - Or install astropy into the python on PATH." >&2
    echo "  - Or install HEASoft to provide ftmerge." >&2
    exit 1
  fi
  echo "ERROR: No available python interpreter can import astropy (needed for deduplication step)." >&2
  echo "Tried: python3, python" >&2
  echo "Fix options:" >&2
  echo "  - Activate the environment where astropy is installed, then re-run this script." >&2
  echo "  - Or install astropy into the python on PATH." >&2
  exit 1
fi

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

# Energy range: full Totani-like setup
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
IRFS="P8R3_ULTRACLEANVETO_V3"

# Large cone used only for gtselect prefilter.
# Must be large enough not to clip the |l|,|b|<=60 square.
# 90 deg is safe.
SEL_RA=266.4051
SEL_DEC=-28.936175
SEL_RAD=90

# File products
PHLIST_RAW="${OUT_DIR}/photon_raw_list.txt"
SCLIST_RAW="${OUT_DIR}/spacecraft_raw_list.txt"
PHMERGED="${OUT_DIR}/merged_all_queries_raw.fits"
SCFILE="${OUT_DIR}/spacecraft_merged.fits"
GTI_FILE="${OUT_DIR}/events_gti.fits"
SEL_FILE="${OUT_DIR}/events_gti_sel.fits"
PHDEDUP="${OUT_DIR}/events_gti_sel_dedup.fits"

SEL_LIST="${OUT_DIR}/sel_gti_list.txt"

COUNTS_CCUBE="${OUT_DIR}/counts_ccube_${EMIN}to${EMAX}.fits"
COUNTS_CMAP="${OUT_DIR}/counts_cmap_${EMIN}to${EMAX}.fits"
LTCUBE="${OUT_DIR}/ltcube_${EMIN}to${EMAX}.fits"
EXPCUBE="${OUT_DIR}/expcube_${EMIN}to${EMAX}.fits"

# Optional cleanup of intermediate outputs (does NOT touch raw query folders)
if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -f \
    "${PHMERGED}" \
    "${SCFILE}" \
    "${GTI_FILE}" \
    "${SEL_FILE}" \
    "${PHDEDUP}" \
    "${SEL_LIST}" \
    "${COUNTS_CCUBE}" \
    "${COUNTS_CMAP}" \
    "${LTCUBE}" \
    "${EXPCUBE}"

  # Remove per-file intermediates from disk-safe mode + any gtmktime temp outputs
  rm -f "${OUT_DIR}"/*_gti.fits "${OUT_DIR}"/*_gti_sel.fits \
        "${OUT_DIR}"/events_gti.fits_tempgti "${OUT_DIR}"/events_gti.fits_temp*
fi

# ------------------------------------------------------------
# 0) Discover photon + spacecraft FITS files in the 4 folders
# ------------------------------------------------------------
: > "${PHLIST_RAW}"
: > "${SCLIST_RAW}"

echo "==> Scanning query folders"

for qdir in "${QUERY_DIRS[@]}"; do
  echo "  -> ${qdir}"

  # Photon files: try common naming patterns first, then fallback
  find "${qdir}" -type f \( \
      -iname "*PH*.fits" -o \
      -iname "*EV*.fits" -o \
      -iname "*photon*.fits" \
    \) | sort >> "${PHLIST_RAW}"

  # Spacecraft files
  find "${qdir}" -type f \( \
      -iname "*SC*.fits" -o \
      -iname "*spacecraft*.fits" \
    \) | sort >> "${SCLIST_RAW}"
done

# Remove duplicate path lines if any
sort -u "${PHLIST_RAW}" -o "${PHLIST_RAW}"
sort -u "${SCLIST_RAW}" -o "${SCLIST_RAW}"

NPH=$(wc -l < "${PHLIST_RAW}")
NSC=$(wc -l < "${SCLIST_RAW}")

echo "Found ${NPH} photon file(s)"
echo "Found ${NSC} spacecraft file(s)"

if [ "${NPH}" -eq 0 ]; then
  echo "ERROR: No photon FITS files found."
  exit 1
fi

if [ "${NSC}" -eq 0 ]; then
  echo "ERROR: No spacecraft FITS files found."
  exit 1
fi

echo
echo "Photon files:"
cat "${PHLIST_RAW}"

echo
echo "Spacecraft files:"
cat "${SCLIST_RAW}"

# ------------------------------------------------------------
# 1) Spacecraft file
# ------------------------------------------------------------
echo
if [[ "${NSC}" -eq 1 ]]; then
  ONLY_SC="$(head -n 1 "${SCLIST_RAW}")"
  echo "==> Single spacecraft file detected; using it directly: ${ONLY_SC}"
  ln -sf "${ONLY_SC}" "${SCFILE}"
elif [[ "${MERGE_SC:-0}" != "1" ]]; then
  FIRST_SC="$(head -n 1 "${SCLIST_RAW}")"
  echo "==> Using first spacecraft file only (set MERGE_SC=1 to force merge): ${FIRST_SC}"
  ln -sf "${FIRST_SC}" "${SCFILE}"
else
  echo "==> Merging spacecraft files -> ${SCFILE}"
  paste -sd' ' "${SCLIST_RAW}" > "${OUT_DIR}/_sc_merge_args.txt"
  SC_MERGE_IN=$(cat "${OUT_DIR}/_sc_merge_args.txt")
  if [[ "${FITS_MERGE_TOOL}" == "ftmerge" ]]; then
    ftmerge infile="${SC_MERGE_IN}" outfile="${SCFILE}" clobber=yes
  else
    OUT_DIR="${OUT_DIR}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
from astropy.io import fits
from astropy.table import Table, vstack

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

out_dir = Path(os.environ["OUT_DIR"])
sc_list = (out_dir / "spacecraft_raw_list.txt").read_text().strip().splitlines()
outfile = out_dir / "spacecraft_merged.fits"

if len(sc_list) == 0:
    raise SystemExit("No spacecraft files to merge")

with fits.open(sc_list[0], memmap=True) as h0:
    primary = h0[0].copy()

tables_by_key = {}
headers_by_key = {}

def _key_for_hdu(hdu, idx):
    name = getattr(hdu, "name", "")
    if isinstance(name, str) and name.strip():
        return name.strip().upper()
    return f"EXT_{idx}"

sc_iter = tqdm(sc_list, desc="merge(spacecraft)", unit="file") if tqdm is not None else sc_list
for p in sc_iter:
    with fits.open(p, memmap=True) as hdul:
        for idx in range(1, len(hdul)):
            hdu = hdul[idx]
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            k = _key_for_hdu(hdu, idx)
            if k not in headers_by_key:
                headers_by_key[k] = hdu.header
            tables_by_key.setdefault(k, []).append(Table(hdu.data))

out = fits.HDUList([primary])
for k, tlist in tables_by_key.items():
    merged = vstack(tlist, metadata_conflicts="silent").as_array()
    out.append(fits.BinTableHDU(data=merged, header=headers_by_key[k], name=k))

out.writeto(outfile, overwrite=True)
PY
  fi
fi

if [[ ! -f "${SCFILE}" ]]; then
  echo "ERROR: Spacecraft file was not created: ${SCFILE}" >&2
  exit 1
fi

# ------------------------------------------------------------
# 2) Merge all photon event files
# ------------------------------------------------------------
if [[ "${PER_FILE_FILTER}" != "1" ]]; then
  echo
  echo "==> Merging photon files -> ${PHMERGED}"
  paste -sd' ' "${PHLIST_RAW}" > "${OUT_DIR}/_ph_merge_args.txt"
  PH_MERGE_IN=$(cat "${OUT_DIR}/_ph_merge_args.txt")

  if [[ -f "${PHMERGED}" ]]; then
    echo "==> Photon merged file already exists; skipping: ${PHMERGED}"
  elif [[ "${FITS_MERGE_TOOL}" == "ftmerge" ]]; then
    ftmerge infile="${PH_MERGE_IN}" outfile="${PHMERGED}" clobber=yes
  else
    OUT_DIR="${OUT_DIR}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
import numpy as np
from astropy.io import fits

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import fitsio
except Exception as e:
    raise SystemExit(
        "Python merge fallback requires fitsio for speed/low-memory. "
        "Install it (recommended): conda install -c conda-forge fitsio  OR  pip install fitsio\n"
        f"Import error: {e}"
    )

out_dir = Path(os.environ["OUT_DIR"])
ph_list = (out_dir / "photon_raw_list.txt").read_text().strip().splitlines()
outfile = out_dir / "merged_all_queries_raw.fits"

if len(ph_list) == 0:
    raise SystemExit("No photon files to merge")

chunk = int(os.environ.get("PHOTON_MERGE_CHUNK", "500000"))
progress_every = int(os.environ.get("PHOTON_MERGE_PROGRESS_EVERY", "10"))

def _pbar(done, total):
    if total <= 0:
        msg = f"merge(py): {done} rows"
        print("\r" + msg.ljust(80), end="", flush=True)
        return
    frac = min(1.0, max(0.0, done / total))
    width = 30
    filled = int(width * frac)
    bar = "=" * filled + ">" + "." * max(0, width - filled - 1)
    pct = 100.0 * frac
    msg = f"merge(py): [{bar}] {pct:6.2f}%  ({done:,}/{total:,} rows)"
    print("\r" + msg[:120].ljust(120), end="", flush=True)

def _find_events_hdu(hdul):
    for i, h in enumerate(hdul):
        if getattr(h, "name", "").upper() == "EVENTS":
            return i
    return None

def _find_events_ext_fitsio(f):
    for i in range(1, len(f)):
        try:
            nm = (f[i].get_extname() or "").upper()
        except Exception:
            nm = ""
        if nm == "EVENTS":
            return i
    return None

with fits.open(ph_list[0], memmap=True) as hdul0:
    evt_i0 = _find_events_hdu(hdul0)
    if evt_i0 is None:
        raise RuntimeError(f"No EVENTS extension found in {ph_list[0]}")
    primary = hdul0[0].copy()
    evt0 = hdul0[evt_i0]

    cols = []
    for c in evt0.columns:
        cols.append(
            fits.Column(
                name=c.name,
                format=c.format,
                unit=getattr(c, "unit", None),
                dim=getattr(c, "dim", None),
            )
        )
    out_cols = fits.ColDefs(cols)
    empty_evt = fits.BinTableHDU.from_columns(out_cols, nrows=0, header=evt0.header.copy(), name="EVENTS")
    fits.HDUList([primary, empty_evt]).writeto(outfile, overwrite=True)

total_rows = 0
for p in ph_list:
    with fitsio.FITS(p) as fin:
        evt_ext = _find_events_ext_fitsio(fin)
        if evt_ext is None:
            raise RuntimeError(f"No EVENTS extension found in {p}")
        total_rows += fin[evt_ext].get_nrows()

with fitsio.FITS(str(outfile), "rw") as out:
    out_evt = out["EVENTS"]
    written = 0

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_rows, desc="merge(photons)", unit="rows")
    else:
        _pbar(written, total_rows)

    for file_i, p in enumerate(ph_list):
        with fitsio.FITS(p) as fin:
            evt_ext = _find_events_ext_fitsio(fin)
            if evt_ext is None:
                raise RuntimeError(f"No EVENTS extension found in {p}")
            nrows = fin[evt_ext].get_nrows()
            for j, i0 in enumerate(range(0, nrows, chunk)):
                i1 = min(nrows, i0 + chunk)
                rows = range(i0, i1)
                data = fin[evt_ext].read(rows=rows)
                out_evt.append(data)
                written += len(data)
                if pbar is not None:
                    pbar.update(len(data))
                elif (j % progress_every) == 0:
                    _pbar(written, total_rows)

    if pbar is not None:
        pbar.close()

print("\n")
PY
  fi

  if [[ ! -f "${PHMERGED}" ]]; then
    echo "ERROR: Photon merged file was not created: ${PHMERGED}" >&2
    exit 1
  fi

  # gtmktime expects the input evfile to have a GTI extension (it opens evfile[GTI]).
  # Some merge paths may drop non-EVENTS extensions, so ensure GTI exists.
  OUT_DIR="${OUT_DIR}" IN_FITS="${PHMERGED}" LIST_FILE="${PHLIST_RAW}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
from astropy.io import fits

merged = Path(os.environ["IN_FITS"])
list_file = Path(os.environ["LIST_FILE"])
ph_list = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
if not ph_list:
    raise SystemExit("No photon files listed; cannot validate GTI")

src0 = Path(ph_list[0])

def _has_ext(path: Path, name: str) -> bool:
    with fits.open(path, memmap=True) as hdul:
        for h in hdul:
            if getattr(h, "name", "").upper() == name.upper():
                return True
    return False

if _has_ext(merged, "GTI"):
    raise SystemExit(0)

with fits.open(src0, memmap=True) as hdul0:
    gti_hdu = None
    for h in hdul0:
        if getattr(h, "name", "").upper() == "GTI":
            gti_hdu = h.copy()
            break

if gti_hdu is None:
    raise SystemExit(f"ERROR: Merged evfile is missing GTI and source file has no GTI: {src0}")

with fits.open(merged, mode="update", memmap=True) as hdul:
    hdul.append(gti_hdu)
    hdul.flush()
PY
fi

# ------------------------------------------------------------
# 3-4) GTI + selection
# ------------------------------------------------------------
if [[ "${PER_FILE_FILTER}" == "1" ]]; then
  echo
  echo "==> Low disk mode: running gtmktime+gtselect per photon file"
  : > "${SEL_LIST}"

  i=0
  while IFS= read -r PH; do
    [[ -z "${PH}" ]] && continue
    i=$((i+1))
    base="$(basename "${PH}" .fits)"
    gti="${OUT_DIR}/${base}_gti.fits"
    sel="${OUT_DIR}/${base}_gti_sel.fits"

    # Restart-safe: if the selected file already exists, reuse it.
    if [[ -f "${sel}" ]]; then
      echo
      echo "[${i}] Reusing existing selected file: ${sel}"
      echo "${sel}" >> "${SEL_LIST}"
      continue
    fi

    echo
    echo "[${i}] gtmktime -> ${gti}"
    if [[ -f "${gti}" ]]; then
      echo "[${i}] Reusing existing GTI file: ${gti}"
    else
      gtmktime \
        "scfile=${SCFILE}" \
        "evfile=${PH}" \
        "outfile=${gti}" \
        "roicut=no" \
        "filter=${GTI_FILTER}"
    fi

    echo "[${i}] gtselect -> ${sel}"
    gtselect \
      "infile=${gti}" \
      "outfile=${sel}" \
      "ra=${SEL_RA}" "dec=${SEL_DEC}" "rad=${SEL_RAD}" \
      "tmin=INDEF" "tmax=INDEF" \
      "emin=${EMIN}" "emax=${EMAX}" \
      "zmax=${ZMAX}" \
      "evclass=${EVCLASS}" \
      "evtype=${EVTYPE}"

    echo "${sel}" >> "${SEL_LIST}"
  done < "${PHLIST_RAW}"

  NSEL=$(wc -l < "${SEL_LIST}" | tr -d ' ')
  if [[ "${NSEL}" -eq 0 ]]; then
    echo "ERROR: No selected files produced in low disk mode." >&2
    exit 1
  fi

  echo
  echo "==> Merging selected files -> ${SEL_FILE}"
  paste -sd' ' "${SEL_LIST}" > "${OUT_DIR}/_sel_merge_args.txt"
  SEL_MERGE_IN=$(cat "${OUT_DIR}/_sel_merge_args.txt")

  if [[ -f "${SEL_FILE}" ]]; then
    echo "==> Selected merged file already exists; skipping: ${SEL_FILE}"
  elif [[ "${FITS_MERGE_TOOL}" == "ftmerge" ]]; then
    ftmerge infile="${SEL_MERGE_IN}" outfile="${SEL_FILE}" clobber=yes
  else
    OUT_DIR="${OUT_DIR}" IN_LIST="${SEL_LIST}" OUT_FITS="${SEL_FILE}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
from astropy.io import fits

in_list = Path(os.environ["IN_LIST"])
outfile = Path(os.environ["OUT_FITS"])
paths = [ln.strip() for ln in in_list.read_text().splitlines() if ln.strip()]
if not paths:
    raise SystemExit("No selected files to merge")

with fits.open(paths[0], memmap=True) as hdul0:
    primary = hdul0[0].copy()
    evt_i0 = None
    for i, h in enumerate(hdul0):
        if getattr(h, "name", "").upper() == "EVENTS":
            evt_i0 = i
            break
    if evt_i0 is None:
        raise RuntimeError(f"No EVENTS extension found in {paths[0]}")
    evt0 = hdul0[evt_i0]
    cols = []
    for c in evt0.columns:
        cols.append(
            fits.Column(
                name=c.name,
                format=c.format,
                unit=getattr(c, "unit", None),
                dim=getattr(c, "dim", None),
            )
        )
    out_cols = fits.ColDefs(cols)
    empty_evt = fits.BinTableHDU.from_columns(out_cols, nrows=0, header=evt0.header.copy(), name="EVENTS")
    fits.HDUList([primary, empty_evt]).writeto(outfile, overwrite=True)

try:
    import fitsio
except Exception as e:
    raise SystemExit(
        "Selected merge requires fitsio for speed/low-memory. "
        "Install it: conda install -c conda-forge fitsio  OR  pip install fitsio\n"
        f"Import error: {e}"
    )

def _find_events_ext_fitsio(f):
    for i in range(1, len(f)):
        try:
            nm = (f[i].get_extname() or "").upper()
        except Exception:
            nm = ""
        if nm == "EVENTS":
            return i
    return None

with fitsio.FITS(str(outfile), "rw") as out:
    out_evt = out["EVENTS"]
    for p in paths:
        with fitsio.FITS(p) as fin:
            evt_ext = _find_events_ext_fitsio(fin)
            if evt_ext is None:
                raise RuntimeError(f"No EVENTS extension found in {p}")
            nrows = fin[evt_ext].get_nrows()
            for i0 in range(0, nrows, 500000):
                i1 = min(nrows, i0 + 500000)
                data = fin[evt_ext].read(rows=range(i0, i1))
                out_evt.append(data)
PY
  fi

  if [[ ! -f "${SEL_FILE}" ]]; then
    echo "ERROR: Selected merged file was not created: ${SEL_FILE}" >&2
    exit 1
  fi

  # Ensure GTI exists on the selected merged file (tools open evfile[GTI])
  OUT_FITS="${SEL_FILE}" LIST_FILE="${SEL_LIST}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
from astropy.io import fits

outfile = Path(os.environ["OUT_FITS"])
list_file = Path(os.environ["LIST_FILE"])
paths = [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]
if not paths:
    raise SystemExit("No selected files listed; cannot validate GTI")

def _has_ext(path: Path, name: str) -> bool:
    with fits.open(path, memmap=True) as hdul:
        for h in hdul:
            if getattr(h, "name", "").upper() == name.upper():
                return True
    return False

if _has_ext(outfile, "GTI"):
    raise SystemExit(0)

src0 = Path(paths[0])
with fits.open(src0, memmap=True) as hdul0:
    gti_hdu = None
    for h in hdul0:
        if getattr(h, "name", "").upper() == "GTI":
            gti_hdu = h.copy()
            break

if gti_hdu is None:
    raise SystemExit(f"ERROR: Selected merged file is missing GTI and source file has no GTI: {src0}")

with fits.open(outfile, mode="update", memmap=True) as hdul:
    hdul.append(gti_hdu)
    hdul.flush()
PY

else
  # ------------------------------------------------------------
  # 3) Apply GTIs (full-merge mode)
  # ------------------------------------------------------------
  echo
  echo "==> gtmktime -> ${GTI_FILE}"
  gtmktime \
    "scfile=${SCFILE}" \
    "evfile=${PHMERGED}" \
    "outfile=${GTI_FILE}" \
    "roicut=no" \
    "filter=${GTI_FILTER}"

  # ------------------------------------------------------------
  # 4) Event selection (full-merge mode)
  # ------------------------------------------------------------
  echo
  echo "==> gtselect -> ${SEL_FILE}"
  gtselect \
    "infile=${GTI_FILE}" \
    "outfile=${SEL_FILE}" \
    "ra=${SEL_RA}" "dec=${SEL_DEC}" "rad=${SEL_RAD}" \
    "tmin=INDEF" "tmax=INDEF" \
    "emin=${EMIN}" "emax=${EMAX}" \
    "zmax=${ZMAX}" \
    "evclass=${EVCLASS}" \
    "evtype=${EVTYPE}"
fi

# # Create an empty output table with the full number of rows.
# out_evt = fits.BinTableHDU.from_columns(base_cols, nrows=total_rows, header=base_evt_hdr, name=base_evt_name)
# fits.HDUList([base0, out_evt]).writeto(outfile, overwrite=True)

# # Fill it in chunks.
# row0 = 0
# written = 0
# _pbar(0, total_rows, prefix="merge(astropy)")
# with fits.open(outfile, mode="update", memmap=True) as out_hdul:
#     out_evt_hdu = out_hdul[1]
#     for p in ph_list:
#         with fits.open(p, memmap=True) as hdul:
#             evt_i = _find_events_hdu(hdul)
#             evt = hdul[evt_i]
#             n = int(evt.header.get("NAXIS2", 0))
#             if n == 0:
#                 continue
#             for block_i, i0 in enumerate(range(0, n, chunk)):
#                 i1 = min(n, i0 + chunk)
#                 out_evt_hdu.data[row0 + i0: row0 + i1] = evt.data[i0:i1]
#                 written += (i1 - i0)
#                 if (block_i % progress_every) == 0:
#                     _pbar(written, total_rows, prefix="merge(astropy)")
#         row0 += n
#     out_hdul.flush()

# _pbar(total_rows, total_rows, prefix="merge")
# print("", flush=True)
# PY
# fi

# ------------------------------------------------------------
# 5) De-duplicate overlapping events (post-selection)
# ------------------------------------------------------------
echo
echo "==> De-duplicating selected photon events -> ${PHDEDUP}"

OUT_DIR="${OUT_DIR}" IN_FITS="${SEL_FILE}" OUT_FITS="${PHDEDUP}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
import numpy as np
from astropy.io import fits

infile = Path(os.environ["IN_FITS"])
outfile = Path(os.environ["OUT_FITS"])

with fits.open(infile, memmap=True) as hdul:
    evt_hdu_index = None
    for i, h in enumerate(hdul):
        if getattr(h, "name", "").upper() == "EVENTS":
            evt_hdu_index = i
            break
    if evt_hdu_index is None:
        raise RuntimeError("No EVENTS extension found in selected photon FITS file.")

    events_hdu = hdul[evt_hdu_index]
    data = events_hdu.data
    cols = data.columns.names

    print("EVENTS columns:", cols)

    if "EVENT_ID" in cols and "RUN_ID" in cols:
        key1 = np.asarray(data["RUN_ID"])
        key2 = np.asarray(data["EVENT_ID"])
        keys = np.rec.fromarrays([key1, key2], names="RUN_ID,EVENT_ID")
        _, unique_idx = np.unique(keys, return_index=True)
        key_desc = "RUN_ID + EVENT_ID"
    elif "EVENT_ID" in cols:
        key1 = np.asarray(data["EVENT_ID"])
        _, unique_idx = np.unique(key1, return_index=True)
        key_desc = "EVENT_ID"
    elif all(c in cols for c in ["TIME", "ENERGY", "RA", "DEC"]):
        key1 = np.round(np.asarray(data["TIME"]), 6)
        key2 = np.round(np.asarray(data["ENERGY"]), 3)
        key3 = np.round(np.asarray(data["RA"]), 5)
        key4 = np.round(np.asarray(data["DEC"]), 5)
        keys = np.rec.fromarrays([key1, key2, key3, key4], names="TIME,ENERGY,RA,DEC")
        _, unique_idx = np.unique(keys, return_index=True)
        key_desc = "TIME + ENERGY + RA + DEC (rounded fallback)"
    else:
        raise RuntimeError(
            "Could not find a safe deduplication key. "
            "Need EVENT_ID/RUN_ID or at least TIME/ENERGY/RA/DEC."
        )

    unique_idx = np.sort(unique_idx)
    dedup_data = data[unique_idx]

    print(f"Dedup key: {key_desc}")
    print(f"Input rows : {len(data)}")
    print(f"Output rows: {len(dedup_data)}")
    print(f"Removed    : {len(data) - len(dedup_data)}")

    new_hdul = fits.HDUList()
    for i, h in enumerate(hdul):
        if i == evt_hdu_index:
            new_hdu = fits.BinTableHDU(data=dedup_data, header=h.header, name=h.name)
            new_hdul.append(new_hdu)
        else:
            new_hdul.append(h.copy())

    new_hdul.writeto(outfile, overwrite=True)
PY

# Ensure GTI exists on the deduplicated file for downstream tools
OUT_FITS="${PHDEDUP}" SRC_FITS="${SEL_FILE}" "${PYTHON_BIN}" <<'PY'
import os
from pathlib import Path
from astropy.io import fits

outfile = Path(os.environ["OUT_FITS"])
src = Path(os.environ["SRC_FITS"])

def _has_ext(path: Path, name: str) -> bool:
    with fits.open(path, memmap=True) as hdul:
        for h in hdul:
            if getattr(h, "name", "").upper() == name.upper():
                return True
    return False

if _has_ext(outfile, "GTI"):
    raise SystemExit(0)

with fits.open(src, memmap=True) as hdul0:
    gti_hdu = None
    for h in hdul0:
        if getattr(h, "name", "").upper() == "GTI":
            gti_hdu = h.copy()
            break

if gti_hdu is None:
    raise SystemExit(f"ERROR: Dedup output missing GTI and source has no GTI: {src}")

with fits.open(outfile, mode="update", memmap=True) as hdul:
    hdul.append(gti_hdu)
    hdul.flush()
PY

# ------------------------------------------------------------
# 6) Counts cube (CCUBE)
# ------------------------------------------------------------
echo
echo "==> gtbin CCUBE -> ${COUNTS_CCUBE}"
gtbin \
  "evfile=${PHDEDUP}" \
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

# ------------------------------------------------------------
# 7) Optional integrated counts image
# ------------------------------------------------------------
echo
echo "==> gtbin CMAP -> ${COUNTS_CMAP}"
gtbin \
  "evfile=${PHDEDUP}" \
  "scfile=${SCFILE}" \
  "outfile=${COUNTS_CMAP}" \
  "algorithm=CMAP" \
  "coordsys=${COORDSYS}" \
  "proj=${PROJ}" \
  "axisrot=${AXISROT}" \
  "xref=${XREF}" "yref=${YREF}" \
  "nxpix=${NX}" "nypix=${NY}" \
  "binsz=${BINSZ}"

# ------------------------------------------------------------
# 8) Livetime cube
# ------------------------------------------------------------
echo
echo "==> gtltcube -> ${LTCUBE}"
gtltcube \
  "evfile=${PHDEDUP}" \
  "scfile=${SCFILE}" \
  "outfile=${LTCUBE}" \
  "zmax=${ZMAX}" \
  "dcostheta=0.025" \
  "binsz=1"

# ------------------------------------------------------------
# 9) Exposure cube
# ------------------------------------------------------------
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
echo "Photon merged raw : ${PHMERGED}"
echo "Photon deduped    : ${PHDEDUP}"
echo "Spacecraft merged : ${SCFILE}"
echo "GTI file          : ${GTI_FILE}"
echo "Selected events   : ${SEL_FILE}"
echo "Counts CCUBE      : ${COUNTS_CCUBE}"
echo "Counts CMAP       : ${COUNTS_CMAP}"
echo "LTCUBE            : ${LTCUBE}"
echo "EXPCUBE           : ${EXPCUBE}"
echo "===================================="