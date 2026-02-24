#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import read_counts_and_ebounds

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
COUNTS = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
PSC = os.path.join(DATA_DIR, "templates", "gll_psc_v35.fit")
OUT = os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits")

# Spec requirement:
# Mask a circular region of radius = 2 * (catalog semi-major axis) for each extended source.
R_FACTOR = 2.0


def as_str(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode(errors="ignore")
    return str(x)


def effective_radius_deg(model_form, spatial_func, semi_major, semi_minor):
    """
    Convert catalog morphology parameters into a radius (deg) for a *circular* mask.
    Per spec we use: radius = R_FACTOR * semi-major axis.
    """
    mf = as_str(model_form).strip().lower()
    sf = as_str(spatial_func).strip().lower()

    a = float(semi_major) if np.isfinite(semi_major) else np.nan
    b = float(semi_minor) if np.isfinite(semi_minor) else np.nan

    # Default: use semi-major if present
    if not np.isfinite(a) or a <= 0:
        return np.nan

    return float(R_FACTOR) * a


# ------------------
# Load counts cube geometry
# ------------------
counts, hdr, _Emin_mev, _Emax_mev, Ectr_mev, _dE_mev = read_counts_and_ebounds(COUNTS)
nE, ny, nx = counts.shape
Ectr_GeV = np.asarray(Ectr_mev, dtype=float) / 1000.0
print("Energy bin centers (GeV):", np.round(Ectr_GeV, 3))

# Pixel coords
wcs = WCS(hdr).celestial
yy, xx = np.mgrid[0:ny, 0:nx]
lon, lat = wcs.pixel_to_world_values(xx, yy)
pix_coords = SkyCoord(lon * u.deg, lat * u.deg, frame="galactic")

# ------------------
# Load ExtendedSources table
# ------------------
with fits.open(PSC) as f:
    ext = f["ExtendedSources"].data

print("N extended sources:", len(ext))

# Build list of extended sources with per-source effective radius
ext_list = []
for row in ext:
    name = as_str(row["Source_Name"]).strip()
    glon = float(row["GLON"])
    glat = float(row["GLAT"])
    mf   = row["Model_Form"]
    sf   = row["Spatial_Function"]
    a    = row["Model_SemiMajor"]
    b    = row["Model_SemiMinor"]
    r_ext = effective_radius_deg(mf, sf, a, b)  # deg (containment-ish)
    ext_list.append((name, glon, glat, r_ext, as_str(mf), as_str(sf), as_str(row["Spatial_Filename"])))

# quick debug print
for (name, glon, glat, r_ext, mf, sf, fn) in ext_list[:10]:
    print(f"{name:25s}  l,b=({glon:7.2f},{glat:7.2f})  Rext={r_ext:6.3f} deg  form={mf}  func={sf}")

# ------------------
# Build 2D keep mask
# True=keep, False=masked
# ------------------
mask2d = np.ones((ny, nx), dtype=bool)

for (name, glon, glat, r_ext, mf, sf, fn) in ext_list:
    if not np.isfinite(r_ext) or (r_ext <= 0):
        continue
    src = SkyCoord(glon * u.deg, glat * u.deg, frame="galactic")
    sep = pix_coords.separation(src).deg
    mask2d[sep < float(r_ext)] = False

print("Total masked fraction:", 1.0 - mask2d.mean())

# ------------------
# Save
# ------------------
hdu = fits.PrimaryHDU(mask2d.astype(np.uint8), header=hdr)
hdu.header["COMMENT"] = "Extended-source mask (1=keep, 0=masked)"
hdu.header["COMMENT"] = "Mask circles of radius = 2 * semi-major axis from gll_psc ExtendedSources catalog"
hdu.header["RFACT"] = (R_FACTOR, "Mask radius multiplier applied to Model_SemiMajor")

fits.HDUList([hdu]).writeto(OUT, overwrite=True)
print("\nWrote:", OUT)
