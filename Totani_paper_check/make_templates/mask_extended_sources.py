#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
COUNTS = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
PSC = os.path.join(DATA_DIR, "templates", "gll_psc_v35.fit")
OUT = os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits")

# ------------------
# PSF model (68%) in deg
# ------------------
def psf68_deg(E_GeV):
    return np.maximum(0.8 * (E_GeV)**(-0.8), 0.2)

# Inflate everything slightly (recommended)
EXT_FACTOR = 1.2

# Choose containment for Gaussian-like models:
GAUSS_CONTAIN = "95"  # "68" or "95"
GAUSS_K = 2.45 if GAUSS_CONTAIN == "95" else 1.51  # R_contain ≈ k*sigma for 2D Gaussian

# How to combine extension and PSF:
COMBINE = "quadrature"  # "sum" or "quadrature"


def as_str(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode(errors="ignore")
    return str(x)


def effective_radius_deg(model_form, spatial_func, semi_major, semi_minor):
    """
    Convert catalog morphology parameters into an effective containment radius (deg)
    for a *circular* mask.
    We use semi-major as conservative scale; interpret Gaussian with GAUSS_K.
    """
    mf = as_str(model_form).strip().lower()
    sf = as_str(spatial_func).strip().lower()

    a = float(semi_major) if np.isfinite(semi_major) else np.nan
    b = float(semi_minor) if np.isfinite(semi_minor) else np.nan

    # Default: use semi-major if present
    if not np.isfinite(a) or a <= 0:
        return np.nan

    # Heuristics: detect gaussian vs disk-like forms
    # (naming varies; adjust if you see different strings)
    is_gauss = ("gauss" in mf) or ("gauss" in sf)
    is_disk  = ("disk" in mf) or ("disk" in sf)

    if is_gauss:
        # treat a as sigma along major axis
        return GAUSS_K * a

    if is_disk:
        # treat a as disk radius
        return a

    # generic ellipse: take semi-major as conservative radius
    return a


def combine_radii(r_ext, r_psf):
    if not np.isfinite(r_ext) or r_ext <= 0:
        return r_psf
    if COMBINE == "sum":
        return r_ext + r_psf
    # default quadrature
    return np.sqrt(r_ext**2 + r_psf**2)


# ------------------
# Load counts cube geometry
# ------------------
with fits.open(COUNTS) as h:
    counts = h[0].data
    hdr    = h[0].header
    eb     = h["EBOUNDS"].data

nE, ny, nx = counts.shape
Ectr_GeV = np.sqrt(eb["E_MIN"] * eb["E_MAX"]) / 1000.0
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
# Build mask cube
# True=keep, False=masked
# ------------------
mask = np.ones((nE, ny, nx), dtype=bool)

for k in range(nE):
    r_psf = psf68_deg(Ectr_GeV[k])  # deg

    for (name, glon, glat, r_ext, mf, sf, fn) in ext_list:
        r = combine_radii(r_ext, r_psf) * EXT_FACTOR  # deg

        src = SkyCoord(glon * u.deg, glat * u.deg, frame="galactic")
        sep = pix_coords.separation(src).deg
        mask[k][sep < r] = False

    if k % max(1, nE // 10) == 0:
        print(f"bin {k:02d}: PSF68={r_psf:.3f} deg, masked frac so far={1.0-mask[k].mean():.4f}")

# ------------------
# Save
# ------------------
hdu = fits.PrimaryHDU(mask.astype(np.uint8), header=hdr)
hdu.header["COMMENT"] = "Extended-source mask (1=keep, 0=masked)"
hdu.header["COMMENT"] = f"Rmask(E) = EXT_FACTOR * combine(Rext_contain, PSF68(E)), combine={COMBINE}"
hdu.header["EXTFAC"] = (EXT_FACTOR, "Multiplier on total mask radius")
hdu.header["GCONT"]  = (GAUSS_CONTAIN, "Containment used if Gaussian model")
hdu.header["COMBINE"] = (COMBINE, "How extension and PSF radii are combined")

fits.HDUList([hdu]).writeto(OUT, overwrite=True)
print("\nWrote:", OUT)
print("Total masked fraction:", 1.0 - mask.mean())
