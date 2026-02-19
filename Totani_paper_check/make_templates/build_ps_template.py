#!/usr/bin/env python3
# build_ps_template.py
#
# Outputs (all on counts_ccube grid):
#   - mu_ps_counts.fits   : counts        (expected counts per pixel per bin)
#   - ps_dnde.fits        : ph cm^-2 s^-1 sr^-1 MeV^-1
#   - ps_E2dnde.fits      : MeV cm^-2 s^-1 sr^-1   (Totani y-axis quantity per pixel)
#
# Built from 4FGL Flux1000 (1-100 GeV integral) with a power-law approx per source,
# then PSF-smoothed energy-dependently, conserving integrated bin flux.

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    write_cube,
)


# --------------------------------------------------
# FILES
# --------------------------------------------------
REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
COUNTS = os.path.join(DATA_DIR, "processed/counts_ccube_1000to1000000.fits")
EXPO   = os.path.join(DATA_DIR, "processed/expcube_1000to1000000.fits")
PSC    = os.path.join(DATA_DIR, "templates/gll_psc_v35.fit")

OUTDIR = os.path.join(DATA_DIR, "processed", "templates")
OUT_MU   = os.path.join(OUTDIR, "mu_ps_counts.fits")
OUT_DNDE = os.path.join(OUTDIR, "ps_dnde.fits")
OUT_E2   = os.path.join(OUTDIR, "ps_E2dnde.fits")

os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
BINSZ_DEG = 0.125

# LAT-like PSF width (deg) with floor
def psf_sigma_deg(E_GeV):
    return np.maximum(0.8 * (E_GeV) ** (-0.8), 0.2)

# Catalog definition of Flux1000
Ecat1, Ecat2 = 1.0, 100.0  # GeV


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _to_str(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode(errors="ignore")
    return str(x)


# --------------------------------------------------
# LOAD COUNTS HEADER / ENERGY BINS
# --------------------------------------------------
_, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(COUNTS)
wcs = WCS(hdr).celestial
ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
nE = int(Ectr_mev.size)
print("[E] counts EBOUNDS:")
print("[E]   Emin_MeV:", np.round(Emin_mev, 6))
print("[E]   Emax_MeV:", np.round(Emax_mev, 6))
print("[E]   Ectr_MeV:", np.round(Ectr_mev, 6))
print("[E]   Ectr_GeV:", np.round(Ectr_mev / 1000.0, 6))
print("[E]   dE_MeV  :", np.round(dE_mev, 6))

# Solid angle map
omega = pixel_solid_angle_map(wcs, ny, nx, BINSZ_DEG)


# --------------------------------------------------
# LOAD EXPOSURE (cm^2 s), resample if needed
# --------------------------------------------------
expo_raw, E_expo_mev = read_exposure(EXPO)
print("[E] expcube planes (raw):", expo_raw.shape)

expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
if expo.shape != (nE, ny, nx):
    raise RuntimeError(f"Exposure shape {expo.shape} not compatible with {(nE, ny, nx)}")

print("[E] expcube planes (resampled):", expo.shape)


# --------------------------------------------------
# LOAD POINT SOURCE CATALOG (use only sources within 90° of GC)
# --------------------------------------------------
psc = fits.open(PSC)[1].data

gc = SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic")
src_coord_all = SkyCoord(psc["GLON"] * u.deg, psc["GLAT"] * u.deg, frame="galactic")
roi90 = src_coord_all.separation(gc) <= 90 * u.deg
psc = psc[roi90]
print(f"[PS] Using {len(psc)} point sources in ROI (<=90° from GC)")


# --------------------------------------------------
# BUILD integrated flux per bin per pixel:
#   phi_bin_map[k, pix] = Φ_bin(pix) [ph / cm^2 / s]
# --------------------------------------------------
phi_bin_map = np.zeros((nE, ny, nx), dtype=float)

for src in psc:
    F = float(src["Flux1000"])  # ph / cm^2 / s (integral 1-100 GeV)
    if not np.isfinite(F) or F <= 0:
        continue

    stype = _to_str(src["SpectrumType"]).strip()

    if stype == "PowerLaw":
        gamma = float(src["PL_Index"])
    elif stype == "LogParabola":
        # crude fallback; good enough to get a PS template shape
        gamma = float(src["LP_Index"])
    else:
        continue

    if (not np.isfinite(gamma)) or (gamma <= 0) or (abs(gamma - 1.0) < 1e-8):
        continue

    # Normalise K so that ∫_{1}^{100} K E^{-gamma} dE = Flux1000
    # K in ph/(cm^2 s GeV)
    K = F * (1.0 - gamma) / (Ecat2 ** (1.0 - gamma) - Ecat1 ** (1.0 - gamma))

    coord = SkyCoord(src["GLON"] * u.deg, src["GLAT"] * u.deg, frame="galactic")
    ix, iy = wcs.world_to_pixel(coord)
    ix, iy = int(np.round(ix)), int(np.round(iy))
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        continue

    # Integrated flux in each analysis bin
    for k in range(nE):
        e1, e2 = Emin_mev[k], Emax_mev[k]
        phi = K * (e2 ** (1.0 - gamma) - e1 ** (1.0 - gamma)) / (1.0 - gamma)  # ph/(cm^2 s)
        phi_bin_map[k, iy, ix] += phi


# --------------------------------------------------
# Convert integrated flux to expected counts per bin per pixel:
#   mu_ps_raw = Φ_bin * exposure
# Units: counts
# --------------------------------------------------
mu_ps = phi_bin_map * expo


# --------------------------------------------------
# PSF smoothing (energy-dependent), conserving total counts per bin
# --------------------------------------------------
for k in range(nE):
    sigma_pix = psf_sigma_deg(Ectr_mev[k] / 1000.0) / BINSZ_DEG
    if sigma_pix <= 0:
        continue
    before = np.nansum(mu_ps[k])
    mu_ps[k] = gaussian_filter(mu_ps[k], sigma=sigma_pix, mode="constant", cval=0.0)
    after = np.nansum(mu_ps[k])
    if after > 0 and np.isfinite(before):
        mu_ps[k] *= (before / after)


# --------------------------------------------------
# Convert to dN/dE (intensity):
#   dnde = mu_ps / (exposure * Ω_pix * ΔE)
# Units: ph cm^-2 s^-1 sr^-1 MeV^-1
# --------------------------------------------------
dnde = np.full_like(mu_ps, np.nan, dtype=float)
denom = expo * omega[None, :, :] * dE_mev[:, None, None]
good = np.isfinite(denom) & (denom > 0) & np.isfinite(mu_ps)
dnde[good] = mu_ps[good] / denom[good]

# Totani y-axis cube: E^2 dN/dE
# Units: MeV cm^-2 s^-1 sr^-1
E2dnde = dnde * (Ectr_mev[:, None, None] ** 2)

# mu_ps already computed above in counts-space and PSF-smoothed.


# --------------------------------------------------
# WRITE OUTPUTS
# --------------------------------------------------
write_cube(OUT_MU, mu_ps, hdr, bunit="counts")
write_cube(OUT_DNDE, dnde, hdr, bunit="ph cm-2 s-1 sr-1 MeV-1")
write_cube(OUT_E2, E2dnde, hdr, bunit="MeV cm-2 s-1 sr-1")

print("✓ Wrote:", OUT_MU)
print("✓ Wrote:", OUT_DNDE)
print("✓ Wrote:", OUT_E2)
