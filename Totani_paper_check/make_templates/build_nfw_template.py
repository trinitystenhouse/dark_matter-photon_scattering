#!/usr/bin/env python3
"""
Build an NFW (rho^2.5) template on the counts CCUBE grid.

Outputs (same conventions as your PS template):
  - nfw_dnde.fits     : dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - nfw_E2dnde.fits   : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - mu_nfw_counts.fits: expected counts per bin per pixel [counts]

Notes:
- The spatial template is normalised so sum over ROI pixels = 1.
- The spectrum is unit-normalised over bins (sum_k Phi_bin = 1 ph/cm^2/s).
  Your fit coefficient then becomes the physical total flux amplitude.
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from make_nfw_rho25 import make_nfw_rho25_template

# -------------------------
# Paths
# -------------------------
REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import pixel_solid_angle_map, resample_exposure_logE
COUNTS = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
EXPO = os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits")
OUTDIR = os.path.join(DATA_DIR, "processed", "templates")

os.makedirs(OUTDIR, exist_ok=True)
OUT_DNDE   = os.path.join(OUTDIR, "nfw_dnde.fits")
OUT_E2DNDE = os.path.join(OUTDIR, "nfw_E2dnde.fits")
OUT_MU     = os.path.join(OUTDIR, "mu_nfw_counts.fits")

# -------------------------
# Settings
# -------------------------
BINSZ_DEG = 0.125
ROI_LON_DEG = 60.0
ROI_LAT_DEG = 60.0

# spectral normalisation: flat Phi_bin then renormalise (unit total flux)
SPECTRUM = "flat"  # placeholder; keep flat for now


# -------------------------
# Read counts binning (authoritative)
# -------------------------
with fits.open(COUNTS) as h:
    hdr = h[0].header
    eb  = h["EBOUNDS"].data
    # Debug: print energy binning as stored in the counts cube

Emin_kev = eb["E_MIN"].astype(float)
Emax_kev = eb["E_MAX"].astype(float)
Ectr_kev = np.sqrt(Emin_kev * Emax_kev)

Emin_mev = Emin_kev / 1000.0
Emax_mev = Emax_kev / 1000.0
dE_mev   = (Emax_mev - Emin_mev)
Ectr_mev = np.sqrt(Emin_mev * Emax_mev)

print("[E] counts EBOUNDS:")
print("[E]   Emin_keV:", np.round(Emin_kev, 3))
print("[E]   Emax_keV:", np.round(Emax_kev, 3))
print("[E]   Ectr_keV:", np.round(Ectr_kev, 3))
print("[E]   Emin_MeV:", np.round(Emin_mev, 6))
print("[E]   Emax_MeV:", np.round(Emax_mev, 6))
print("[E]   Ectr_MeV:", np.round(Ectr_mev, 6))
print("[E]   Ectr_GeV:", np.round(Ectr_mev / 1000.0, 6))
print("[E]   dE_MeV  :", np.round(dE_mev, 6))

nE = len(Ectr_mev)

# WCS + spatial shape comes from counts header
wcs = WCS(hdr).celestial
ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]

# -------------------------
# Read exposure, resample if needed
# -------------------------
with fits.open(EXPO) as h:
    expo_raw = h[0].data.astype(float)
    E_expo_mev = None
    if "ENERGIES" in h:
        col0 = h["ENERGIES"].columns.names[0]
        E_expo_mev = np.array(h["ENERGIES"].data[col0], dtype=float)  # MeV
        print("[E] expcube ENERGIES (as stored, assumed MeV):")
        print("[E]   N:", int(E_expo_mev.size))
        print("[E]   min/max:", float(np.nanmin(E_expo_mev)), float(np.nanmax(E_expo_mev)))
        print("[E]   first/last:", float(E_expo_mev[0]), float(E_expo_mev[-1]))
    else:
        print("[E] expcube has no ENERGIES extension")

print("[E] expcube planes (raw):", expo_raw.shape)

if expo_raw.shape[1:] != (ny, nx):
    raise RuntimeError(f"Exposure spatial shape {expo_raw.shape[1:]} != counts {(ny, nx)}")

expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
if expo.shape[0] != nE:
    raise RuntimeError(f"Exposure after resampling has {expo.shape[0]} planes, expected {nE}")

print("[E] expcube planes (resampled):", expo.shape)

# -------------------------
# Build lon/lat grid
# -------------------------
yy, xx = np.mgrid[:ny, :nx]
lon, lat = wcs.pixel_to_world_values(xx, yy)

# Wrap lon to [-180,180) for consistent ROI definition
lon = ((lon + 180.0) % 360.0) - 180.0

# -------------------------
# NFW spatial template + ROI + normalisation
# -------------------------
nfw_spatial = make_nfw_rho25_template(lon, lat).astype(float)

roi = (np.abs(lon) <= ROI_LON_DEG) & (np.abs(lat) <= ROI_LAT_DEG)
nfw_spatial[~roi] = 0.0

norm = np.nansum(nfw_spatial)
if not np.isfinite(norm) or norm <= 0:
    raise RuntimeError("NFW template is zero everywhere in ROI")
nfw_spatial /= norm  # sum over ROI pixels = 1

# -------------------------
# Pixel solid angle map (same as PS template)
# -------------------------
omega = pixel_solid_angle_map(wcs, ny, nx, BINSZ_DEG)  # sr

# -------------------------
# Spectrum (unit total flux across bins)
# Phi_bin has units ph/cm^2/s integrated over each bin
# -------------------------
if SPECTRUM == "flat":
    Phi_bin = np.ones(nE, float)
else:
    raise RuntimeError(f"Unknown SPECTRUM='{SPECTRUM}'")

Phi_bin /= Phi_bin.sum()  # sum_k Phi_bin = 1 ph/cm^2/s

# -------------------------
# Build dN/dE cube (per MeV)
# dN/dE = (Phi_bin * spatial) / (Omega_pix * dE)
# Units: ph cm^-2 s^-1 sr^-1 MeV^-1
# -------------------------
nfw_dnde = np.empty((nE, ny, nx), float)
for k in range(nE):
    nfw_dnde[k] = (Phi_bin[k] * nfw_spatial) / (omega * dE_mev[k])

# E^2 dN/dE
nfw_E2dnde = nfw_dnde * (Ectr_mev[:, None, None] ** 2)  # MeV cm^-2 s^-1 sr^-1

# Expected counts per bin per pixel:
# mu = (dN/dE) * exposure * Omega_pix * dE
mu_nfw = nfw_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

# -------------------------
# Write outputs
# -------------------------
hdu = fits.PrimaryHDU(nfw_dnde.astype(np.float32), header=hdr)
hdu.header["BUNIT"] = "ph cm-2 s-1 sr-1 MeV-1"
hdu.header["COMMENT"] = "gNFW (gamma=1.25) rho^2 spatial template; sum ROI pixels = 1"
hdu.header["COMMENT"] = "Spectrum: Phi_bin flat, renormalised so sum_k Phi_bin = 1 ph/cm^2/s"
hdu.writeto(OUT_DNDE, overwrite=True)

hdu = fits.PrimaryHDU(nfw_E2dnde.astype(np.float32), header=hdr)
hdu.header["BUNIT"] = "MeV cm-2 s-1 sr-1"
hdu.header["COMMENT"] = "E^2 dN/dE version of NFW template"
hdu.writeto(OUT_E2DNDE, overwrite=True)

hdu = fits.PrimaryHDU(mu_nfw.astype(np.float32), header=hdr)
hdu.header["BUNIT"] = "counts"
hdu.header["COMMENT"] = "Expected counts = dN/dE * exposure * Omega_pix * dE"
hdu.writeto(OUT_MU, overwrite=True)

print("✓ wrote", OUT_DNDE)
print("✓ wrote", OUT_E2DNDE)
print("✓ wrote", OUT_MU)
print("sum Phi_bin:", Phi_bin.sum())
print("mu_nfw total counts:", float(np.nansum(mu_nfw)))
print("mu_nfw per-bin:", np.nansum(mu_nfw, axis=(1, 2)))
