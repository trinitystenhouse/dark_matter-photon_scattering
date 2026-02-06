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
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter


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

def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    """
    Ω_pix ≈ Δl Δb cos(b) for CAR.
    Matches what you used in other templates.
    """
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), dtype=float)

def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    """Interpolate exposure planes onto target energy centers in log(E)."""
    if expo_raw.shape[0] == len(E_tgt_mev):
        return expo_raw
    if E_expo_mev is None:
        raise RuntimeError("Exposure planes != counts planes and EXPO has no ENERGIES table.")

    order = np.argsort(E_expo_mev)
    E_expo_mev = E_expo_mev[order]
    expo_raw = expo_raw[order]

    logEs = np.log(E_expo_mev)
    logEt = np.log(E_tgt_mev)

    ne, ny, nx = expo_raw.shape
    flat = expo_raw.reshape(ne, ny * nx)

    idx = np.searchsorted(logEs, logEt)
    idx = np.clip(idx, 1, ne - 1)
    i0 = idx - 1
    i1 = idx
    w = (logEt - logEs[i0]) / (logEs[i1] - logEs[i0])

    out = np.empty((len(E_tgt_mev), ny * nx), float)
    for j in range(len(E_tgt_mev)):
        out[j] = (1 - w[j]) * flat[i0[j]] + w[j] * flat[i1[j]]
    return out.reshape(len(E_tgt_mev), ny, nx)


# --------------------------------------------------
# LOAD COUNTS HEADER / ENERGY BINS
# --------------------------------------------------
with fits.open(COUNTS) as h:
    hdr = h[0].header
    eb  = h["EBOUNDS"].data

wcs = WCS(hdr).celestial
ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])

Emin_kev = eb["E_MIN"].astype(float)
Emax_kev = eb["E_MAX"].astype(float)
Ectr_kev = np.sqrt(Emin_kev * Emax_kev)

# Correct MeV->GeV conversion (IMPORTANT)
Emin_mev = Emin_kev / 1000.0
Emax_mev = Emax_kev / 1000.0
Ectr_mev = Ectr_kev / 1000.0
dE_mev   = (Emax_mev - Emin_mev)

nE = len(Ectr_mev)
print("[E] Bin centers (MeV):", np.round(Ectr_mev, 3))

# Solid angle map
omega = pixel_solid_angle_map(wcs, ny, nx, BINSZ_DEG)


# --------------------------------------------------
# LOAD EXPOSURE (cm^2 s), resample if needed
# --------------------------------------------------
with fits.open(EXPO) as h:
    expo_raw = h[0].data.astype(float)
    E_expo_mev = None
    if "ENERGIES" in h:
        col0 = h["ENERGIES"].columns.names[0]
        E_expo_mev = np.array(h["ENERGIES"].data[col0], dtype=float)
        print(E_expo_mev)

expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
if expo.shape != (nE, ny, nx):
    raise RuntimeError(f"Exposure shape {expo.shape} not compatible with {(nE, ny, nx)}")


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
# PSF smoothing (energy-dependent), conserving integrated flux per bin
# --------------------------------------------------
for k in range(nE):
    sigma_pix = psf_sigma_deg(Ectr_mev[k]) / BINSZ_DEG
    if sigma_pix <= 0:
        continue
    before = np.nansum(phi_bin_map[k])
    phi_bin_map[k] = gaussian_filter(phi_bin_map[k], sigma=sigma_pix, mode="constant", cval=0.0)
    after = np.nansum(phi_bin_map[k])
    if after > 0 and np.isfinite(before):
        phi_bin_map[k] *= (before / after)


# --------------------------------------------------
# Convert to dN/dE (intensity):
#   dnde = Φ_bin / (Ω_pix * ΔE)
# Units: ph cm^-2 s^-1 sr^-1 MeV^-1
# --------------------------------------------------
dnde = (phi_bin_map / omega[None, :, :]) / dE_mev[:, None, None]

# Totani y-axis cube: E^2 dN/dE
# Units: MeV cm^-2 s^-1 sr^-1
E2dnde = dnde * (Ectr_mev[:, None, None] ** 2)

# Counts template:
#   mu = dnde * exposure * Ω_pix * ΔE  (=> counts)
mu_ps = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]


# --------------------------------------------------
# WRITE OUTPUTS (ASCII-safe FITS headers)
# --------------------------------------------------
def write_primary_with_bunit(path, data, hdr, bunit, comments):
    hdu = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
    hdu.header["BUNIT"] = bunit
    # Add COMMENT cards safely (ASCII only)
    for c in comments:
        c_ascii = c.encode("ascii", "replace").decode("ascii")
        hdu.header.add_comment(c_ascii)
    hdu.writeto(path, overwrite=True)

write_primary_with_bunit(
    OUT_MU, mu_ps, hdr, "counts",
    comments=[
        "Point-source template: expected counts per bin per pixel",
        "Built from Flux1000 (1-100 GeV) + power-law approx; PSF-smoothed; uses exposure",
    ],
)

write_primary_with_bunit(
    OUT_DNDE, dnde, hdr, "ph cm-2 s-1 sr-1 MeV-1",
    comments=[
        "Point-source template: dN/dE intensity normalized by bin width (per MeV)",
        "Built from Flux1000 (1-100 GeV) + power-law approx; PSF-smoothed",
        "Definition: dN/dE = Phi_bin / (Omega_pix * dE_mev)",
    ],
)

write_primary_with_bunit(
    OUT_E2, E2dnde, hdr, "MeV cm-2 s-1 sr-1",
    comments=[
        "Point-source template: E^2 dN/dE (MeV) derived from dN/dE per MeV",
        "Includes division by bin width dE (MeV)",
    ],
)

print("✓ Wrote:", OUT_MU)
print("✓ Wrote:", OUT_DNDE)
print("✓ Wrote:", OUT_E2)
