#!/usr/bin/env python3
"""
MODIFIED: adds a per-energy-bin fit and writes the TOTAL best-fit model counts cube.

What it does:
- For each energy bin k:
    Fit counts[k] with NNLS to a set of COUNTS templates (mu_*[k]) on mask_all[k]
    (ROI + disk cut + point/extended-source mask).
- Writes:
    OUTDIR/mu_modelsum_counts.fits   : (nE,ny,nx) total best-fit expected counts cube
    OUTDIR/fit_coeffs_per_bin.npz    : coefficients A[k, i] and labels
    OUTDIR/fit_coeffs_per_bin.txt    : human-readable table

Notes:
- Uses weighted NNLS (same weighting you already use).
- No extra normalisations of templates; assumes mu_* are already "true counts templates".
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u
from totani_helpers.totani_io import (
    load_mask_any_shape,
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    resample_exposure_logE_interp,
    read_expcube_energies_mev,
    smooth_nan_2d,
)
from totani_helpers.fit_utils import build_fit_mask3d
from totani_helpers.fit_utils import component_counts_from_cellwise_fit
from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter

try:
    from skimage.measure import find_contours
except Exception:
    find_contours = None

# -------------------------
# Inputs
# -------------------------
REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
COUNTS = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
EXPO = os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits")
PSMASK = os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits")
PSC = os.path.join(DATA_DIR, "templates", "gll_psc_v35.fit")

OUTDIR = os.path.join(os.path.dirname(__file__), "plots_fig1")
os.makedirs(OUTDIR, exist_ok=True)

DISK_CUT_DEG = 10.0
BINSZ_DEG = 0.125
TARGET_MEV = [1500, 4300]  # MeV (your variable name says GEV but these are MeV)

ROI_LON_DEG = 60.0
ROI_LAT_DEG = 60.0
CELL_DEG = 10.0

# Template inputs for Totani-style bubble construction (counts templates)
MU_GAS = os.path.join(DATA_DIR, "processed", "templates", "mu_gas_counts.fits")
MU_ICS = os.path.join(DATA_DIR, "processed", "templates", "mu_ics_counts.fits")
MU_ISO = os.path.join(DATA_DIR, "processed", "templates", "mu_iso_counts.fits")
MU_PS = os.path.join(DATA_DIR, "processed", "templates", "mu_ps_counts.fits")
MU_NFW = os.path.join(DATA_DIR, "processed", "templates", "mu_nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno_counts.fits")
MU_LOOPI = os.path.join(DATA_DIR, "processed", "templates", "mu_loopI_counts.fits")

# Prebuilt Totani-style bubble templates (dN/dE; positive/negative are nonnegative by construction)
BUBBLES_POS_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_pos_dnde.fits")
BUBBLES_NEG_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_neg_dnde.fits")
BUBBLES_FLAT_DNDE = os.path.join(DATA_DIR, "processed", "templates", "fb_flat_dnde.fits")
MU_FB_FLAT = os.path.join(DATA_DIR, "processed", "templates", "mu_fb_flat_counts.fits")

BUBBLE_MASK_FITS = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_binary_mask.fits")

PLOT_EXT_SOURCES = True
MARKER_RADIUS_DEG = 1.0
MARKER_LW = 1.0

# -------------------------


def read_extended_sources(psc_path):
    with fits.open(psc_path) as f:
        psc = f[1].data
        cols = {c.upper() for c in psc.columns.names}

        if "EXTENDED_SOURCE_NAME" in cols:
            ext_name = psc["EXTENDED_SOURCE_NAME"]
            ext_name = np.array([
                x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                for x in ext_name
            ])
            is_ext = np.array([len(s.strip()) > 0 and s.strip().upper() != "NONE" for s in ext_name])
        elif "EXTENDED" in cols:
            is_ext = np.array(psc["EXTENDED"]).astype(bool)
        else:
            raise RuntimeError("No EXTENDED_SOURCE_NAME or EXTENDED column found in PSC file")

        psc_ext = psc[is_ext]

    return SkyCoord(psc_ext["GLON"] * u.deg, psc_ext["GLAT"] * u.deg, frame="galactic")


# -------------------------
# Load data
# -------------------------
counts, hdr, Emin, Emax, Ectr, dE = read_counts_and_ebounds(COUNTS)
nE, ny, nx = counts.shape
wcs = WCS(hdr).celestial

with fits.open(COUNTS) as h:
    eb_hdu = h["EBOUNDS"].copy()

expo_raw, E_expo = read_exposure(EXPO)
if expo_raw.shape[1:] != (ny, nx):
    raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
expo = resample_exposure_logE(expo_raw, E_expo, Ectr)
if expo.shape[0] != nE:
    raise RuntimeError("Exposure resampling did not produce same nE as counts")

ps_mask = load_mask_any_shape(PSMASK, counts.shape)  # (nE,ny,nx) bool keep-mask

omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=BINSZ_DEG)

# Lon/lat for ROI etc.
lon, lat = lonlat_grids(wcs, ny, nx)

# Disk cut mask (2D)
disk_mask2d = np.abs(lat) >= DISK_CUT_DEG

roi2d = (np.abs(lon) <= ROI_LON_DEG) & (np.abs(lat) <= ROI_LAT_DEG)

# Full keep-mask used for fitting (nE,ny,nx): ROI & disk & srcmask & data coverage
mask_all = build_fit_mask3d(
    roi2d=roi2d,
    srcmask3d=ps_mask,
    counts=counts,
    expo=expo,
    extra2d=disk_mask2d,
)

# -------------------------
# Load counts templates
# -------------------------
def _read_cube(path, expected_shape):
    with fits.open(path) as h:
        d = h[0].data.astype(float)
    if d.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {d.shape}, expected {expected_shape}")
    return d

mu_gas = _read_cube(MU_GAS, counts.shape)
mu_ics = _read_cube(MU_ICS, counts.shape)
mu_iso = _read_cube(MU_ISO, counts.shape)
mu_ps = _read_cube(MU_PS, counts.shape)
mu_nfw = _read_cube(MU_NFW, counts.shape)
mu_loopi = _read_cube(MU_LOOPI, counts.shape)

if os.path.exists(MU_FB_FLAT):
    mu_flat = _read_cube(MU_FB_FLAT, counts.shape)
else:
    bubbles_flat_dnde = _read_cube(BUBBLES_FLAT_DNDE, counts.shape)
    mu_flat = (bubbles_flat_dnde > 0).astype(np.float64)

# -------------------------
# Fit and write model sum cube
# -------------------------
mu_cubes = [mu_gas, mu_ics, mu_iso, mu_ps, mu_nfw, mu_loopi, mu_flat]
labels = ["gas", "ics", "iso", "ps", "nfw", "loopI", "fb_flat"]

templates_counts = {lab: mu for lab, mu in zip(labels, mu_cubes)}

res_fit = fit_cellwise_poisson_mle_counts(
    counts=counts,
    templates=templates_counts,
    mask3d=mask_all,
    lon=lon,
    lat=lat,
    roi_lon=float(ROI_LON_DEG),
    roi_lat=float(ROI_LAT_DEG),
    cell_deg=float(CELL_DEG),
    nonneg=True,
    column_scale="l2",
    drop_tol=0.0,
    ridge=0.0,
)
comp_counts, model_counts = component_counts_from_cellwise_fit(
    templates_counts=templates_counts,
    res_fit=res_fit,
    mask3d=mask_all,
)

fb_mask2d = None
if os.path.exists(BUBBLE_MASK_FITS):
    try:
        fb_mask2d = fits.getdata(BUBBLE_MASK_FITS).astype(bool)
        if fb_mask2d.ndim == 3:
            fb_mask2d = fb_mask2d[0].astype(bool)
        if fb_mask2d.shape != (ny, nx):
            fb_mask2d = None
    except Exception:
        fb_mask2d = None


ext_coords = None
if PLOT_EXT_SOURCES and os.path.exists(PSC):
    try:
        ext_coords = read_extended_sources(PSC)
    except Exception:
        ext_coords = None

for target_mev in TARGET_MEV:
    k = int(np.argmin(np.abs(Ectr - target_mev)))

    m = mask_all[k]
    bubble_image_counts_k = np.array(comp_counts["fb_flat"][k], dtype=float) + (np.array(counts[k], dtype=float) - np.array(model_counts[k], dtype=float))

    denom = np.array(expo[k], dtype=float) * np.array(omega, dtype=float) * float(dE[k])
    bubble_flux_k = np.full((ny, nx), np.nan, dtype=float)
    ok = m & np.isfinite(denom) & (denom > 0)
    bubble_flux_k[ok] = bubble_image_counts_k[ok] / denom[ok]

    sig_pix = float(1.0 / BINSZ_DEG)
    bubble_flux_smooth_k = smooth_nan_2d(bubble_flux_k, sig_pix)
    bubble_flux_smooth_k[~m] = np.nan

    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.set_xlabel("l")
    ax.set_ylabel("b")

    if target_mev == TARGET_MEV[0]:
        vmin, vmax = -5e-10, 5e-10
    elif target_mev == TARGET_MEV[1]:
        vmin, vmax = -5e-11, 5e-11
    else:
        vmax = np.nanpercentile(np.abs(bubble_flux_smooth_k), 99.5)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = -vmax

    im = ax.imshow(bubble_flux_smooth_k, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(f"Best-fit bubble image (flux), smoothed $\\sigma$=1$^\\circ$\nE={Ectr[k]:.3g} GeV (k={k})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if fb_mask2d is not None:
        try:
            ax.contour(
                fb_mask2d.astype(float),
                levels=[0.5],
                colors="k",
                linewidths=1.0,
                alpha=0.8,
            )
        except Exception:
            ax.imshow(
                np.where(fb_mask2d, 1.0, np.nan),
                origin="lower",
                cmap="gray",
                alpha=0.12,
            )


    out_png = os.path.join(OUTDIR, f"fig1_bubble_image_flux_smooth1deg_E{target_mev:.0f}MeV_k{k}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print("✓ wrote", out_png)

print("✓ Done:", OUTDIR)
