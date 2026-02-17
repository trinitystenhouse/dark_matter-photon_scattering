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
from totani_helpers.totani_io import *
from scipy.optimize import nnls

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

BUBBLE_MASK_FITS = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_binary_mask.fits")

PLOT_EXT_SOURCES = True
MARKER_RADIUS_DEG = 1.0
MARKER_LW = 1.0

# -------------------------


def fit_bin_weighted_nnls(counts_2d, mu_components, mask_2d, eps=1.0):
    """
    Weighted NNLS fit:
      minimize || W (X a - y) ||_2, with a>=0
    Weight ~ 1/sqrt(y+eps) (Poisson-ish).
    """
    m = mask_2d.ravel()
    y = counts_2d.ravel()[m]
    X = np.vstack([mu.ravel()[m] for mu in mu_components]).T

    w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
    yw = y * w
    Xw = X * w[:, None]

    A, _ = nnls(Xw, yw)
    return A


def build_total_model_cube(counts_cube, mu_cubes, mask_all, labels, eps=1.0):
    """
    Fit each energy bin with NNLS and build model sum cube.
    Returns:
      mu_modelsum (nE,ny,nx)
      A (nE,ncomp)
    """
    nE, ny, nx = counts_cube.shape
    ncomp = len(mu_cubes)

    A = np.zeros((nE, ncomp), dtype=np.float64)
    mu_modelsum = np.zeros((nE, ny, nx), dtype=np.float64)

    for k in range(nE):
        mk = mask_all[k]
        if not np.any(mk):
            # nothing to fit
            continue

        mu_components_k = [mu[k] for mu in mu_cubes]
        Ak = fit_bin_weighted_nnls(counts_cube[k], mu_components_k, mk, eps=eps)
        A[k] = Ak

        # Build model sum everywhere (but keep masked pixels at 0 for clarity)
        model_k = np.zeros((ny, nx), dtype=np.float64)
        for i, mu in enumerate(mu_components_k):
            model_k += Ak[i] * mu
        model_k[~mk] = 0.0
        mu_modelsum[k] = model_k

        if (k % 5) == 0 or (k == nE - 1):
            msg = " ".join([f"{labels[i]}={Ak[i]:.3g}" for i in range(ncomp)])
            print(f"[fit] k={k:02d}  " + msg)

    return mu_modelsum, A


def load_mask_2d(mask_path, shape_2d):
    m = fits.getdata(mask_path)
    if m.ndim == 3:
        m = m[0]
    m = np.asarray(m)
    if m.shape != shape_2d:
        raise RuntimeError(f"Mask shape {m.shape} not compatible with expected {shape_2d}")
    return (np.isfinite(m) & (m != 0))


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
with fits.open(COUNTS) as h:
    counts = h[0].data.astype(float)
    hdr = h[0].header
    eb = h["EBOUNDS"].data
    eb_hdu = h["EBOUNDS"].copy()

with fits.open(EXPO) as h:
    expo_raw = np.array(h[0].data, dtype=np.float64)
    E_expo = read_expcube_energies_mev(h)

nE, ny, nx = counts.shape
ps_mask = load_mask_any_shape(PSMASK, counts.shape)  # assumes your helper returns (nE,ny,nx) bool keep-mask

Ectr = np.sqrt(eb["E_MIN"] * eb["E_MAX"]) / 1000.0  # MeV
dE = (eb["E_MAX"] - eb["E_MIN"]) / 1000.0  # MeV

expo = resample_exposure_logE_interp(expo_raw, E_expo, Ectr)

wcs = WCS(hdr).celestial
omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=BINSZ_DEG)

# Disk cut mask (2D)
yy, xx = np.mgrid[:ny, :nx]
_, lat = wcs.pixel_to_world_values(xx, yy)
disk_mask2d = np.abs(lat) >= DISK_CUT_DEG

# Lon/lat for ROI etc.
lon, lat2 = wcs.pixel_to_world_values(xx, yy)
lon_w = ((lon + 180.0) % 360.0) - 180.0
roi2d = (np.abs(lon_w) <= ROI_LON_DEG) & (np.abs(lat2) <= ROI_LAT_DEG)

# Full keep-mask used for fitting (nE,ny,nx)
mask_all = ps_mask & disk_mask2d[None, :, :] & roi2d[None, :, :]

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

# Optional: include flat FB in the baseline fit (recommended for Totani step-1)
# We have bubbles_flat_dnde; convert it to mu_flat_counts per bin.
bubbles_flat_dnde = _read_cube(BUBBLES_FLAT_DNDE, counts.shape)
mu_flat = (bubbles_flat_dnde > 0).astype(np.float64)

# -------------------------
# Fit and write model sum cube
# -------------------------
mu_cubes = [mu_gas, mu_ics, mu_iso, mu_ps, mu_nfw, mu_loopi]
labels = ["gas", "ics", "iso", "ps", "nfw", "loopI"]


# include flat FB (Totani initial stage)
mu_cubes.append(mu_flat)
labels.append("fb_flat")

mu_modelsum, A = build_total_model_cube(counts, mu_cubes, mask_all, labels, eps=1.0)

# Write total model counts cube
out_model = os.path.join(OUTDIR, "mu_modelsum_counts.fits")
phdu = fits.PrimaryHDU(mu_modelsum.astype(np.float32), header=hdr)
phdu.header["BUNIT"] = "counts"
phdu.header["COMMENT"] = "Total best-fit model expected counts cube from per-bin NNLS fit."
phdu.header["COMMENT"] = "Components: " + ", ".join(labels)
phdu.header["COMMENT"] = "Fit mask: ROI box AND disk cut AND extended-source mask."
phdu.header["COMMENT"] = f"ROI: |l|<={ROI_LON_DEG:g}, |b|<={ROI_LAT_DEG:g}; disk cut |b|>={DISK_CUT_DEG:g}."
phdu.header["COMMENT"] = "NNLS is weighted by 1/sqrt(counts+1)."
fits.HDUList([phdu, eb_hdu]).writeto(out_model, overwrite=True)
print("✓ wrote", out_model)

# Save coefficients
out_npz = os.path.join(OUTDIR, "fit_coeffs_per_bin.npz")
np.savez(out_npz, A=A, labels=np.array(labels, dtype="U"), Ectr_mev=Ectr)
print("✓ wrote", out_npz)

out_txt = os.path.join(OUTDIR, "fit_coeffs_per_bin.txt")
with open(out_txt, "w") as f:
    f.write("# k  Ectr(MeV)  " + "  ".join([f"{lab:>10s}" for lab in labels]) + "\n")
    for k in range(nE):
        vals = "  ".join([f"{A[k,i]:10.4e}" for i in range(len(labels))])
        f.write(f"{k:2d}  {Ectr[k]:10.3f}  {vals}\n")
print("✓ wrote", out_txt)

ext_coords = None
if PLOT_EXT_SOURCES and os.path.exists(PSC):
    try:
        ext_coords = read_extended_sources(PSC)
    except Exception:
        ext_coords = None

for target_mev in TARGET_MEV:
    k = int(np.argmin(np.abs(Ectr - target_mev)))

    data_k = np.array(counts[k], dtype=float)
    model_k = np.array(mu_modelsum[k], dtype=float)
    resid_k = data_k - model_k
    m = mask_all[k]

    data_k[~m] = np.nan
    model_k[~m] = np.nan
    resid_k[~m] = np.nan

    fig = plt.figure(figsize=(12, 4.2))
    for i, (arr, title) in enumerate([
        (data_k, "Data"),
        (model_k, "Model"),
        (resid_k, "Residual (Data-Model)"),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection=wcs)
        ax.set_xlabel("l")
        ax.set_ylabel("b")

        if i < 2:
            v = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
            norm = simple_norm(v, stretch="log", min_cut=np.nanpercentile(v, 1.0), max_cut=np.nanpercentile(v, 99.5))
            im = ax.imshow(v, origin="lower", cmap="magma", norm=norm)
        else:
            vmax = np.nanpercentile(np.abs(arr), 99.5)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)

        ax.set_title(f"{title}\nE={Ectr[k]:.3g} GeV (k={k})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if ext_coords is not None:
            for c in ext_coords:
                cc = c.galactic
                ax.add_patch(
                    Circle(
                        (cc.l.deg, cc.b.deg),
                        radius=MARKER_RADIUS_DEG,
                        transform=ax.get_transform("world"),
                        fill=False,
                        lw=MARKER_LW,
                        edgecolor="0.7",
                        alpha=0.7,
                    )
                )

    out_png = os.path.join(OUTDIR, f"fig1_maps_E{target_mev:.2f}GeV_k{k}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print("✓ wrote", out_png)

print("✓ Done:", OUTDIR)
