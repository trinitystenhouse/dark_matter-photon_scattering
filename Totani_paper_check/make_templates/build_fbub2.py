#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from totani_helpers.exposure import resample_exposure
from scipy.optimize import nnls
from scipy.ndimage import label

try:
    from skimage.measure import find_contours
except Exception:
    find_contours = None

def set_lon_ticks_wrapped(ax, wcs, ny, nx, lons_deg=(-60, -30, 0, 30, 60)):
    xs = []
    for L in lons_deg:
        L360 = L % 360
        x, _ = wcs.world_to_pixel_values(L360, 0.0)  # place ticks at b=0
        xs.append(x)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{L:d}" for L in lons_deg])
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
PIX_SR = np.deg2rad(BINSZ_DEG) ** 2
TARGET_GEV = [1500, 4300]  # GeV
LAT_FIT_MAX_DEG = 50.0
THR_SIG = .2

ROI_LON_DEG = 60.0
ROI_LAT_DEG = 60.0
CELL_DEG = 10.0

# Template inputs for Totani-style bubble construction (counts templates)
MU_GAS = os.path.join(DATA_DIR, "processed", "templates", "mu_gas_counts.fits")
MU_ICS = os.path.join(DATA_DIR, "processed", "templates", "mu_ics_counts.fits")
MU_ISO = os.path.join(DATA_DIR, "processed", "templates", "mu_iso_counts.fits")
MU_PS = os.path.join(DATA_DIR, "processed", "templates", "mu_ps_counts.fits")
MU_NFW = os.path.join(DATA_DIR, "processed", "templates", "mu_nfw_counts.fits")

# Prebuilt Totani-style bubble templates (made by make_templates/build_fermi_bubbles_templates.py)
# BUBBLES_POS_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_pos_dnde.fits")
# BUBBLES_NEG_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_neg_dnde.fits")
# BUBBLES_FLAT_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_dnde.fits")

# Optional: provide a FITS mask (ny,nx) on the counts grid representing Totani's
# final flat-bubble boundary (1 inside, 0 outside). If None, uses heuristic box.
BUBBLE_MASK_FITS = None

# Overlay settings
PLOT_EXT_SOURCES = True
MARKER_RADIUS_DEG = 1.0
MARKER_LW = 1.0

# -------------------------
def smooth_nan_2d(x2d, sigma_pix):
    m = np.isfinite(x2d)
    if not np.any(m):
        return np.full_like(x2d, np.nan)
    w = m.astype(float)
    xs = gaussian_filter(np.where(m, x2d, 0.0), sigma_pix)
    ws = gaussian_filter(w, sigma_pix)
    y = np.full_like(x2d, np.nan, dtype=float)
    ok = ws > 0
    y[ok] = xs[ok] / ws[ok]
    return y


def _heuristic_flat_bubble_mask(lon_deg, lat_deg, lon_max=20.0, lat_min=10.0, lat_max=55.0):
    return (
        (np.abs(lon_deg) <= lon_max)
        & (np.abs(lat_deg) >= lat_min)
        & (np.abs(lat_deg) <= lat_max)
    )


def fit_bin_weighted_nnls(counts_2d, mu_components, mask_2d, eps=1.0):
    m = mask_2d.ravel()
    y = counts_2d.ravel()[m]
    X = np.vstack([mu.ravel()[m] for mu in mu_components]).T

    w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
    yw = y * w
    Xw = X * w[:, None]

    A, _ = nnls(Xw, yw)
    return A

def add_totani_overlays(ax, *, disk_band_2d, bubble_boundary_2d, ext_x=None, ext_y=None, marker_r_pix=8.0):
    # Grey band for the excluded disk region
    disk_overlay = np.where(disk_band_2d, 1.0, np.nan)
    ax.imshow(
        disk_overlay,
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        alpha=0.85,
        interpolation="nearest",
    )

    # Bubble boundary outline (flat mask)
    ax.contour(
        bubble_boundary_2d.astype(float),
        levels=[0.5],
        colors="w",
        linewidths=1.0,
        alpha=0.9,
    )

    # Extended source circles
    if ext_x is not None and ext_y is not None:
        for x, y in zip(ext_x, ext_y):
            circ = Circle(
                (x, y),
                radius=marker_r_pix,
                edgecolor="0.7",
                facecolor="none",
                linewidth=1.0,
                alpha=0.9,
            )
            ax.add_patch(circ)


def load_mask_any_shape(mask_path, counts_shape):
    m = fits.getdata(mask_path).astype(bool)
    nE, ny, nx = counts_shape
    if m.shape == (nE, ny, nx):
        return m
    if m.shape == (ny, nx):
        return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
    raise RuntimeError(f"Mask shape {m.shape} not compatible with counts shape {(nE, ny, nx)}")


def _rdp_simplify(points_xy, epsilon):
    if points_xy.shape[0] < 3:
        return points_xy

    p0 = points_xy[0]
    p1 = points_xy[-1]
    v = p1 - p0
    vv = float(np.dot(v, v))
    if vv == 0.0:
        d = np.sqrt(np.sum((points_xy - p0) ** 2, axis=1))
    else:
        t = np.dot(points_xy - p0, v) / vv
        t = np.clip(t, 0.0, 1.0)
        proj = p0[None, :] + t[:, None] * v[None, :]
        d = np.sqrt(np.sum((points_xy - proj) ** 2, axis=1))

    i = int(np.argmax(d))
    dmax = float(d[i])
    if dmax <= epsilon:
        return np.vstack([p0, p1])

    left = _rdp_simplify(points_xy[: i + 1], epsilon)
    right = _rdp_simplify(points_xy[i:], epsilon)
    return np.vstack([left[:-1], right])


def plot_boundary_polygon(ax, boundary_2d, *, color="w", lw=1.5, alpha=0.9, simplify_eps_pix=2.0):
    if find_contours is None:
        ax.contour(boundary_2d.astype(float), levels=[0.5], colors=color, linewidths=lw, alpha=alpha)
        return

    contours = find_contours(boundary_2d.astype(float), level=0.5)
    if not contours:
        return

    contours = sorted(contours, key=lambda a: a.shape[0], reverse=True)
    for c in contours:
        if c.shape[0] < 10:
            continue
        xy = np.column_stack([c[:, 1], c[:, 0]])
        xy_s = _rdp_simplify(xy, epsilon=simplify_eps_pix)
        ax.plot(xy_s[:, 0], xy_s[:, 1], color=color, linewidth=lw, alpha=alpha)


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

with fits.open(EXPO) as h:
    expo_raw = h[0].data.astype(float)
    if "ENERGIES" in h:
        col0 = h["ENERGIES"].columns.names[0]
        E_expo = np.array(h["ENERGIES"].data[col0], dtype=float)  # MeV
        print(E_expo, "E_expo")
    else:
        raise RuntimeError("Exposure cube has no ENERGIES extension")

nE, ny, nx = counts.shape
ps_mask = load_mask_any_shape(PSMASK, counts.shape)

Ectr = np.sqrt(eb["E_MIN"] * eb["E_MAX"]) / 1000.0  # MeV
dE   = (eb["E_MAX"] - eb["E_MIN"]) / 1000.0         # MeV

expo = resample_exposure(expo_raw, E_expo, Ectr)

wcs = WCS(hdr).celestial

# Disk cut mask
yy, xx = np.mgrid[:ny, :nx]
_, lat = wcs.pixel_to_world_values(xx, yy)
disk_mask = np.abs(lat) >= DISK_CUT_DEG

# Lon/lat for bubble mask
lon, lat2 = wcs.pixel_to_world_values(xx, yy)
lon_w = ((lon + 180.0) % 360.0) - 180.0

roi2d = (np.abs(lon_w) <= ROI_LON_DEG) & (np.abs(lat2) <= ROI_LAT_DEG)

# Flux cube
denom = expo * PIX_SR * dE[:, None, None]
flux = np.full_like(counts, np.nan, dtype=float)
ok = denom > 0
flux[ok] = counts[ok] / denom[ok]

# Load counts templates for Totani-style bubble construction
def _read_mu(path, expected_shape):
    with fits.open(path) as h:
        d = h[0].data.astype(float)
    if d.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {d.shape}, expected {expected_shape}")
    return d

mu_gas = _read_mu(MU_GAS, counts.shape)
mu_ics = _read_mu(MU_ICS, counts.shape)
mu_iso = _read_mu(MU_ISO, counts.shape)
mu_ps = _read_mu(MU_PS, counts.shape)
mu_nfw = _read_mu(MU_NFW, counts.shape)


# Extended sources (overlay)
ext_coords = read_extended_sources(PSC) if PLOT_EXT_SOURCES else None
if ext_coords is not None:
    ext_x, ext_y = wcs.world_to_pixel(ext_coords)
    ext_x, ext_y = np.asarray(ext_x), np.asarray(ext_y)
    onmap = (ext_x >= 0) & (ext_x < nx) & (ext_y >= 0) & (ext_y < ny)
    ext_x, ext_y = ext_x[onmap], ext_y[onmap]
    print(f"[EXT] Overlaying {len(ext_x)} extended sources on map")

marker_r_pix = MARKER_RADIUS_DEG / BINSZ_DEG


# -------------------------
# Plot
# -------------------------
sigma_pix = 1 / BINSZ_DEG

for itgt, tgt in enumerate(TARGET_GEV):
    k = int(np.argmin(np.abs(Ectr - tgt)))
    print("Energy bin:", k, "Ectr=", Ectr[k])

    tgt_gev = tgt / 1000.0

    lat_cut = (np.abs(lat2) <= LAT_FIT_MAX_DEG)
    mask2d = roi2d & disk_mask & lat_cut & ps_mask[k]
    # mask_all: (nE, ny, nx) True=keep, False=mask
    mask_all = ps_mask & disk_mask[None, :, :] & roi2d[None, :, :]

    # set masked pixels to NaN across the entire cube
    flux_masked = flux.copy()
    flux_masked[~mask_all] = np.nan

    f = flux_masked[k].copy()
    f[~mask2d] = np.nan

    with np.errstate(invalid="ignore"):
        T_b = np.nanmean(f, axis=1)
    # If an entire latitude row is masked, nanmean returns NaN; keep it NaN to avoid warnings
    res = f - T_b[:, None]
    res_s = smooth_nan_2d(res, sigma_pix=sigma_pix)
    res_s[~mask2d] = np.nan

    denom_k = denom[k]
    resid_counts_s = np.full((ny, nx), np.nan, dtype=float)
    ok_sig = mask2d & np.isfinite(res_s) & np.isfinite(denom_k) & (denom_k > 0)
    resid_counts_s[ok_sig] = res_s[ok_sig] * denom_k[ok_sig]

    sig = np.full((ny, nx), np.nan, dtype=float)
    ok_sig2 = ok_sig & np.isfinite(counts[k])
    sig[ok_sig2] = resid_counts_s[ok_sig2] / np.sqrt(np.maximum(counts[k][ok_sig2], 0.0) + 1.0)
    sig_s = smooth_nan_2d(sig, sigma_pix=sigma_pix)
    sig_s[~mask2d] = np.nan

    above = mask2d & np.isfinite(sig_s) & (sig_s >= THR_SIG)

    above_n = above & (lat2 > 0.0)
    lbln, nlabn = label(above_n)
    if nlabn > 0:
        sizes = np.bincount(lbln.ravel())
        sizes[0] = 0
        keep = int(np.argmax(sizes))
        cc_mask_n = (lbln == keep)
    else:
        cc_mask_n = np.zeros((ny, nx), dtype=bool)

    above_s = above & (lat2 < 0.0)
    lbls, nlabs = label(above_s)
    if nlabs > 0:
        sizes = np.bincount(lbls.ravel())
        sizes[0] = 0
        keep = int(np.argmax(sizes))
        cc_mask_s = (lbls == keep)
    else:
        cc_mask_s = np.zeros((ny, nx), dtype=bool)

    cc_mask = cc_mask_n | cc_mask_s

    h2 = wcs.to_header()
    out_cc = f"{OUTDIR}/ccmask_residual_{tgt:.1f}MeV_extsrc.fits"
    out_cc_n = f"{OUTDIR}/ccmask_residual_{tgt:.1f}MeV_extsrc_north.fits"
    out_cc_s = f"{OUTDIR}/ccmask_residual_{tgt:.1f}MeV_extsrc_south.fits"
    fits.writeto(out_cc, cc_mask.astype(np.int16), header=h2, overwrite=True)
    fits.writeto(out_cc_n, cc_mask_n.astype(np.int16), header=h2, overwrite=True)
    fits.writeto(out_cc_s, cc_mask_s.astype(np.int16), header=h2, overwrite=True)
    print("✓ wrote", out_cc)
    print("✓ wrote", out_cc_n)
    print("✓ wrote", out_cc_s)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection=wcs)

    if itgt == 0:
        vmin, vmax = -5e-10, 5e-10
    else:
        vmin, vmax = -5e-11, 5e-11
    im = ax.imshow(res_s, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
    set_lon_ticks_wrapped(ax, wcs, ny, nx)

    # ✅ Outline the pixels you actually removed (extended sources + disk cut)
    ax.contour(~mask2d, levels=[0.5], linewidths=0.8, colors="k", alpha=0.8)

    # Largest connected components (north/south) above significance threshold
    ax.contour(cc_mask_n.astype(float), levels=[0.5], linewidths=1.2, colors="c", alpha=0.95)
    ax.contour(cc_mask_s.astype(float), levels=[0.5], linewidths=1.2, colors="m", alpha=0.95)

    ax.set_title(f"Residual flux (E~{Ectr[k]:.1f} MeV)\nExtended-source + disk masked")
    ax.set_xlabel("Galactic longitude")
    ax.set_ylabel("Galactic latitude")
    plt.colorbar(im, ax=ax, label=r"residual flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
    plt.tight_layout()

    out = f"{OUTDIR}/residual_{tgt:.1f}MeV_extsrc.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("✓ wrote", out)

    res_cc = np.where(cc_mask, res_s, np.nan)
    res_cc_n = np.where(cc_mask_n, res_s, np.nan)
    res_cc_s = np.where(cc_mask_s, res_s, np.nan)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221, projection=wcs)
    ax2 = fig.add_subplot(222, projection=wcs)
    ax3 = fig.add_subplot(223, projection=wcs)
    ax4 = fig.add_subplot(224, projection=wcs)

    im1 = ax1.imshow(res_s, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
    ax1.contour(cc_mask_n.astype(float), levels=[0.5], colors="c", linewidths=1.2, alpha=0.95)
    ax1.contour(cc_mask_s.astype(float), levels=[0.5], colors="m", linewidths=1.2, alpha=0.95)
    im2 = ax2.imshow(cc_mask.astype(float), origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
    im3 = ax3.imshow(res_cc, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
    im4 = ax4.imshow(res_cc_n, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
    ax4.imshow(res_cc_s, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu", alpha=0.85)

    for axx in (ax1, ax2, ax3, ax4):
        set_lon_ticks_wrapped(axx, wcs, ny, nx)
        axx.contour(~mask2d, levels=[0.5], linewidths=0.6, colors="k", alpha=0.7)
        axx.set_xlabel("l")
        axx.set_ylabel("b")

    ax1.set_title("Residual + CC outline")
    ax2.set_title("Largest connected component")
    ax3.set_title("Residual masked to CC")
    ax4.set_title("CC residual: North (opaque) + South (transparent overlay)")

    plt.colorbar(im1, ax=ax1, fraction=0.046)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    plt.tight_layout()

    out_m = f"{OUTDIR}/morphology_residual_{tgt:.1f}MeV_extsrc.png"
    plt.savefig(out_m, dpi=200)
    plt.close()
    print("✓ wrote", out_m)


print("✓ Done:", OUTDIR)
