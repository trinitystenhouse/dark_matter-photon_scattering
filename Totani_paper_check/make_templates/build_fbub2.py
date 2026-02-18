#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from totani_helpers.totani_io import resample_exposure
from scipy.optimize import nnls
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes

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


def add_fit_box_overlay(ax, *, wcs, lon_max_deg, lat_max_deg, color="y", lw=1.2, alpha=0.9):
    x1, y1 = wcs.world_to_pixel_values(((-lon_max_deg) % 360.0), -lat_max_deg)
    x2, y2 = wcs.world_to_pixel_values((( lon_max_deg) % 360.0),  lat_max_deg)
    x0, x1p = sorted([float(x1), float(x2)])
    y0, y1p = sorted([float(y1), float(y2)])
    rect = Rectangle(
        (x0, y0),
        (x1p - x0),
        (y1p - y0),
        fill=False,
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(rect)
def safe_percentile(a, q, default=np.nan):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return default
    return float(np.nanpercentile(a, q))


def fit_octagon_from_mask(mask_2d, *, n_vertices=8):
    pts_yx = np.argwhere(mask_2d)
    if pts_yx.shape[0] < 10:
        return None

    xy = np.column_stack([pts_yx[:, 1].astype(float), pts_yx[:, 0].astype(float)])
    try:
        hull = ConvexHull(xy)
    except Exception:
        return None

    hull_xy = xy[hull.vertices]
    c = np.mean(hull_xy, axis=0)
    d = hull_xy - c[None, :]
    ang = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])

    edges = np.linspace(-np.pi, np.pi, n_vertices + 1)
    pick = []
    for i in range(n_vertices):
        m = (ang >= edges[i]) & (ang < edges[i + 1])
        if not np.any(m):
            continue
        j = int(np.argmax(r[m]))
        pick.append(hull_xy[np.where(m)[0][j]])

    if len(pick) < 3:
        return None

    pick_xy = np.vstack(pick)
    d2 = pick_xy - np.mean(pick_xy, axis=0)[None, :]
    ang2 = np.arctan2(d2[:, 1], d2[:, 0])
    order = np.argsort(ang2)
    pick_xy = pick_xy[order]

    if pick_xy.shape[0] > n_vertices:
        pick_xy = pick_xy[:n_vertices]
    return pick_xy


def _disk_structure(radius_pix):
    r = int(max(1, round(float(radius_pix))))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= (r * r)

def safe_percentile(a, q, default=np.nan):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return default
    return float(np.nanpercentile(a, q))


def make_mu_flat_from_mask(flat_mask_2d, expo, pix_sr, dE_mev):
    """
    flat_mask_2d: (ny,nx) True/1 inside bubbles boundary
    expo: (nE,ny,nx) exposure [cm^2 s]
    pix_sr: scalar sr/pix
    dE_mev: (nE,) MeV bin widths
    Returns mu_flat: (nE,ny,nx) counts template for unit intensity inside mask
    """
    flat = flat_mask_2d.astype(float)[None, :, :]
    return flat * expo * float(pix_sr) * dE_mev[:, None, None]


def nnls_fit_per_energy(counts, templates_dict, fit_mask_2d, eps=1.0):
    """
    counts: (nE,ny,nx)
    templates_dict: name->(nE,ny,nx) counts
    fit_mask_2d: (ny,nx) True=use
    Returns coeffs: name->(nE,), model: (nE,ny,nx)
    """
    nE, ny, nx = counts.shape
    m = fit_mask_2d.ravel()
    if m.sum() < 10:
        raise RuntimeError("fit mask too small")

    names = list(templates_dict.keys())
    coeffs = {n: np.zeros(nE, float) for n in names}
    model = np.zeros_like(counts, dtype=float)

    for k in range(nE):
        y = counts[k].ravel()[m].astype(float)
        X = np.vstack([templates_dict[n][k].ravel()[m].astype(float) for n in names]).T

        # weighted NNLS ~ Poisson
        w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
        yw = y * w
        Xw = X * w[:, None]

        a, _ = nnls(Xw, yw)
        for i, n in enumerate(names):
            coeffs[n][k] = a[i]

        # model map
        mk = np.zeros((ny, nx), float)
        for ai, n in zip(a, names):
            mk += ai * templates_dict[n][k]
        model[k] = mk

    return coeffs, model


def _cleanup_binary_mask(mask_2d, *, r_open_pix, r_close_pix):
    if not np.any(mask_2d):
        return mask_2d
    st_open = _disk_structure(r_open_pix)
    st_close = _disk_structure(r_close_pix)
    m = binary_opening(mask_2d, structure=st_open)
    m = binary_closing(m, structure=st_close)
    m = binary_fill_holes(m)
    return m
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
THR_SIG = 0.3
LON_FIT_MAX_DEG = 20.0
MORPH_R_OPEN_DEG = 0.75
MORPH_R_CLOSE_DEG = 1.25

ROI_LON_DEG = 60.0
ROI_LAT_DEG = 60.0
CELL_DEG = 10.0

# Template inputs for Totani-style bubble construction (counts templates)
MU_GAS = os.path.join(DATA_DIR, "processed", "templates", "mu_gas_counts.fits")
MU_ICS = os.path.join(DATA_DIR, "processed", "templates", "mu_ics_counts.fits")
MU_ISO = os.path.join(DATA_DIR, "processed", "templates", "mu_iso_counts.fits")
MU_PS = os.path.join(DATA_DIR, "processed", "templates", "mu_ps_counts.fits")
MU_NFW = os.path.join(DATA_DIR, "processed", "templates", "mu_nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno_counts.fits")
MU_LOOPI = os.path.join(DATA_DIR, "processed", "templates", "mu_loopi_counts.fits")

# Prebuilt Totani-style bubble templates (made by make_templates/build_fermi_bubbles_templates.py)
# BUBBLES_POS_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_pos_dnde.fits")
# BUBBLES_NEG_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_neg_dnde.fits")
# BUBBLES_FLAT_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_dnde.fits")

# Optional: provide a FITS mask (ny,nx) on the counts grid representing Totani's
# final flat-bubble boundary (1 inside, 0 outside). If None, uses heuristic box.
BUBBLE_MASK_FITS = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_binary_mask.fits")

# Overlay settings
PLOT_EXT_SOURCES = True
MARKER_RADIUS_DEG = 1.0
MARKER_LW = 1.0

# -------------------------
def normalise_templates_counts(templates_counts, fit_mask_2d):
    """
    templates_counts: dict name -> (nE,ny,nx) counts template
    fit_mask_2d: (ny,nx) boolean mask used for the fit

    Returns:
      templates_norm: dict name -> (nE,ny,nx) counts template, where for each k:
                       sum_k(template[k][fit_mask]) = 1 (if original sum > 0)
      norms: dict name -> (nE,) original sums in the fit mask
    """
    templates_norm = {}
    norms = {}

    for name, T in templates_counts.items():
        T = np.asarray(T, dtype=float)
        if T.ndim != 3:
            raise ValueError(f"{name} must be 3D (nE,ny,nx), got shape {T.shape}")

        nE = T.shape[0]
        Tn = T.copy()
        s = np.zeros(nE, dtype=float)

        for k in range(nE):
            sk = float(np.nansum(Tn[k][fit_mask_2d]))
            s[k] = sk
            if sk > 0:
                Tn[k] /= sk

        templates_norm[name] = Tn
        norms[name] = s

    return templates_norm, norms

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
if PSMASK is not None and os.path.exists(str(PSMASK)):
    ps_mask = load_mask_any_shape(PSMASK, counts.shape)
else:
    ps_mask = np.ones_like(counts, dtype=bool)

Ectr = np.sqrt(eb["E_MIN"] * eb["E_MAX"]) / 1000.0  # MeV
dE   = (eb["E_MAX"] - eb["E_MIN"]) / 1000.0         # MeV
print(Ectr, "Ectr")
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
mu_loopi = _read_mu(MU_LOOPI, counts.shape)

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
# Totani-style FB pos/neg template construction
# -------------------------

# Flat bubbles boundary mask (2D)
# Predefined bubble boundary (2D keep mask inside bubbles envelope)
bubble_boundary_2d = None
if BUBBLE_MASK_FITS is not None:
    bubble_boundary_2d = load_mask_2d(BUBBLE_MASK_FITS, (ny, nx))  # True inside boundary
    print("[FB] Loaded predefined bubble boundary:", bubble_boundary_2d.shape, "sum=", int(bubble_boundary_2d.sum()))
else:
    bubble_boundary_2d = _heuristic_flat_bubble_mask(lon_w, lat2, lon_max=LON_FIT_MAX_DEG,
                                                     lat_min=DISK_CUT_DEG, lat_max=LAT_FIT_MAX_DEG)
    print("[FB] Using heuristic bubble boundary:", int(bubble_boundary_2d.sum()))

# Build counts templates dict (all in counts space)
templates = {
    "gas": mu_gas,
    "ics": mu_ics,
    "iso": mu_iso,
    "ps":  mu_ps,
    "nfw": mu_nfw,
    "loopI": mu_loopi,
}

# Choose construction bin near 4.3 GeV
k_construct = int(np.argmin(np.abs((Ectr / 1000.0) - 4.3)))
print("[FB] k_construct:", k_construct, "E~", (Ectr[k_construct] / 1000.0), "GeV")

# FIT mask (Totani-style): include disk, mask sources + ROI
fit_mask_2d = roi2d.copy()
fit_mask_2d &= ps_mask[k_construct]     # and ext keep-mask if you have it

# Flat bubbles counts template (unit intensity inside boundary)
mu_flat = make_mu_flat_from_mask(bubble_boundary_2d, expo, PIX_SR, dE)  # counts

# IMPORTANT: normalise ALL templates (including fb_flat) inside fit mask for NNLS stability
templates_fit = dict(templates)  # gas/ics/iso/ps/loopI/nfw...
templates_fit["fb_flat"] = mu_flat

templates_fit_norm, norms = normalise_templates_counts(templates_fit, fit_mask_2d)

coeffs_norm, model_norm = nnls_fit_per_energy(counts, templates_fit_norm, fit_mask_2d, eps=1.0)

# Convert fb_flat coefficient back to physical counts scaling
c_flat_phys = coeffs_norm["fb_flat"] / norms["fb_flat"]     # (nE,)
model = model_norm.copy()
resid_counts = counts - model

bubble_flat_counts = c_flat_phys[:, None, None] * mu_flat
bubble_image_counts = bubble_flat_counts + resid_counts

# Convert to flux (ph cm^-2 s^-1 sr^-1 MeV^-1)
denom = expo * PIX_SR * dE[:, None, None]
bubble_flux = np.full_like(bubble_image_counts, np.nan, dtype=float)
ok = denom > 0
bubble_flux[ok] = bubble_image_counts[ok] / denom[ok]

k_construct = 2
# -------------------------
# Plot
# -------------------------
sig_pix = 1.0 / BINSZ_DEG
bubble_flux_smooth = gaussian_filter(
    np.nan_to_num(bubble_flux, nan=0.0),
    sigma=(0.0, sig_pix, sig_pix)
)


sig_by_tgt = {}
res_by_tgt = {}
mask_by_tgt = {}
ccn_by_tgt = {}
ccs_by_tgt = {}

for itgt, tgt in enumerate(TARGET_GEV):
    k = int(np.argmin(np.abs(Ectr - tgt)))
    print("Energy bin:", k, "Ectr=", Ectr[k])

    tgt_gev = tgt / 1000.0

    mask2d = roi2d & disk_mask & ps_mask[k]
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
    res_s = smooth_nan_2d(res, sigma_pix=sig_pix)
    res_s[~mask2d] = np.nan

    denom_k = denom[k]
    resid_counts_s = np.full((ny, nx), np.nan, dtype=float)
    ok_sig = mask2d & np.isfinite(res_s) & np.isfinite(denom_k) & (denom_k > 0)
    resid_counts_s[ok_sig] = res_s[ok_sig] * denom_k[ok_sig]

    sig = np.full((ny, nx), np.nan, dtype=float)
    ok_sig2 = ok_sig & np.isfinite(counts[k])
    sig[ok_sig2] = resid_counts_s[ok_sig2] / np.sqrt(np.maximum(counts[k][ok_sig2], 0.0) + 1.0)
    sig_s = smooth_nan_2d(sig, sigma_pix=sig_pix)
    sig_s[~mask2d] = np.nan

    sig_by_tgt[float(tgt)] = sig_s
    res_by_tgt[float(tgt)] = res_s
    mask_by_tgt[float(tgt)] = mask2d
    # -------------------------------------------------------
    # NEW: split residual into POS and NEG regions inside predefined boundary
    # -------------------------------------------------------

    # Base region to define pos/neg:
    # - inside ROI
    # - outside disk (since you already mask it in mask2d)
    # - inside predefined bubble boundary
    base = roi2d & disk_mask & ps_mask[k_construct] & bubble_boundary_2d

    # Use residual flux map res_s (already smoothed and disk/ROI masked)
    # Use Totani bubble-image flux (smoothed) at construction bin
    r = bubble_flux_smooth[k_construct].copy()
    r[~base] = np.nan

    pos_vals = r[(r > 0) & np.isfinite(r)]
    neg_vals = r[(r < 0) & np.isfinite(r)]

    # Choose thresholds from tails INSIDE boundary
    pos_thr = safe_percentile(pos_vals, 70, default=np.nan)  # tune 85–95
    neg_thr = safe_percentile(neg_vals, 30, default=np.nan)  # tune 5–20 (more negative)

    pos_raw = np.zeros((ny, nx), dtype=bool)
    neg_raw = np.zeros((ny, nx), dtype=bool)

    if np.isfinite(pos_thr):
        pos_raw = base & np.isfinite(r) & (r > pos_thr)
    else:
        print("[WARN] no positive residual pixels inside boundary")

    if np.isfinite(neg_thr):
        neg_raw = base & np.isfinite(r) & (r < neg_thr)
    else:
        print("[WARN] no negative residual pixels inside boundary")

    # Optional cleanup (same morphology radii you already use)
    pos_clean = _cleanup_binary_mask(
        pos_raw,
        r_open_pix=(MORPH_R_OPEN_DEG / BINSZ_DEG),
        r_close_pix=(MORPH_R_CLOSE_DEG / BINSZ_DEG),
    )
    neg_clean = _cleanup_binary_mask(
        neg_raw,
        r_open_pix=(MORPH_R_OPEN_DEG / BINSZ_DEG),
        r_close_pix=(MORPH_R_CLOSE_DEG / BINSZ_DEG),
    )

    # Keep largest CC in North and South separately for pos and neg (optional but recommended)
    def keep_largest_cc(mask_bool):
        lbl, nlab = label(mask_bool)
        if nlab <= 0:
            return np.zeros_like(mask_bool, dtype=bool)
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        keep = int(np.argmax(sizes))
        return (lbl == keep)

    pos_n = keep_largest_cc(pos_clean & (lat2 > 0))
    pos_s = keep_largest_cc(pos_clean & (lat2 < 0))
    neg_n = keep_largest_cc(neg_clean & (lat2 > 0))
    neg_s = keep_largest_cc(neg_clean & (lat2 < 0))

    pos_mask_2d = pos_n | pos_s
    neg_mask_2d = neg_n | neg_s

    print(f"[POS/NEG] E={Ectr[k]:.1f} MeV  pos_pix={int(pos_mask_2d.sum())}  neg_pix={int(neg_mask_2d.sum())} "
        f"pos_thr={pos_thr}  neg_thr={neg_thr}")

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
    # Positive residual bubbles: cyan (north), magenta (south) — same as before
    ax.contour(pos_n.astype(float), levels=[0.5], linewidths=1.2, colors="c", alpha=0.95)
    ax.contour(pos_s.astype(float), levels=[0.5], linewidths=1.2, colors="m", alpha=0.95)

    # Negative residual bubbles: yellow (north), orange (south)
    ax.contour(neg_n.astype(float), levels=[0.5], linewidths=1.2, colors="y", alpha=0.95)
    ax.contour(neg_s.astype(float), levels=[0.5], linewidths=1.2, colors="orange", alpha=0.95)

    # Also outline the predefined boundary
    ax.contour(bubble_boundary_2d.astype(float), levels=[0.5], linewidths=1.0, colors="w", alpha=0.8)

    ax.set_title(f"Residual flux (E~{Ectr[k]:.1f} MeV)\nExtended-source + disk masked")
    ax.set_xlabel("Galactic longitude")
    ax.set_ylabel("Galactic latitude")
    plt.colorbar(im, ax=ax, label=r"residual flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
    plt.tight_layout()

    out = f"{OUTDIR}/residual_{tgt:.1f}MeV_extsrc.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("✓ wrote", out)

if pos_mask_2d is None or neg_mask_2d is None:
    raise RuntimeError("pos/neg masks were not created (check k_construct selection).")

# counts templates (nE,ny,nx)
mu_pos = pos_mask_2d.astype(float)[None, :, :] * expo * PIX_SR * dE[:, None, None]
mu_neg = neg_mask_2d.astype(float)[None, :, :] * expo * PIX_SR * dE[:, None, None]

# (optional) normalise per energy inside your fit region for stable NNLS later
fit_mask_2d = roi2d & disk_mask & ps_mask[k_construct]   # or roi2d & disk_mask & (ext keep mask)
mu_pos_norm = mu_pos.copy()
mu_neg_norm = mu_neg.copy()

pos_norm = np.zeros(nE, float)
neg_norm = np.zeros(nE, float)
for kk in range(nE):
    pos_norm[kk] = float(np.nansum(mu_pos_norm[kk][fit_mask_2d]))
    neg_norm[kk] = float(np.nansum(mu_neg_norm[kk][fit_mask_2d]))
    if pos_norm[kk] > 0: mu_pos_norm[kk] /= pos_norm[kk]
    if neg_norm[kk] > 0: mu_neg_norm[kk] /= neg_norm[kk]

hdr_out = wcs.to_header()
fits.writeto(os.path.join(OUTDIR, "mu_fb_pos_counts.fits"), mu_pos.astype(np.float32), header=hdr_out, overwrite=True)
fits.writeto(os.path.join(OUTDIR, "mu_fb_neg_counts.fits"), mu_neg.astype(np.float32), header=hdr_out, overwrite=True)
fits.writeto(os.path.join(OUTDIR, "mu_fb_pos_norm_counts.fits"), mu_pos_norm.astype(np.float32), header=hdr_out, overwrite=True)
fits.writeto(os.path.join(OUTDIR, "mu_fb_neg_norm_counts.fits"), mu_neg_norm.astype(np.float32), header=hdr_out, overwrite=True)

print("✓ wrote mu_fb_pos/neg counts templates")


# -------------------------
# Diagnostic plot: flat / pos / neg / flux_smooth + counts + mu_pos/mu_neg + histogram
# -------------------------
def apply_nan(A, mask2d):
    B = np.array(A, dtype=float, copy=True)
    B[~mask2d] = np.nan
    return B

# Display mask for plots: show ROI, hide disk band, hide masked sources (if any)
display_mask = roi2d.copy()
display_mask &= (np.abs(lat2) >= DISK_CUT_DEG)      # hide disk for display
display_mask &= ps_mask[k_construct]               # hide whatever your ps_mask removes at this energy

A_flat = apply_nan(bubble_boundary_2d.astype(float), display_mask)
A_pos  = apply_nan(pos_mask_2d.astype(float), display_mask)
A_neg  = apply_nan(neg_mask_2d.astype(float), display_mask)

# flux smooth at construction bin (symmetric scale)
A_flux = apply_nan(bubble_flux_smooth[k_construct], display_mask)
finite = np.isfinite(A_flux)
vmax = float(np.nanpercentile(np.abs(A_flux[finite]), 99)) if finite.any() else 1.0
if vmax <= 0:
    vmax = 1.0

# data + templates (counts) at construction bin
A_data = apply_nan(counts[k_construct], display_mask)
A_mu_pos = apply_nan(mu_pos[k_construct], display_mask)
A_mu_neg = apply_nan(mu_neg[k_construct], display_mask)

# histogram data
vals = bubble_flux_smooth[k_construct][display_mask]
vals = vals[np.isfinite(vals)]

fig = plt.figure(figsize=(14, 9))

# top row
ax00 = fig.add_subplot(2, 4, 1, projection=wcs)
ax01 = fig.add_subplot(2, 4, 2, projection=wcs)
ax02 = fig.add_subplot(2, 4, 3, projection=wcs)
ax03 = fig.add_subplot(2, 4, 4, projection=wcs)

# bottom row
ax10 = fig.add_subplot(2, 4, 5, projection=wcs)
ax11 = fig.add_subplot(2, 4, 6, projection=wcs)
ax12 = fig.add_subplot(2, 4, 7, projection=wcs)
ax13 = fig.add_subplot(2, 4, 8)  # histogram (no WCS)

# --- flat mask ---
im0 = ax00.imshow(A_flat, origin="lower", vmin=0.0, vmax=1.0)
ax00.set_title("Totani FB: flat_mask")
set_lon_ticks_wrapped(ax00, wcs, ny, nx)
fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)

# --- pos region ---
im1 = ax01.imshow(A_pos, origin="lower", vmin=0.0, vmax=1.0)
ax01.set_title(f"Totani FB: pos region (pos2d>0)\nnonzero pix: {int(np.nansum(pos_mask_2d))}")
set_lon_ticks_wrapped(ax01, wcs, ny, nx)
fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)

# --- neg region ---
im2 = ax02.imshow(A_neg, origin="lower", vmin=0.0, vmax=1.0)
ax02.set_title(f"Totani FB: neg region (neg2d>0)\nnonzero pix: {int(np.nansum(neg_mask_2d))}")
set_lon_ticks_wrapped(ax02, wcs, ny, nx)
fig.colorbar(im2, ax=ax02, fraction=0.046, pad=0.04)

# --- smoothed bubble-image flux (construction bin) ---
im3 = ax03.imshow(A_flux, origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu")
ax03.set_title(f"Totani FB: bubble_flux_smooth ({Ectr[k_construct]/1000:.2f} GeV)\n(symmetric scale)")
set_lon_ticks_wrapped(ax03, wcs, ny, nx)
fig.colorbar(im3, ax=ax03, fraction=0.046, pad=0.04)

# --- data counts ---
im4 = ax10.imshow(A_data, origin="lower")
ax10.set_title(f"Totani FB: data counts ({Ectr[k_construct]/1000:.2f} GeV)")
set_lon_ticks_wrapped(ax10, wcs, ny, nx)
fig.colorbar(im4, ax=ax10, fraction=0.046, pad=0.04)

# --- mu_pos counts template ---
im5 = ax11.imshow(A_mu_pos, origin="lower")
ax11.set_title(f"Totani FB: mu_pos counts template ({Ectr[k_construct]/1000:.2f} GeV)")
set_lon_ticks_wrapped(ax11, wcs, ny, nx)
fig.colorbar(im5, ax=ax11, fraction=0.046, pad=0.04)

# --- mu_neg counts template ---
im6 = ax12.imshow(A_mu_neg, origin="lower")
ax12.set_title(f"Totani FB: mu_neg counts template ({Ectr[k_construct]/1000:.2f} GeV)")
set_lon_ticks_wrapped(ax12, wcs, ny, nx)
fig.colorbar(im6, ax=ax12, fraction=0.046, pad=0.04)

# --- histogram ---
ax13.hist(vals, bins=80)
ax13.set_title(f"Totani FB: flux_smooth histogram ({Ectr[k_construct]/1000:.2f} GeV)\nN={vals.size}")
ax13.set_xlabel(r"flux [ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
ax13.set_ylabel("pixels")

# overlays (disk + ext sources + fit box) on WCS axes
disk_band_2d = (~disk_mask) & roi2d
for axx in (ax00, ax01, ax02, ax03, ax10, ax11, ax12):
    axx.imshow(np.where(disk_band_2d, 1.0, np.nan),
               origin="lower", cmap="gray", vmin=0.0, vmax=1.0, alpha=0.85, interpolation="nearest")
    axx.set_xlabel("l")
    axx.set_ylabel("b")

plt.tight_layout()
out_png = os.path.join(OUTDIR, "diagnostics_totani_posneg.png")
plt.savefig(out_png, dpi=200)
plt.close()
print("✓ wrote", out_png)
#     res_cc = np.where(cc_mask, res_s, np.nan)
#     res_cc_n = np.where(cc_mask_n, res_s, np.nan)
#     res_cc_s = np.where(cc_mask_s, res_s, np.nan)

#     fig = plt.figure(figsize=(12, 8))
#     ax1 = fig.add_subplot(221, projection=wcs)
#     ax2 = fig.add_subplot(222, projection=wcs)
#     ax3 = fig.add_subplot(223, projection=wcs)
#     ax4 = fig.add_subplot(224, projection=wcs)

#     im1 = ax1.imshow(res_s, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
#     # Positive residual bubbles: cyan (north), magenta (south) — same as before
#     ax1.contour(pos_n.astype(float), levels=[0.5], linewidths=1.2, colors="c", alpha=0.95)
#     ax1.contour(pos_s.astype(float), levels=[0.5], linewidths=1.2, colors="m", alpha=0.95)

#     # Negative residual bubbles: yellow (north), orange (south)
#     ax1.contour(neg_n.astype(float), levels=[0.5], linewidths=1.2, colors="y", alpha=0.95)
#     ax1.contour(neg_s.astype(float), levels=[0.5], linewidths=1.2, colors="orange", alpha=0.95)

#     # Also outline the predefined boundary
#     ax1.contour(bubble_boundary_2d.astype(float), levels=[0.5], linewidths=1.0, colors="w", alpha=0.8)

#     im2 = ax2.imshow(cc_mask.astype(float), origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
#     im3 = ax3.imshow(res_cc, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
#     im4 = ax4.imshow(res_cc_n, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu")
#     ax4.imshow(res_cc_s, origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu", alpha=0.85)

#     for axx in (ax1, ax2, ax3, ax4):
#         set_lon_ticks_wrapped(axx, wcs, ny, nx)
#         axx.contour(~mask2d, levels=[0.5], linewidths=0.6, colors="k", alpha=0.7)
#         add_fit_box_overlay(axx, wcs=wcs, lon_max_deg=LON_FIT_MAX_DEG, lat_max_deg=LAT_FIT_MAX_DEG, color="y", lw=1.0, alpha=0.9)
#         axx.set_xlabel("l")
#         axx.set_ylabel("b")

#     ax1.set_title("Residual + CC outline")
#     ax2.set_title("Largest connected component")
#     ax3.set_title("Residual masked to CC")
#     ax4.set_title("CC residual: North (opaque) + South (transparent overlay)")

#     plt.colorbar(im1, ax=ax1, fraction=0.046)
#     plt.colorbar(im3, ax=ax3, fraction=0.046)
#     plt.colorbar(im4, ax=ax4, fraction=0.046)
#     plt.tight_layout()

#     out_m = f"{OUTDIR}/morphology_residual_{tgt:.1f}MeV_extsrc.png"
#     plt.savefig(out_m, dpi=200)
#     plt.close()
#     print("✓ wrote", out_m)


# if len(TARGET_GEV) >= 2:
#     t1 = float(TARGET_GEV[0])
#     t2 = float(TARGET_GEV[1])
#     if (t1 in sig_by_tgt) and (t2 in sig_by_tgt):
#         w1, w2 = 0.5, 0.5
#         sig_mix = w1 * sig_by_tgt[t1] + w2 * sig_by_tgt[t2]
#         mask_mix = mask_by_tgt[t1] & mask_by_tgt[t2]
#         sig_mix[~mask_mix] = np.nan

#         above = mask_mix & np.isfinite(sig_mix) & (sig_mix >= THR_SIG)

#         above_n = above & (lat2 > 0.0)
#         lbln, nlabn = label(above_n)
#         if nlabn > 0:
#             sizes = np.bincount(lbln.ravel())
#             sizes[0] = 0
#             keep = int(np.argmax(sizes))
#             cc_n = (lbln == keep)
#         else:
#             cc_n = np.zeros((ny, nx), dtype=bool)

#         above_s = above & (lat2 < 0.0)
#         lbls, nlabs = label(above_s)
#         if nlabs > 0:
#             sizes = np.bincount(lbls.ravel())
#             sizes[0] = 0
#             keep = int(np.argmax(sizes))
#             cc_s = (lbls == keep)
#         else:
#             cc_s = np.zeros((ny, nx), dtype=bool)

#         cc = cc_n | cc_s

#         box_mask = (np.abs(lon_w) <= LON_FIT_MAX_DEG) & (np.abs(lat2) <= LAT_FIT_MAX_DEG) & roi2d

#         # Fit octagons based on the 1500 MeV morphology (t1), not the intermediate sig-mix
#         base_cc_n = ccn_by_tgt.get(t1)
#         base_cc_s = ccs_by_tgt.get(t1)
#         if base_cc_n is None or base_cc_s is None:
#             base_cc_n = cc_n
#             base_cc_s = cc_s

#         cc_n_in_box = base_cc_n & box_mask
#         cc_s_in_box = base_cc_s & box_mask
#         oct_xy_n = fit_octagon_from_mask(cc_n_in_box, n_vertices=8)
#         oct_xy_s = fit_octagon_from_mask(cc_s_in_box, n_vertices=8)

#         h2 = wcs.to_header()
#         fits.writeto(os.path.join(OUTDIR, "ccmask_intermediate_sigmix.fits"), cc.astype(np.int16), header=h2, overwrite=True)
#         fits.writeto(os.path.join(OUTDIR, "ccmask_intermediate_sigmix_north.fits"), cc_n.astype(np.int16), header=h2, overwrite=True)
#         fits.writeto(os.path.join(OUTDIR, "ccmask_intermediate_sigmix_south.fits"), cc_s.astype(np.int16), header=h2, overwrite=True)
#         print("✓ wrote", os.path.join(OUTDIR, "ccmask_intermediate_sigmix.fits"))

#         fig = plt.figure(figsize=(12, 4))
#         ax1 = fig.add_subplot(131, projection=wcs)
#         ax2 = fig.add_subplot(132, projection=wcs)
#         ax3 = fig.add_subplot(133, projection=wcs)

#         res_ref = res_by_tgt.get(t2, res_by_tgt[t1])
#         im1 = ax1.imshow(res_ref, origin="lower", cmap="RdBu")
#         ax1.contour(cc_n.astype(float), levels=[0.5], colors="c", linewidths=1.2)
#         ax1.contour(cc_s.astype(float), levels=[0.5], colors="m", linewidths=1.2)
#         if oct_xy_n is not None:
#             ax1.plot(
#                 np.r_[oct_xy_n[:, 0], oct_xy_n[0, 0]],
#                 np.r_[oct_xy_n[:, 1], oct_xy_n[0, 1]],
#                 color="c",
#                 linewidth=2.0,
#             )
#         if oct_xy_s is not None:
#             ax1.plot(
#                 np.r_[oct_xy_s[:, 0], oct_xy_s[0, 0]],
#                 np.r_[oct_xy_s[:, 1], oct_xy_s[0, 1]],
#                 color="m",
#                 linewidth=2.0,
#             )

#         im2 = ax2.imshow(sig_mix, origin="lower", cmap="viridis")
#         ax2.contour(cc.astype(float), levels=[0.5], colors="w", linewidths=1.0)
#         if oct_xy_n is not None:
#             ax2.plot(
#                 np.r_[oct_xy_n[:, 0], oct_xy_n[0, 0]],
#                 np.r_[oct_xy_n[:, 1], oct_xy_n[0, 1]],
#                 color="c",
#                 linewidth=2.0,
#             )
#         if oct_xy_s is not None:
#             ax2.plot(
#                 np.r_[oct_xy_s[:, 0], oct_xy_s[0, 0]],
#                 np.r_[oct_xy_s[:, 1], oct_xy_s[0, 1]],
#                 color="m",
#                 linewidth=2.0,
#             )

#         im3 = ax3.imshow(cc.astype(float), origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
#         if oct_xy_n is not None:
#             ax3.plot(
#                 np.r_[oct_xy_n[:, 0], oct_xy_n[0, 0]],
#                 np.r_[oct_xy_n[:, 1], oct_xy_n[0, 1]],
#                 color="c",
#                 linewidth=2.0,
#             )
#         if oct_xy_s is not None:
#             ax3.plot(
#                 np.r_[oct_xy_s[:, 0], oct_xy_s[0, 0]],
#                 np.r_[oct_xy_s[:, 1], oct_xy_s[0, 1]],
#                 color="m",
#                 linewidth=2.0,
#             )

#         for axx in (ax1, ax2, ax3):
#             set_lon_ticks_wrapped(axx, wcs, ny, nx)
#             axx.contour(~mask_mix, levels=[0.5], linewidths=0.6, colors="k", alpha=0.7)
#             add_fit_box_overlay(axx, wcs=wcs, lon_max_deg=LON_FIT_MAX_DEG, lat_max_deg=LAT_FIT_MAX_DEG, color="y", lw=1.0, alpha=0.9)

#         ax1.set_title("Intermediate CC overlay (N cyan / S magenta)")
#         ax2.set_title(f"sig_mix = 0.5*sig({t1:.0f}) + 0.5*sig({t2:.0f}), thr={THR_SIG}")
#         ax3.set_title("Intermediate CC mask")

#         plt.colorbar(im1, ax=ax1, fraction=0.046)
#         plt.colorbar(im2, ax=ax2, fraction=0.046)
#         plt.tight_layout()
#         out_mix = os.path.join(OUTDIR, "morphology_intermediate_sigmix.png")
#         plt.savefig(out_mix, dpi=200)
#         plt.close()
#         print("✓ wrote", out_mix)

#         if oct_xy_n is not None:
#             lon_o, lat_o = wcs.pixel_to_world_values(oct_xy_n[:, 0], oct_xy_n[:, 1])
#             lon_o = ((np.asarray(lon_o) + 180.0) % 360.0) - 180.0
#             lat_o = np.asarray(lat_o)
#             out_oct = os.path.join(OUTDIR, "octagon_vertices_base1500_north.txt")
#             arr = np.column_stack([lon_o, lat_o, oct_xy_n[:, 0], oct_xy_n[:, 1]])
#             np.savetxt(out_oct, arr, header="lon_deg lat_deg x_pix y_pix")
#             print("✓ wrote", out_oct)

#         if oct_xy_s is not None:
#             lon_o, lat_o = wcs.pixel_to_world_values(oct_xy_s[:, 0], oct_xy_s[:, 1])
#             lon_o = ((np.asarray(lon_o) + 180.0) % 360.0) - 180.0
#             lat_o = np.asarray(lat_o)
#             out_oct = os.path.join(OUTDIR, "octagon_vertices_base1500_south.txt")
#             arr = np.column_stack([lon_o, lat_o, oct_xy_s[:, 0], oct_xy_s[:, 1]])
#             np.savetxt(out_oct, arr, header="lon_deg lat_deg x_pix y_pix")
#             print("✓ wrote", out_oct)


# if len(TARGET_GEV) >= 2:
#     t1 = float(TARGET_GEV[0])
#     t2 = float(TARGET_GEV[1])
#     if (t1 in res_by_tgt) and (t2 in res_by_tgt) and (t1 in ccn_by_tgt) and (t1 in ccs_by_tgt):
#         box_mask = (np.abs(lon_w) <= LON_FIT_MAX_DEG) & (np.abs(lat2) <= LAT_FIT_MAX_DEG) & roi2d
#         oct_xy_n = fit_octagon_from_mask(ccn_by_tgt[t1] & box_mask, n_vertices=8)
#         oct_xy_s = fit_octagon_from_mask(ccs_by_tgt[t1] & box_mask, n_vertices=8)

#         disk_band_2d = (~disk_mask) & roi2d

#         fig = plt.figure(figsize=(10.5, 4.0))
#         axL = fig.add_subplot(121, projection=wcs)
#         axR = fig.add_subplot(122, projection=wcs)

#         resL = res_by_tgt[t1]
#         resR = res_by_tgt[t2]

#         vminL, vmaxL = -5e-10, 5e-10
#         vminR, vmaxR = -5e-11, 5e-11

#         imL = axL.imshow(resL, origin="lower", vmin=vminL, vmax=vmaxL, cmap="RdBu")
#         imR = axR.imshow(resR, origin="lower", vmin=vminR, vmax=vmaxR, cmap="RdBu")

#         for axx, title in ((axL, "1.5 GeV"), (axR, "4.3 GeV")):
#             axx.imshow(
#                 np.where(disk_band_2d, 1.0, np.nan),
#                 origin="lower",
#                 cmap="gray",
#                 vmin=0.0,
#                 vmax=1.0,
#                 alpha=0.85,
#                 interpolation="nearest",
#             )
#             set_lon_ticks_wrapped(axx, wcs, ny, nx)
#             axx.set_xlabel(r"longitude $l$ [deg]")
#             axx.set_ylabel(r"latitude $b$ [deg]")
#             axx.text(0.03, 0.95, title, transform=axx.transAxes, color="w", ha="left", va="top", fontsize=12)

#             if oct_xy_n is not None:
#                 axx.plot(
#                     np.r_[oct_xy_n[:, 0], oct_xy_n[0, 0]],
#                     np.r_[oct_xy_n[:, 1], oct_xy_n[0, 1]],
#                     color="w",
#                     linewidth=1.4,
#                     alpha=0.95,
#                 )
#             if oct_xy_s is not None:
#                 axx.plot(
#                     np.r_[oct_xy_s[:, 0], oct_xy_s[0, 0]],
#                     np.r_[oct_xy_s[:, 1], oct_xy_s[0, 1]],
#                     color="w",
#                     linewidth=1.4,
#                     alpha=0.95,
#                 )

#         cbarL = fig.colorbar(imL, ax=axL, orientation="horizontal", pad=0.02, fraction=0.08, location="top")
#         cbarL.set_label(r"flux  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
#         cbarR = fig.colorbar(imR, ax=axR, orientation="horizontal", pad=0.02, fraction=0.08, location="top")
#         cbarR.set_label(r"flux  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")

#         plt.tight_layout()
#         out_ref = os.path.join(OUTDIR, "morphology_reference_style.png")
#         plt.savefig(out_ref, dpi=200)
#         plt.close()
#         print("✓ wrote", out_ref)


# print("✓ Done:", OUTDIR)
