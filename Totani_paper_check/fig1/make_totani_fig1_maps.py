#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
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
TARGET_GEV = [1500, 4300]  # GeV

ROI_LON_DEG = 60.0
ROI_LAT_DEG = 60.0
CELL_DEG = 10.0

# Template inputs for Totani-style bubble construction (counts templates)
MU_GAS = os.path.join(DATA_DIR, "processed", "templates", "mu_gas_counts.fits")
MU_ICS = os.path.join(DATA_DIR, "processed", "templates", "mu_ics_counts.fits")
MU_ISO = os.path.join(DATA_DIR, "processed", "templates", "mu_iso_counts.fits")
MU_PS = os.path.join(DATA_DIR, "processed", "templates", "mu_ps_counts.fits")
MU_NFW = os.path.join(DATA_DIR, "processed", "templates", "mu_nfw_rho2.5_g1.25_counts.fits")
MU_LOOPI = os.path.join(DATA_DIR, "processed", "templates", "mu_loopI_counts.fits")

# Prebuilt Totani-style bubble templates (made by make_templates/build_fermi_bubbles_templates.py)
BUBBLES_POS_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_pos_dnde.fits")
BUBBLES_NEG_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_neg_dnde.fits")
BUBBLES_FLAT_DNDE = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_binary_dnde.fits")

# Provide a FITS mask (ny,nx) on the counts grid representing the flat-bubble boundary.
# Falls back to the nonzero region of BUBBLES_FLAT_DNDE if this file is missing.
BUBBLE_MASK_FITS = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_binary_mask.fits")

# Overlay settings
PLOT_EXT_SOURCES = True
MARKER_RADIUS_DEG = 1.0
MARKER_LW = 1.0

# -------------------------


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


def build_bubble_image_counts_cellwise(
    counts_2d,
    *,
    mu_gas_2d,
    mu_ics_2d,
    mu_iso_2d,
    mu_ps_2d,
    mu_nfw_2d,
    mu_loopi_2d=None,
    mu_flat_2d,
    fit_mask2d,
    lon_wrapped_deg,
    lat_deg,
    cell_deg=10.0,
):
    bubble_img_counts = np.full_like(counts_2d, np.nan, dtype=float)

    l_edges = np.arange(-ROI_LON_DEG, ROI_LON_DEG + 1e-6, cell_deg)
    b_edges = np.arange(-ROI_LAT_DEG, ROI_LAT_DEG + 1e-6, cell_deg)

    for l0 in l_edges[:-1]:
        l1 = l0 + cell_deg
        in_l = (lon_wrapped_deg >= l0) & (lon_wrapped_deg < l1)
        for b0 in b_edges[:-1]:
            b1 = b0 + cell_deg
            cell_mask = fit_mask2d & in_l & (lat_deg >= b0) & (lat_deg < b1)
            if not np.any(cell_mask):
                continue

            mu_components = [mu_gas_2d, mu_ics_2d, mu_iso_2d, mu_ps_2d, mu_nfw_2d]
            if mu_loopi_2d is not None:
                mu_components.append(mu_loopi_2d)
            mu_components.append(mu_flat_2d)

            i_bub = len(mu_components) - 1
            A = fit_bin_weighted_nnls(counts_2d, mu_components, cell_mask)

            model_all_cell = np.zeros(int(np.sum(cell_mask)), dtype=float)
            for i, mu in enumerate(mu_components):
                model_all_cell += A[i] * mu[cell_mask]
            residual_cell = counts_2d[cell_mask] - model_all_cell
            bubble_img_counts[cell_mask] = (A[i_bub] * mu_flat_2d[cell_mask]) + residual_cell

    return bubble_img_counts


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
    expo_raw = np.array(h[0].data, dtype=np.float64)
    E_expo = read_expcube_energies_mev(h)

nE, ny, nx = counts.shape
ps_mask = load_mask_any_shape(PSMASK, counts.shape)

Ectr = np.sqrt(eb["E_MIN"] * eb["E_MAX"]) / 1000.0  # MeV
dE   = (eb["E_MAX"] - eb["E_MIN"]) / 1000.0         # MeV

expo = resample_exposure_logE_interp(expo_raw, E_expo, Ectr)

wcs = WCS(hdr).celestial

omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=BINSZ_DEG)

# Disk cut mask
yy, xx = np.mgrid[:ny, :nx]
_, lat = wcs.pixel_to_world_values(xx, yy)
disk_mask = np.abs(lat) >= DISK_CUT_DEG

# Lon/lat for bubble mask
lon, lat2 = wcs.pixel_to_world_values(xx, yy)
lon_w = ((lon + 180.0) % 360.0) - 180.0

roi2d = (np.abs(lon_w) <= ROI_LON_DEG) & (np.abs(lat2) <= ROI_LAT_DEG)

# Flux cube
denom = expo * omega[None, :, :] * dE[:, None, None]
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

mu_loopi = None
if "MU_LOOPI" in globals() and (MU_LOOPI is not None) and os.path.exists(MU_LOOPI):
    mu_loopi = _read_mu(MU_LOOPI, counts.shape)

# Load prebuilt bubble templates (dN/dE; positive/negative are nonnegative by construction)
bubbles_pos_dnde = _read_mu(BUBBLES_POS_DNDE, counts.shape)
bubbles_neg_dnde = _read_mu(BUBBLES_NEG_DNDE, counts.shape)
bubbles_flat_dnde = _read_mu(BUBBLES_FLAT_DNDE, counts.shape)

# Extended sources (overlay)
ext_coords = read_extended_sources(PSC) if PLOT_EXT_SOURCES else None
if ext_coords is not None:
    ext_x, ext_y = wcs.world_to_pixel(ext_coords)
    ext_x, ext_y = np.asarray(ext_x), np.asarray(ext_y)
    onmap = (ext_x >= 0) & (ext_x < nx) & (ext_y >= 0) & (ext_y < ny)
    ext_x, ext_y = ext_x[onmap], ext_y[onmap]
    print(f"[EXT] Overlaying {len(ext_x)} extended sources on map")

marker_r_pix = MARKER_RADIUS_DEG / BINSZ_DEG

# ... all your imports + helpers unchanged ...

# -------------------------
# Plot
# -------------------------
sigma_pix = 1 / BINSZ_DEG

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

    # Latitude-profile subtraction (Totani-style). Compute row-by-row to avoid
    # RuntimeWarning: Mean of empty slice when a whole row is masked.
    T_b = np.full(ny, np.nan, dtype=float)
    for iy in range(ny):
        row = f[iy]
        if np.any(np.isfinite(row)):
            T_b[iy] = float(np.nanmean(row))
    res = f - T_b[:, None]
    res_s = smooth_nan_2d(res, sigma_pix=sigma_pix)
    res_s[~mask2d] = np.nan

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

    if (BUBBLE_MASK_FITS is not None) and os.path.exists(BUBBLE_MASK_FITS):
        bubble_boundary_2d = load_mask_2d(BUBBLE_MASK_FITS, (ny, nx)) & roi2d
    else:
        bubble_boundary_2d = (bubbles_flat_dnde[k] > 0) & roi2d
    plot_boundary_polygon(ax, bubble_boundary_2d, color="w", lw=1.8, alpha=0.95, simplify_eps_pix=2.5)

    ax.set_title(f"Residual flux (E~{Ectr[k]:.1f} MeV)\nExtended-source + disk masked")
    ax.set_xlabel("Galactic longitude")
    ax.set_ylabel("Galactic latitude")
    plt.colorbar(im, ax=ax, label=r"residual flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
    plt.tight_layout()

    out = f"{OUTDIR}/residual_{tgt:.1f}MeV_extsrc.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("✓ wrote", out)

    # Totani-style bubble image construction: fit background components per 10° cell,
    # then take (best-fit bubbles + residual) in counts space and convert to flux.
    mu_flat_2d = bubbles_flat_dnde[k] * expo[k] * omega * dE[k]
    bubble_img_counts = build_bubble_image_counts_cellwise(
        counts[k],
        mu_gas_2d=mu_gas[k],
        mu_ics_2d=mu_ics[k],
        mu_iso_2d=mu_iso[k],
        mu_ps_2d=mu_ps[k],
        mu_nfw_2d=mu_nfw[k],
        mu_loopi_2d=(mu_loopi[k] if mu_loopi is not None else None),
        mu_flat_2d=mu_flat_2d,
        fit_mask2d=mask2d,
        lon_wrapped_deg=lon_w,
        lat_deg=lat2,
        cell_deg=CELL_DEG,
    )

    denom2 = expo[k] * omega * dE[k]
    bubble_flux = np.full((ny, nx), np.nan, dtype=float)
    good = mask2d & np.isfinite(bubble_img_counts) & np.isfinite(denom2) & (denom2 > 0)
    bubble_flux[good] = bubble_img_counts[good] / denom2[good]

    bubble_sm_plot = smooth_nan_2d(bubble_flux, sigma_pix=sigma_pix)
    bubble_sm_plot[~mask2d] = np.nan

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection=wcs)
    if itgt == 0:
        vmin2, vmax2 = -5e-10, 5e-10
    else:
        vmin2, vmax2 = -5e-11, 5e-11
    im = ax.imshow(bubble_sm_plot, origin="lower", vmin=vmin2, vmax=vmax2, cmap="RdBu")
    set_lon_ticks_wrapped(ax, wcs, ny, nx)

    disk_band = (~disk_mask)  # True inside |b|<DISK_CUT_DEG
    add_totani_overlays(
        ax,
        disk_band_2d=disk_band,
        bubble_boundary_2d=bubble_boundary_2d,
        ext_x=ext_x if ext_coords is not None else None,
        ext_y=ext_y if ext_coords is not None else None,
        marker_r_pix=marker_r_pix,
    )

    # Panel label like Totani (top-left)
    ax.text(
        0.03,
        0.95,
        f"{tgt_gev:.1f} GeV",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="w",
        fontsize=12,
        weight="bold",
    )

    ax.set_title(r"flux  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
    ax.set_xlabel("Galactic longitude")
    ax.set_ylabel("Galactic latitude")
    plt.colorbar(im, ax=ax, label=r"flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
    plt.tight_layout()

    out2 = f"{OUTDIR}/bubble_image_{tgt:.1f}MeV_extsrc.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    print("✓ wrote", out2)

print("✓ Done:", OUTDIR)
