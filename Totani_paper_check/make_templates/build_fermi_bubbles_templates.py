#!/usr/bin/env python3
"""
Totani-style Fermi Bubbles template construction:

1) Choose one or two energies (e.g. 1.5 GeV and 4.3 GeV) where bubbles are visible.
2) Fit templates including an initial "flat bubbles" spatial template (hard edge).
3) Make bubble-image = (best-fit flat bubbles counts) + residual counts.
4) Convert to flux (divide by exposure * omega * dE).
5) Smooth with Gaussian sigma=1 deg.
6) From the 4.3 GeV smoothed bubble-image flux map, define:
      pos_template = positive flux pixels (keep values, or just 1s)
      neg_template = negative flux pixels (keep abs values, or just 1s)
   (Both are spatial templates that later get their own spectra in the fit.)
"""
LON_FIT_MAX_DEG = 20.0
MORPH_R_OPEN_DEG = 0.75
MORPH_R_CLOSE_DEG = 1.25

ROI_LON_DEG = 60.0
ROI_LAT_DEG = 60.0
CELL_DEG = 10.0
#!/usr/bin/env python3
import os
from scipy.ndimage import label
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path
from scipy.optimize import minimize
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
import numpy as np
from typing import Dict, Tuple, Union
from scipy.optimize import nnls

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    load_mask_any_shape,
    write_cube,
)
from totani_helpers.fit_utils import *
from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import os
import argparse
from astropy.io import fits

from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

def _disk_structure(radius_pix):
    r = int(max(1, round(float(radius_pix))))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= (r * r)

def _cleanup_binary_mask(mask_2d, *, r_open_pix, r_close_pix):
    if not np.any(mask_2d):
        return mask_2d
    st_open = _disk_structure(r_open_pix)
    st_close = _disk_structure(r_close_pix)
    m = binary_opening(mask_2d, structure=st_open)
    m = binary_closing(m, structure=st_close)
    m = binary_fill_holes(m)
    return m

def _read_vertices_txt(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        raise RuntimeError(f"Vertex file {path} has <3 valid points")
    return np.asarray(pts, dtype=float)

def _poly_mask_lonlat(*, lon2d, lat2d, verts_lonlat):
    lon2d = np.asarray(lon2d, dtype=float)
    lat2d = np.asarray(lat2d, dtype=float)
    verts_lonlat = np.asarray(verts_lonlat, dtype=float)
    if verts_lonlat.ndim != 2 or verts_lonlat.shape[1] != 2:
        raise ValueError("verts_lonlat must be (N,2) in (lon,lat)")

    p = Path(verts_lonlat)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = p.contains_points(pts)
    return inside.reshape(lon2d.shape)

def plot_residual_sign_diagnostics(
    *,
    out,                 # BubbleTemplates
    counts,              # (nE,ny,nx) data counts
    expo,                # (nE,ny,nx)
    omega_2d,            # (ny,nx)
    dE_mev,              # (nE,) or scalar
    k=2,
    roi2d=None,
    srckeep2d=None,      # True where keep
    lat_deg_2d=None,
    mask_disk_deg=10.0,
    Ectr_gev=None,
    title_prefix="",
):
    nE, ny, nx = counts.shape

    # build display mask
    m = np.ones((ny, nx), dtype=bool)
    if roi2d is not None:
        m &= roi2d.astype(bool)
    if srckeep2d is not None:
        m &= srckeep2d.astype(bool)
    if lat_deg_2d is not None:
        m &= (np.abs(lat_deg_2d) >= float(mask_disk_deg))

    # residual counts is available directly
    resid_counts_k = np.asarray(out.bubble_image_counts[k], float) - np.asarray(out.flat_counts_template[k], float)
    # Explanation:
    # bubble_image_counts = bubble_flat_counts + resid_counts
    # so resid_counts = bubble_image_counts - bubble_flat_counts
    # and bubble_flat_counts = coeff*mu_flat, but you don't store it in BubbleTemplates;
    # if you *do* have coeff["fb_flat"] in scope, use that instead.
    #
    # However: if your out.bubble_image_counts already includes flat+resid, the sign check
    # we care about is on out.bubble_image_flux_smooth[k] (below). Resid_counts_k is optional.

    # bubble-image flux smoothed at k (this is what defines pos/neg templates)
    bubble_flux_smooth_k = np.asarray(out.bubble_image_flux_smooth[k], float)

    # Also compute *unsmoothed* flux from bubble_image_counts for comparison
    bubble_flux_k = counts_to_flux(out.bubble_image_counts[k], expo[k], omega_2d, np.asarray(dE_mev)[k] if np.ndim(dE_mev) else dE_mev)

    # mask to nan for plotting
    def mn(a):
        a = np.asarray(a, float).copy()
        a[~m] = np.nan
        return a

    A_flux  = mn(bubble_flux_k)
    A_smoo  = mn(bubble_flux_smooth_k)

    # stats
    vals = A_smoo[np.isfinite(A_smoo)]
    neg_frac = float(np.mean(vals < 0)) if vals.size else 0.0
    vmin = float(np.nanmin(vals)) if vals.size else np.nan
    vmax = float(np.nanmax(vals)) if vals.size else np.nan

    Etxt = f"{np.asarray(Ectr_gev)[k]:.3g} GeV" if Ectr_gev is not None else f"k={k}"
    print(f"[{Etxt}] bubble_flux_smooth: min={vmin:.3e}, max={vmax:.3e}, neg_frac={neg_frac:.4f}, N={vals.size}")

    # symmetric color scale
    if vals.size:
        vmax_abs = float(np.nanpercentile(np.abs(vals), 99))
        if not np.isfinite(vmax_abs) or vmax_abs <= 0:
            vmax_abs = 1.0
    else:
        vmax_abs = 1.0

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axs[0].imshow(A_flux, origin="lower")
    axs[0].set_title(f"{title_prefix}bubble_image_flux (unsmoothed) {Etxt}")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(A_smoo, origin="lower", vmin=-vmax_abs, vmax=vmax_abs)
    axs[1].set_title(f"{title_prefix}bubble_image_flux_smooth {Etxt}\nneg_frac={neg_frac:.3f}")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].hist(vals, bins=120)
    axs[2].set_title(f"{title_prefix}hist(bubble_flux_smooth) {Etxt}")
    axs[2].set_xlabel("flux (ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
    axs[2].set_ylabel("pixels")

    for ax in axs[:2]:
        ax.set_xticks([]); ax.set_yticks([])

    plt.show()

def plot_fb_summary(
    *,
    out,
    counts,            # (nE,ny,nx)
    mu_pos,            # (nE,ny,nx) counts template
    mu_neg,            # (nE,ny,nx) counts template
    Ectr_gev=None,     # (nE,) optional
    k=2,
    roi2d=None,        # (ny,nx) keep
    srckeep2d=None,    # (ny,nx) keep (extended sources mask as KEEP)
    lat_deg_2d=None,   # (ny,nx) for |b| masking
    mask_disk_deg=10.0,
    title_prefix="",
):
    nE, ny, nx = counts.shape
    if not (0 <= k < nE):
        raise ValueError(f"k={k} out of range (nE={nE})")

    # --- build display mask ---
    disp = np.ones((ny, nx), dtype=bool)
    if roi2d is not None:
        disp &= roi2d.astype(bool)
    if srckeep2d is not None:
        disp &= srckeep2d.astype(bool)
    if lat_deg_2d is not None:
        disp &= (np.abs(lat_deg_2d) >= float(mask_disk_deg))

    def M(a):
        a = np.asarray(a, float).copy()
        a[~disp] = np.nan
        return a

    # --- pull maps ---
    flat = M(out.flat_mask.astype(float))
    flux = M(np.asarray(out.bubble_image_flux_smooth[k], float))

    # show pos/neg as *regions* (binary), even if your templates are flux-weighted
    pos_region = M((np.asarray(out.pos_template) > 0).astype(float))
    neg_region = M((np.asarray(out.neg_template) > 0).astype(float))

    data_k   = M(counts[k])
    mu_pos_k = M(mu_pos[k])
    mu_neg_k = M(mu_neg[k])

    # --- scales ---
    # symmetric scale for flux
    finite = np.isfinite(flux)
    vmax_flux = np.nanpercentile(np.abs(flux[finite]), 99) if finite.any() else 1.0
    vmax_flux = float(vmax_flux) if np.isfinite(vmax_flux) and vmax_flux > 0 else 1.0

    # percentile scales for counts
    def vmax_p(a, p=99.5):
        f = np.isfinite(a)
        if not f.any():
            return 1.0
        v = np.nanpercentile(a[f], p)
        return float(v) if np.isfinite(v) and v > 0 else 1.0

    vmax_data = vmax_p(data_k, 99.5)
    vmax_pos  = vmax_p(mu_pos_k, 99.5)
    vmax_neg  = vmax_p(mu_neg_k, 99.5)

    Etxt = f"{np.asarray(Ectr_gev)[k]:.3g} GeV" if Ectr_gev is not None else f"k={k}"

    # --- figure ---
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])

    # flat mask
    im = ax0.imshow(flat, origin="lower")
    ax0.set_title(f"{title_prefix}flat_mask")
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # pos/neg regions
    im = ax1.imshow(pos_region, origin="lower")
    ax1.set_title(f"{title_prefix}pos region (pos_template>0)\nnonzero pix: {int(np.nansum(pos_region))}")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    im = ax2.imshow(neg_region, origin="lower")
    ax2.set_title(f"{title_prefix}neg region (neg_template>0)\nnonzero pix: {int(np.nansum(neg_region))}")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # smoothed flux bubble-image
    im = ax3.imshow(flux, origin="lower", vmin=-vmax_flux, vmax=vmax_flux)
    ax3.set_title(f"{title_prefix}bubble_image_flux_smooth ({Etxt})\n(symmetric scale)")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # data counts @k
    im = ax4.imshow(data_k, origin="lower", vmin=0, vmax=vmax_data)
    ax4.set_title(f"{title_prefix}data counts ({Etxt})")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # mu_pos / mu_neg counts templates @k
    im = ax5.imshow(mu_pos_k, origin="lower", vmin=0, vmax=vmax_pos)
    ax5.set_title(f"{title_prefix}mu_pos counts template ({Etxt})")
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    im = ax6.imshow(mu_neg_k, origin="lower", vmin=0, vmax=vmax_neg)
    ax6.set_title(f"{title_prefix}mu_neg counts template ({Etxt})")
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # histogram of flux values inside display mask
    vals = flux[np.isfinite(flux)]
    ax7.hist(vals, bins=80)
    ax7.set_title(f"{title_prefix}flux_smooth histogram ({Etxt})\nN={vals.size}")
    ax7.set_xlabel("flux (ph cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)")
    ax7.set_ylabel("pixels")

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("diagnostics.png")
    plt.show()

def plot_fb_shape_comparison(
    *,
    bubble_flux_smooth_2d,
    mask_2d,
    orig_pos_2d,
    orig_neg_2d,
    new_pos_2d,
    new_neg_2d,
    title_prefix="",
    outpath=None,
):
    mask_2d = np.asarray(mask_2d, dtype=bool)

    def M(a):
        a = np.asarray(a, float).copy()
        a[~mask_2d] = np.nan
        return a

    flux = M(bubble_flux_smooth_2d)
    finite = np.isfinite(flux)
    vmax = float(np.nanpercentile(np.abs(flux[finite]), 99)) if finite.any() else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0

    op = M((np.asarray(orig_pos_2d) > 0).astype(float))
    on = M((np.asarray(orig_neg_2d) > 0).astype(float))
    np2 = M((np.asarray(new_pos_2d) > 0).astype(float))
    nn2 = M((np.asarray(new_neg_2d) > 0).astype(float))

    fig, axs = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

    im = axs[0, 0].imshow(flux, origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu")
    axs[0, 0].set_title(f"{title_prefix}smoothed bubble flux (construction)")
    plt.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im = axs[0, 1].imshow(op, origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
    axs[0, 1].set_title(f"{title_prefix}original POS shape\nN={int(np.nansum(op))}")
    plt.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im = axs[0, 2].imshow(on, origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
    axs[0, 2].set_title(f"{title_prefix}original NEG shape\nN={int(np.nansum(on))}")
    plt.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

    im = axs[1, 1].imshow(np2, origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
    axs[1, 1].set_title(f"{title_prefix}new POS shape (main logic)\nN={int(np.nansum(np2))}")
    plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im = axs[1, 2].imshow(nn2, origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
    axs[1, 2].set_title(f"{title_prefix}new NEG shape (main logic)\nN={int(np.nansum(nn2))}")
    plt.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

    axs[1, 0].axis("off")

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    plt.show()

def _get2(x, k=None):
    """
    Helper: if x is 3D (nE,ny,nx) return slice k, else return x as-is.
    """
    x = np.asarray(x)
    if x.ndim == 3:
        if k is None:
            raise ValueError("Need k for 3D array")
        return x[k]
    return x

def plot_bubbles_diagnostics(out, *, Ectr_gev=None, k=None, roi2d=None, srcmask2d=None, title_prefix="", lat=None, lon=None):
    # Decide k
    if k is None:
        if Ectr_gev is None:
            raise ValueError("Provide either k or Ectr_gev")
        k = int(np.argmin(np.abs(np.asarray(Ectr_gev) - 4.3)))

    flat_mask = out.flat_mask.astype(float)  # (ny,nx)

    bubble_flux_smooth_2d = _get2(out.bubble_image_flux_smooth, k=k)  # (ny,nx)

    pos2d = _get2(out.pos_template, k=k)  # could be 2D or 3D depending on how you built it
    neg2d = _get2(out.neg_template, k=k)

    # Optional combined mask for display (makes things clearer)
    show_mask = None
    if roi2d is not None and srcmask2d is not None:
        show_mask = (roi2d & srcmask2d)
    elif roi2d is not None:
        show_mask = roi2d
    elif srcmask2d is not None:
        show_mask = srcmask2d

    def apply_nan(a):
        a = a.astype(float).copy()
        if show_mask is not None:
            a[~show_mask] = np.nan
        return a

    A_flat = apply_nan(flat_mask)
    A_flux = apply_nan(bubble_flux_smooth_2d)
    A_pos  = apply_nan(pos2d)
    A_neg  = apply_nan(neg2d)

    display_mask = np.ones_like(lat, dtype=bool)
    display_mask &= (np.abs(lat) >= 10.0)          # mask disk
    display_mask &= srcmask2d                             # mask extended sources
    if roi2d is not None:
        display_mask &= roi2d

    def apply_display_mask(A):
        B = np.array(A, dtype=float, copy=True)
        B[~display_mask] = np.nan
        return B

    A_flat = apply_display_mask(A_flat)
    A_flux = apply_display_mask(A_flux)
    A_pos  = apply_display_mask(A_pos)
    A_neg  = apply_display_mask(A_neg)

    # For flux plot: use symmetric limits around 0 so you see pos/neg structure
    finite = np.isfinite(A_flux)
    vmax = np.nanpercentile(np.abs(A_flux[finite]), 99) if finite.any() else 1.0
    vmax = float(vmax) if vmax > 0 else 1.0

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    im0 = axs[0, 0].imshow(A_flat, origin="lower")
    axs[0, 0].set_title(f"{title_prefix}Flat bubbles mask")
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(A_flux, origin="lower", vmin=-vmax, vmax=vmax)
    Etxt = f"{np.asarray(Ectr_gev)[k]:.3g} GeV" if Ectr_gev is not None else f"k={k}"
    axs[0, 1].set_title(f"{title_prefix}Bubble-image flux (smoothed), {Etxt}\n(flat-fit + residual) / (expo·Ω·ΔE)")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[1, 0].imshow(A_pos, origin="lower")
    axs[1, 0].set_title(f"{title_prefix}Positive template (from smoothed flux)")
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im3 = axs[1, 1].imshow(A_neg, origin="lower")
    axs[1, 1].set_title(f"{title_prefix}Negative template (from smoothed flux)")
    plt.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# ----------------------------
# Utilities
# ----------------------------

def solid_angle_per_pixel(ny: int, nx: int, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """
    Returns omega (sr) per pixel for a CAR-like lat/lon grid.
    If you already have omega map, skip this and pass yours in.
    """
    # Assume lon_deg, lat_deg are 2D maps (ny,nx) giving pixel centers.
    # Approx: omega ≈ dlon * dlat * cos(lat) in radians.
    # Estimate dlon/dlat from neighbors (works for regular grids).
    dlon = np.deg2rad(np.nanmedian(np.abs(np.diff(lon_deg, axis=1))))
    dlat = np.deg2rad(np.nanmedian(np.abs(np.diff(lat_deg, axis=0))))
    omega = dlon * dlat * np.cos(np.deg2rad(lat_deg))
    return omega


def gaussian_sigma_pix(sigma_deg: float, pixsize_deg: float) -> float:
    return float(sigma_deg / pixsize_deg)

def counts_to_flux(
    counts: np.ndarray,                 # (ny,nx) or (nE,ny,nx)
    expo: np.ndarray,                   # (ny,nx) or (nE,ny,nx)
    omega_2d: np.ndarray,               # (ny,nx)
    dE_mev: Union[float, np.ndarray],   # scalar or (nE,)
) -> np.ndarray:
    """
    Convert counts -> differential flux (per MeV per cm^2 per s per sr):
        flux = counts / (expo * omega * dE)

    Supports:
      - 2D counts/expo with scalar dE
      - 3D counts/expo (nE,ny,nx) with scalar dE or 1D dE (nE,)
      - mixed 2D expo with 3D counts is NOT supported (make expo 3D first)
    """
    counts = np.asarray(counts, dtype=float)
    expo   = np.asarray(expo, dtype=float)
    omega  = np.asarray(omega_2d, dtype=float)

    if omega.ndim != 2:
        raise ValueError(f"omega_2d must be 2D (ny,nx); got {omega.shape}")

    dE = np.asarray(dE_mev, dtype=float)

    # ---- 2D case ----
    if counts.ndim == 2:
        if expo.ndim != 2:
            raise ValueError(f"2D counts requires 2D expo; got expo {expo.shape}")
        if dE.ndim != 0:
            raise ValueError("2D counts requires scalar dE_mev")
        denom = expo * omega * float(dE)
        flux = np.full_like(counts, np.nan, dtype=float)
        good = np.isfinite(counts) & np.isfinite(denom) & (denom > 0)
        flux[good] = counts[good] / denom[good]
        return flux

    # ---- 3D case ----
    if counts.ndim != 3:
        raise ValueError(f"counts must be 2D or 3D; got {counts.shape}")
    if expo.ndim != 3:
        raise ValueError(f"3D counts requires 3D expo; got expo {expo.shape}")

    nE, ny, nx = counts.shape
    if expo.shape != (nE, ny, nx):
        raise ValueError(f"expo shape {expo.shape} must match counts shape {counts.shape}")
    if omega.shape != (ny, nx):
        raise ValueError(f"omega_2d shape {omega.shape} must be (ny,nx)={(ny,nx)}")

    if dE.ndim == 0:
        dE3 = float(dE)
        denom = expo * omega[None, :, :] * dE3
    elif dE.ndim == 1:
        if dE.shape[0] != nE:
            raise ValueError(f"dE_mev length {dE.shape[0]} != nE {nE}")
        denom = expo * omega[None, :, :] * dE[:, None, None]
    else:
        raise ValueError(f"dE_mev must be scalar or 1D (nE,), got {dE.shape}")

    flux = np.full_like(counts, np.nan, dtype=float)
    good = np.isfinite(counts) & np.isfinite(denom) & (denom > 0)
    flux[good] = counts[good] / denom[good]
    return flux

import numpy as np

def audit_templates_at_k(*, k, counts, templates_counts, fit_mask_2d, name=""):
    """
    counts: (nE,ny,nx)
    templates_counts: dict name -> (nE,ny,nx) or (ny,nx)
    fit_mask_2d: (ny,nx) boolean region used in fit
    """
    nE, ny, nx = counts.shape
    m = fit_mask_2d & np.isfinite(counts[k])
    ysum = float(np.nansum(counts[k][m]))
    print(f"\n=== Template audit at k={k} ({name}) ===")
    print("data sum (mask):", ysum, "masked pix:", int(m.sum()))

    for tname, T in templates_counts.items():
        T = np.asarray(T)
        Tk = T[k] if T.ndim == 3 else T
        if Tk.shape != (ny, nx):
            print(f"  {tname:10s} BAD SHAPE {Tk.shape}")
            continue

        good = m & np.isfinite(Tk)
        s = float(np.nansum(Tk[good]))
        mn = float(np.nanmin(Tk[good])) if good.any() else np.nan
        mx = float(np.nanmax(Tk[good])) if good.any() else np.nan
        negpix = int(np.count_nonzero((Tk[good] < 0)))
        zpix = int(np.count_nonzero((Tk[good] == 0)))

        ratio = s / ysum if ysum > 0 else np.nan
        print(f"  {tname:10s} sum={s: .3e}  ratio_to_data={ratio: .3g}  min/max=({mn: .3e},{mx: .3e})  negpix={negpix}  zeropix={zpix}")

def _as_cube(x):
    """Ensure output is (nE,ny,nx) for write_cube."""
    x = np.asarray(x)
    if x.ndim == 2:
        return x[None, :, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")

def safe_percentile(arr, q, default=np.nan):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.nanpercentile(arr, q))

# ----------------------------
# Flat bubbles template (initial hard-edge)
# ----------------------------

def make_flat_bubbles_mask(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    # a simple default "hourglass-ish" cut; you should iterate this boundary
    lat_min_deg: float = 10.0,
    lat_max_deg: float = 55.0,
    lon_max_deg: float = 20.0,
    taper_start_deg: float = 25.0,
) -> np.ndarray:
    """
    A *very* simple starter mask:
      - |b| between lat_min and lat_max
      - |l| < lon_max near mid-latitudes
      - a mild "taper" at high latitudes

    Replace this with your own polygon boundary once you have one.
    """
    L = lon_deg.copy()
    # wrap to [-180, 180] if needed
    L = (L + 180.0) % 360.0 - 180.0
    B = lat_deg

    absB = np.abs(B)
    absL = np.abs(L)

    core = (absB >= lat_min_deg) & (absB <= lat_max_deg)

    # taper: allow a bit wider at some latitudes if desired (or narrower)
    # Here: shrink allowed |l| slightly above taper_start_deg
    lon_allow = np.full_like(B, lon_max_deg, dtype=float)
    hi = absB > taper_start_deg
    lon_allow[hi] = lon_max_deg * (1.0 - 0.4 * (absB[hi] - taper_start_deg) / max(1e-6, (lat_max_deg - taper_start_deg)))
    lon_allow = np.clip(lon_allow, 6.0, lon_max_deg)

    mask = core & (absL <= lon_allow)
    return mask


def flat_template_counts_from_mask(
    flat_mask_2d: np.ndarray,     # (ny,nx) bool/0-1
    expo: np.ndarray,             # (ny,nx) or (nE,ny,nx)
    omega_2d: np.ndarray,         # (ny,nx)
    dE_mev: Union[float, np.ndarray],  # scalar or (nE,)
) -> np.ndarray:
    """
    Turn a spatial flat mask into a counts template.

    We want a template that corresponds to *constant flux* inside the mask:
        mu_flat(E, x) ∝ expo(E,x) * omega(x) * dE(E) * mask(x)

    Supports:
      - expo shape (ny,nx): returns (ny,nx) if dE scalar, or (nE,ny,nx) if dE is (nE,)
      - expo shape (nE,ny,nx): returns (nE,ny,nx) and dE can be scalar or (nE,)
    """
    mask = flat_mask_2d.astype(float)
    omega = omega_2d.astype(float)

    dE = np.asarray(dE_mev, dtype=float)
    if dE.ndim == 0:
        # scalar bin width
        if expo.ndim == 2:
            return mask * expo * omega * float(dE)
        elif expo.ndim == 3:
            return (mask[None, :, :] * expo * omega[None, :, :] * float(dE))
        else:
            raise ValueError(f"expo must be 2D or 3D, got shape {expo.shape}")

    # dE is 1D: (nE,)
    if dE.ndim != 1:
        raise ValueError(f"dE_mev must be scalar or 1D (nE,), got shape {dE.shape}")

    if expo.ndim == 2:
        # broadcast to (nE,ny,nx)
        return (mask[None, :, :]
                * expo[None, :, :]
                * omega[None, :, :]
                * dE[:, None, None])

    if expo.ndim == 3:
        nE = expo.shape[0]
        if dE.shape[0] != nE:
            raise ValueError(f"dE_mev length {dE.shape[0]} != expo nE {nE}")
        return (mask[None, :, :]
                * expo
                * omega[None, :, :]
                * dE[:, None, None])

    raise ValueError(f"expo must be 2D or 3D, got shape {expo.shape}")

import numpy as np


# ----------------------------
# Main Totani-style builder
# ----------------------------

@dataclass
class BubbleTemplates:
    flat_mask: np.ndarray              # (ny,nx) bool
    flat_counts_template: np.ndarray   # (ny,nx) counts template for the *construction energy*
    bubble_image_counts: np.ndarray    # (ny,nx) flat-fit+resid in counts
    bubble_image_flux_smooth: np.ndarray  # (ny,nx) smoothed flux map
    pos_template: np.ndarray           # (ny,nx) spatial weights (>=0)
    neg_template: np.ndarray           # (ny,nx) spatial weights (>=0)


def build_bubbles_templates(
    *,
    # inputs at the chosen construction energy bin (e.g. 4.3 GeV)
    data_counts_2d: np.ndarray,     # counts map at E*
    expo_2d: np.ndarray,            # exposure map at E* (cm^2 s)
    omega_2d: np.ndarray,           # sr per pixel
    dE_mev: float,                  # MeV bin width at E*
    lon_deg: np.ndarray,            # lon map (ny,nx)
    lat_deg: np.ndarray,            # lat map (ny,nx)
    roi_mask_2d: np.ndarray,        # ROI keep-mask (ny,nx)
    srcmask_2d: np.ndarray,         # True where you KEEP (i.e. not masked by ext sources)
    other_templates_counts: Dict[str, np.ndarray],  # other components in counts at E* (gas/ics/iso/ps/loopI/...)
    pixsize_deg: float,             # e.g. 0.5 if 0.5 deg pixels
    smooth_sigma_deg: float = 1.0,
    flat_mask_kwargs: Optional[dict] = None,
    # whether pos/neg templates keep the smoothed flux amplitudes, or just become 0/1 masks
    keep_flux_amplitudes: bool = True,
    flat_mask: Optional[np.ndarray] = None,
) -> BubbleTemplates:
    """
    Returns flat + pos/neg templates using the Totani-style recipe.

    IMPORTANT:
      - other_templates_counts must be in COUNTS SPACE (same binning/psf etc as data_counts_2d)
      - srcmask_2d should exclude extended sources etc (Totani masks extended sources)
    """
    nE, ny, nx = data_counts_2d.shape
    flat_mask_kwargs = flat_mask_kwargs or {}

    # Totani-style: fit includes disk, etc. Your roi_mask_2d should include whatever region you want for this fit.
    fit_mask = roi_mask_2d & srcmask_2d

    # 1) initial flat mask boundary
    if flat_mask is None:
        flat_mask = make_flat_bubbles_mask(lon_deg, lat_deg, **flat_mask_kwargs)

    # 2) convert flat mask -> counts template at E*
    mu_flat = flat_template_counts_from_mask(flat_mask, expo_2d, omega_2d, dE_mev)

    # 3) fit: {other templates + flat bubbles}
    templates = dict(other_templates_counts)
    templates["fb_flat"] = mu_flat

    coeff, model_counts = fit_nnls_counts_3d(
        data_counts=data_counts_2d,
        templates_counts=templates,
        fit_mask_2d=fit_mask,
        mode="per_energy"
    )

    resid_counts = data_counts_2d - model_counts

    # 4) bubble image (counts) = best-fit flat component + residual
    c_flat = np.asarray(coeff["fb_flat"])
    if c_flat.ndim == 0:
        bubble_flat_counts = float(c_flat) * mu_flat
    elif c_flat.ndim == 1:
        bubble_flat_counts = c_flat[:, None, None] * mu_flat
    else:
        raise ValueError(f"Unexpected fb_flat coeff shape: {c_flat.shape}")

    bubble_image_counts = bubble_flat_counts + resid_counts

    # 5) convert to flux and smooth with sigma=1 deg
    bubble_image_flux = counts_to_flux(bubble_image_counts, expo_2d, omega_2d, dE_mev)
    sig_pix = gaussian_sigma_pix(smooth_sigma_deg, pixsize_deg)
    bubble_image_flux_smooth = gaussian_filter(
        np.nan_to_num(bubble_image_flux, nan=0.0),
        sigma=(0.0, sig_pix, sig_pix)
    )

    # 6) define positive/negative templates from smoothed flux map (apply same mask)
    use = fit_mask & flat_mask # you can choose roi_mask_2d only, but keeping srcmask is closer to their masking step
    k_construct = 2   # <-- set this to your 4.3 GeV index
    f2 = bubble_image_flux_smooth[k_construct].copy()
    f2[~use] = np.nan

    # pick top tail on each side independently
    pos_thr = np.nanpercentile(f2[(f2 > 0) & np.isfinite(f2)], 95)   # tune 90–99
    neg_thr =np.nanpercentile(f2[(f2 < 0) & np.isfinite(f2)], 5)    # bottom 5% (more negative)

    pos = (f2 > pos_thr).astype(float)
    neg = (f2 < neg_thr).astype(float)

    print("pos pixels:", int(np.nansum(pos)), "neg pixels:", int(np.nansum(neg)))
    print("pos_thr:", pos_thr, "neg_thr:", neg_thr)

    if not keep_flux_amplitudes:
        pos = (pos > 0).astype(float)
        neg = (neg > 0).astype(float)

    return BubbleTemplates(
        flat_mask=flat_mask,
        flat_counts_template=mu_flat,
        bubble_image_counts=bubble_image_counts,
        bubble_image_flux_smooth=bubble_image_flux_smooth,
        pos_template=pos,
        neg_template=neg,
    )


REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
# ----------------------------
# Example wiring (replace with your own arrays)
# ----------------------------
if __name__ == "__main__":
    # You MUST replace everything below with your project arrays.
    # Shapes: all (ny,nx). Units: counts, cm^2 s, sr, MeV.

    ap = argparse.ArgumentParser()
    default_counts = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
    default_expo = os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits")
    default_bubble_mask = os.path.join(DATA_DIR, "processed", "templates", "bubbles_flat_binary_mask.fits")
    default_srcmask = os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits")
    default_outdir = os.path.join(DATA_DIR, "processed", "templates")
    templates_dir = os.path.join(DATA_DIR, "processed", "templates")

    ap.add_argument("--counts", default=default_counts, help="Counts CCUBE (authoritative WCS + EBOUNDS)")
    ap.add_argument("--expo", default=default_expo, help="Exposure cube (expcube)")
    ap.add_argument("--bubble-mask", default=default_bubble_mask, help="FITS mask defining FB rough area (0/1 map)")
    ap.add_argument("--ext_mask", default=default_srcmask, help="FITS mask defining extended sources (0/1 map)")
    ap.add_argument("--outdir", default=default_outdir)

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Read CCUBE header + EBOUNDS (authoritative energy binning) ---
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    # --- Read exposure cube + energies, resample to CCUBE binning ---
    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure spatial shape {expo_raw.shape[1:]} != counts {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)

    # --- Restrict to pixels with actual data coverage ---
    expo_safe = np.array(expo, dtype=float, copy=True)
    expo_safe[~np.isfinite(expo_safe) | (expo_safe <= 0)] = np.nan
    data_ok3d = np.isfinite(counts) & np.isfinite(expo_safe)
    data_ok2d = np.any(data_ok3d, axis=0)

    counts_cov = np.array(counts, dtype=float, copy=True)
    counts_cov[~data_ok3d] = np.nan

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    print("omega median:", np.nanmedian(omega))


    # Lon/lat and ROI box
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi_mask_2d = (np.abs(lon) <= 60) & (np.abs(lat) <= 60)
    ext_mask3d = load_mask_any_shape(args.ext_mask, counts_cov.shape)
    ext_mask2d = np.any(ext_mask3d, axis=0)   # (ny,nx) True if masked in any E

    # Other templates (counts space!) at this energy:
    # Load templates
    labels = ["gas", "iso", "ps", "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno", "loopI", "ics"]

    mu_list, headers = load_mu_templates_from_fits(
        template_dir=templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",  
        hdu=0,
    )

    other_templates_counts = {}
    for i in range(len(mu_list)):
        other_templates_counts[labels[i]] = mu_list[i]

    if not os.path.exists(str(args.bubble_mask)):
        verts_n_path = os.path.join(os.path.dirname(__file__), "bubble_vertices_north.txt")
        verts_s_path = os.path.join(os.path.dirname(__file__), "bubble_vertices_south.txt")
        if (not os.path.exists(verts_n_path)) or (not os.path.exists(verts_s_path)):
            raise FileNotFoundError(
                f"Bubble mask not found: {args.bubble_mask}\n"
                f"Also missing vertex files: {verts_n_path} and/or {verts_s_path}"
            )

        verts_n = _read_vertices_txt(verts_n_path)
        verts_s = _read_vertices_txt(verts_s_path)
        mask_n = _poly_mask_lonlat(lon2d=lon, lat2d=lat, verts_lonlat=verts_n)
        mask_s = _poly_mask_lonlat(lon2d=lon, lat2d=lat, verts_lonlat=verts_s)
        fb_mask_2d_tmp = (mask_n | mask_s)
        fb_mask_2d_tmp = _cleanup_binary_mask(fb_mask_2d_tmp, r_open_pix=1, r_close_pix=2)

        os.makedirs(os.path.dirname(str(args.bubble_mask)), exist_ok=True)
        fits.PrimaryHDU(fb_mask_2d_tmp.astype(np.int16), header=hdr).writeto(str(args.bubble_mask), overwrite=True)
        print(f"✓ wrote fallback bubble mask: {args.bubble_mask}")

    fb_mask_2d = fits.getdata(args.bubble_mask).astype(bool)
    if fb_mask_2d.ndim == 3:
        fb_mask_2d = fb_mask_2d[0].astype(bool)
    if fb_mask_2d.shape != (ny, nx):
        raise SystemExit(f"Bubbles mask shape {fb_mask_2d.shape} != {(ny, nx)}")

    # --- choose construct bin (you can keep k=2 for now) ---
    k_construct = 2

    # --- fit mask (must be KEEP mask for ext sources) ---
    fit_mask = roi_mask_2d & ext_mask2d & data_ok2d

    # --- NORMALISE the other templates in counts space ---
    other_templates_norm, other_norms = normalise_templates_counts(other_templates_counts, fit_mask)


    # --- build bubbles using the NORMALISED other templates ---
    out = build_bubbles_templates(
        data_counts_2d=counts_cov,
        expo_2d=expo_safe,
        omega_2d=omega,
        dE_mev=dE_mev,
        lon_deg=lon,
        lat_deg=lat,
        roi_mask_2d=roi_mask_2d,
        srcmask_2d=ext_mask2d,
        other_templates_counts=other_templates_norm,   # <-- IMPORTANT
        pixsize_deg=args.binsz,
        smooth_sigma_deg=1.0,
        flat_mask_kwargs=dict(lat_min_deg=10.0, lat_max_deg=55.0, lon_max_deg=20.0, taper_start_deg=25.0),
        keep_flux_amplitudes=True,
        flat_mask=fb_mask_2d,
    )

    print("flat_mask:", out.flat_mask.shape, out.flat_mask.sum())
    print("pos_template sum:", out.pos_template.sum(), "neg_template sum:", out.neg_template.sum())

    # --- build fb_flat counts cube from out ---
    mu_flat = np.asarray(out.flat_counts_template, float)  # (nE,ny,nx)

    # --- combine templates ---
    templates_all = dict(other_templates_counts)   # UNnormalised originals (counts)
    templates_all["fb_flat"] = mu_flat

    # --- normalise EVERYTHING together ---
    templates_all_norm, norms_all = normalise_templates_counts(templates_all, fit_mask)

    print("\nSums in fit mask at k_construct (before norm):")
    for n, s in norms_all.items():
        print(f"  {n:10s} sum={s[k_construct]:.3e}")

    fit_mask3d = build_fit_mask3d(
        roi2d=roi_mask_2d,
        srcmask3d=ext_mask3d,
        counts=counts_cov,
        expo=expo_safe,
        extra2d=None,
    )
    res_fit = fit_cellwise_poisson_mle_counts(
        counts=counts_cov,
        templates=templates_all,
        mask3d=fit_mask3d,
        lon=lon,
        lat=lat,
        roi_lon=float(args.roi_lon),
        roi_lat=float(args.roi_lat),
        cell_deg=float(CELL_DEG),
        nonneg=True,
        column_scale="l2",
        drop_tol=0.0,
        ridge=0.0,
    )
    comp_counts, model_counts = component_counts_from_cellwise_fit(
        templates_counts=templates_all,
        res_fit=res_fit,
        mask3d=fit_mask3d,
    )
    resid_counts = counts_cov - model_counts
    bubble_flat_counts = np.asarray(comp_counts["fb_flat"], float)
    bubble_image_counts = bubble_flat_counts + resid_counts

    # --- convert bubble_image_counts -> flux and smooth (spatial only) ---
    bubble_image_flux = counts_to_flux(
        bubble_image_counts,   # (nE,ny,nx)
        expo_safe,             # (nE,ny,nx)
        omega,                 # (ny,nx)
        dE_mev,                # (nE,)
    )
 
    # --- pick construct bin near 4.3 GeV ---
    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0
    k_construct = int(np.argmin(np.abs(Ectr_gev - 4.3)))
    print("k_construct:", k_construct, "E:", Ectr_gev[k_construct], "GeV")

    # --- smooth (spatial only) ---
    sig_pix = gaussian_sigma_pix(2.0, args.binsz)  # 2 deg for construction
    bubble_image_flux_smooth = gaussian_filter(
        np.nan_to_num(bubble_image_flux, nan=0.0),
        sigma=(0.0, sig_pix, sig_pix),
    )

    # --- base construction region (envelope-restricted) ---
    use_construct = (fit_mask & fb_mask_2d & data_ok2d).astype(bool)

    # --- threshold on refit-derived smooth map ---
    f2 = np.asarray(bubble_image_flux_smooth[k_construct], float).copy()
    f2[~use_construct] = np.nan

    finite_use = use_construct & np.isfinite(f2)
    pos_vals = f2[finite_use & (f2 > 0)]
    neg_vals = f2[finite_use & (f2 < 0)]

    pos_thr = 0 # safe_percentile(pos_vals, 80, default=np.nan)
    neg_thr = 0 # safe_percentile(neg_vals, 40, default=np.nan)

    # if np.any(finite_use):
    #     fmin = float(np.nanmin(f2[finite_use]))
    #     fmax = float(np.nanmax(f2[finite_use]))
    # else:
    #     fmin, fmax = np.nan, np.nan

    # print(
    #     "Within use_construct: n_pos=", int(pos_vals.size),
    #     "n_neg=", int(neg_vals.size),
    #     "min/max=", fmin, fmax,
    # )

    # -------------------------
    # Convert threshold -> raw boolean masks
    # -------------------------
    pos_raw = np.zeros_like(f2, dtype=bool)
    neg_raw = np.zeros_like(f2, dtype=bool)

    if np.isfinite(pos_thr):
        pos_raw = (f2 > pos_thr) & use_construct & np.isfinite(f2)
    else:
        print("WARNING: no positive pixels; pos_raw empty.")

    if np.isfinite(neg_thr):
        neg_raw = (f2 < neg_thr) & use_construct & np.isfinite(f2)
    else:
        print("WARNING: no negative pixels; neg_raw empty.")

    # -------------------------
    # Optional cleanup (same morphology radii you already use)
    # -------------------------
    pos_clean = _cleanup_binary_mask(
        pos_raw,
        r_open_pix=(MORPH_R_OPEN_DEG / args.binsz),
        r_close_pix=(MORPH_R_CLOSE_DEG / args.binsz),
    )
    neg_clean = _cleanup_binary_mask(
        neg_raw,
        r_open_pix=(MORPH_R_OPEN_DEG / args.binsz),
        r_close_pix=(MORPH_R_CLOSE_DEG / args.binsz),
    )

    # -------------------------
    # Keep largest CC in North and South separately for pos and neg
    # -------------------------
    def keep_largest_cc(mask_bool):
        lbl, nlab = label(mask_bool)
        if nlab <= 0:
            return np.zeros_like(mask_bool, dtype=bool)
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        keep = int(np.argmax(sizes))
        return (lbl == keep)

    pos_n = keep_largest_cc(pos_clean & (lat > 0))
    pos_s = keep_largest_cc(pos_clean & (lat < 0))
    neg_n = keep_largest_cc(neg_clean & (lat > 0))
    neg_s = keep_largest_cc(neg_clean & (lat < 0))

    # Recombine into final float masks for downstream
    pos2d = (pos_n | pos_s).astype(float)

    neg_keep = (neg_n | neg_s)
    if (not np.any(neg_keep)) and np.any(neg_clean):
        neg_keep = neg_clean
    if (not np.any(neg_keep)) and np.any(neg_raw):
        neg_keep = neg_raw
    neg2d = neg_keep.astype(float)

    orig_pos2d = (np.asarray(out.pos_template) > 0).astype(float)
    orig_neg2d = (np.asarray(out.neg_template) > 0).astype(float)

    plot_fb_shape_comparison(
        bubble_flux_smooth_2d=bubble_image_flux_smooth[k_construct],
        mask_2d=use_construct,
        orig_pos_2d=orig_pos2d,
        orig_neg_2d=orig_neg2d,
        new_pos_2d=pos2d,
        new_neg_2d=neg2d,
        title_prefix="Totani FB shapes: ",
        outpath=os.path.join(args.outdir, "fb_shape_comparison.png"),
    )

    print("pos pixels:", int(pos2d.sum()), "neg pixels:", int(neg2d.sum()))
    print(
        "pos_thr:", float(pos_thr) if np.isfinite(pos_thr) else pos_thr,
        "neg_thr:", float(neg_thr) if np.isfinite(neg_thr) else neg_thr,
    )

    # ---------- convert pos/neg masks to counts templates ----------
    nE, ny, nx = expo.shape
    assert omega.shape == (ny, nx)

    dE = np.asarray(dE_mev, float)
    if dE.ndim == 0:
        dE = np.full((nE,), float(dE))
    assert dE.shape == (nE,)

    pos3 = pos2d[None, :, :]   # broadcasts to (nE,ny,nx)
    neg3 = neg2d[None, :, :]

    mu_pos = pos3 * expo * omega[None, :, :] * dE[:, None, None]   # counts
    mu_neg = neg3 * expo * omega[None, :, :] * dE[:, None, None]   # counts

    # ---------- normalise per energy in fit region for stable NNLS ----------
    use2d = roi_mask_2d & ext_mask2d  # must be KEEP mask

    mu_pos_norm = mu_pos.astype(np.float64, copy=True)
    mu_neg_norm = mu_neg.astype(np.float64, copy=True)

    pos_norm = np.zeros(nE, dtype=float)
    neg_norm = np.zeros(nE, dtype=float)

    for kk in range(nE):
        pos_norm[kk] = float(np.nansum(mu_pos_norm[kk][use2d]))
        neg_norm[kk] = float(np.nansum(mu_neg_norm[kk][use2d]))

        if pos_norm[kk] > 0:
            mu_pos_norm[kk] /= pos_norm[kk]
        if neg_norm[kk] > 0:
            mu_neg_norm[kk] /= neg_norm[kk]

    print("pos_norm median (counts):", float(np.median(pos_norm[pos_norm > 0])) if np.any(pos_norm > 0) else 0.0)
    print("neg_norm median (counts):", float(np.median(neg_norm[neg_norm > 0])) if np.any(neg_norm > 0) else 0.0)

    # ---------- write outputs ----------
    out_flatmask = os.path.join(args.outdir, "fb_flat_mask.fits")
    out_muflat   = os.path.join(args.outdir, "mu_fb_flat_counts.fits")
    out_pos      = os.path.join(args.outdir, "mu_fb_pos_counts.fits")
    out_neg      = os.path.join(args.outdir, "mu_fb_neg_counts.fits")
    out_posn     = os.path.join(args.outdir, "mu_fb_pos_norm_counts.fits")
    out_negn     = os.path.join(args.outdir, "mu_fb_neg_norm_counts.fits")

    # flat mask and mu_flat from builder output
    write_cube(out_flatmask, _as_cube(out.flat_mask.astype(np.float32)), hdr, bunit="1")
    write_cube(out_muflat,   _as_cube(out.flat_counts_template.astype(np.float32)), hdr, bunit="counts")

    # pos/neg counts templates from refit-derived masks
    write_cube(out_pos,  mu_pos.astype(np.float32), hdr, bunit="counts")
    write_cube(out_neg,  mu_neg.astype(np.float32), hdr, bunit="counts")
    write_cube(out_posn, mu_pos_norm.astype(np.float32), hdr, bunit="counts")
    write_cube(out_negn, mu_neg_norm.astype(np.float32), hdr, bunit="counts")

    print("✓ Wrote", out_flatmask)
    print("✓ Wrote", out_muflat)
    print("✓ Wrote", out_pos)
    print("✓ Wrote", out_neg)
    print("✓ Wrote", out_posn)
    print("✓ Wrote", out_negn)

    # ---------- plotting ----------
    # IMPORTANT: plot_bubbles_diagnostics(out, ...) still uses out.bubble_image_flux_smooth,
    # which came from the builder (pre-refit). So either:
    #   (a) skip it, OR
    #   (b) plot using plot_fb_summary which uses mu_pos/mu_neg and your maps.
    #
    # If you want to keep plot_fb_summary, call it with the new masks/templates:

    plot_fb_summary(
        out=out,                 # for flat_mask display only
        counts=counts,
        mu_pos=mu_pos,
        mu_neg=mu_neg,
        Ectr_gev=Ectr_mev/1000,
        k=k_construct,
        roi2d=roi_mask_2d,
        srckeep2d=ext_mask2d,
        lat_deg_2d=lat,
        mask_disk_deg=10.0,
        title_prefix="Totani FB (refit pos/neg): ",
    )