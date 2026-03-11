#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize, nnls

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (  # noqa: E402
    lonlat_grids,
    load_mask_any_shape,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)

from totani_helpers.fit_utils import build_fit_mask3d
from totani_helpers.mcmc_io import combine_loopI, load_mcmc_coeffs_by_label

REPO_DIR_DEFAULT = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR_DEFAULT, "fermi_data", "totani")


def counts_to_flux(counts_map, expo_map, omega_sr_map, dE_mev):
    denom = expo_map * omega_sr_map * float(dE_mev)
    out = np.full_like(counts_map, np.nan, dtype=float)
    m = np.isfinite(denom) & (denom > 0)
    out[m] = counts_map[m] / denom[m]
    return out


def counts_to_flux_from_denom(counts_map, denom_map):
    out = np.full_like(counts_map, np.nan, dtype=float)
    denom_map = np.asarray(denom_map, dtype=float)
    m = np.isfinite(denom_map) & (denom_map > 0)
    out[m] = np.asarray(counts_map, dtype=float)[m] / denom_map[m]
    return out


def read_coeff_counts(coeff_file, component_name):
    Ectr_gev = []
    coeffs = []
    with open(coeff_file, "r") as f:
        header = f.readline().strip()
        if header.startswith("#"):
            header = header.lstrip("#").strip()
        cols = header.split()
        if len(cols) < 3:
            raise ValueError(f"Malformed coefficient header: '{header}'")

        def _norm(s):
            return "".join(ch for ch in str(s).lower() if ch.isalnum())

        req = _norm(component_name)
        ncols = [_norm(c) for c in cols]
        if req in ncols:
            comp_col = ncols.index(req)
        else:
            hits = [i for i, nc in enumerate(ncols) if req in nc]
            if len(hits) == 1:
                comp_col = hits[0]
            else:
                raise ValueError(
                    f"Component '{component_name}' not found in coeff file. Available: {cols[2:]}"
                )

        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= comp_col:
                continue
            Ectr_gev.append(float(parts[1]))
            coeffs.append(float(parts[comp_col]))

    return np.asarray(Ectr_gev, float), np.asarray(coeffs, float)


def counts_coeff_to_map(mu_k, coeff_counts_k, mask2d):
    mu_k = np.asarray(mu_k, float)
    m = mask2d & np.isfinite(mu_k)
    s = float(np.nansum(mu_k[m]))
    out = np.zeros_like(mu_k, dtype=float)
    if not np.isfinite(s) or s <= 0:
        return out
    out[m] = float(coeff_counts_k) * (mu_k[m] / s)
    return out


def cellwise_coeffs_to_total_counts(*, cells, coeff_cells, mu_list, mask3d):
    """Compute per-bin total counts attributed to each component from cellwise multipliers."""
    nCells, nE, nComp = coeff_cells.shape
    out = np.zeros((nE, nComp), dtype=float)
    for k in range(nE):
        for ci, cell2d in enumerate(cells):
            cm = mask3d[k] & cell2d
            if not np.any(cm):
                continue
            a = coeff_cells[ci, k, :]
            if not np.all(np.isfinite(a)):
                continue
            for j in range(nComp):
                s = float(np.nansum(mu_list[j][k][cm]))
                if np.isfinite(s) and s != 0.0:
                    out[k, j] += float(a[j]) * s
    return out


def cellwise_coeffs_to_model_map_k(*, k, cells, coeff_cells, mu_list, mask2d):
    """Build a per-pixel model counts map at one energy bin k from cellwise multipliers."""
    ny, nx = mu_list[0].shape[1:]
    model = np.zeros((ny, nx), dtype=float)
    for ci, cell2d in enumerate(cells):
        cm = mask2d & cell2d
        if not np.any(cm):
            continue
        a = coeff_cells[ci, k, :]
        if not np.all(np.isfinite(a)):
            continue
        for aj, mu in zip(a, mu_list):
            if aj != 0.0:
                model[cm] += float(aj) * mu[k][cm]
    return model


def _read_cube(path, expected_shape=None, dtype=float):
    with fits.open(path) as h:
        d = np.array(h[0].data, dtype=dtype)
        hdr = h[0].header
    if expected_shape is not None and d.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {d.shape}, expected {expected_shape}")
    return d, hdr


def smooth_map_deg(arr2d, sigma_deg, binsz_deg):
    if sigma_deg is None or sigma_deg <= 0:
        return arr2d
    sigma_pix = float(sigma_deg) / float(binsz_deg)
    # NaN-safe smoothing: smooth numerator & weight
    w = np.isfinite(arr2d).astype(float)
    a0 = np.nan_to_num(arr2d, nan=0.0)
    num = gaussian_filter(a0, sigma=sigma_pix, mode="constant", cval=0.0)
    den = gaussian_filter(w,  sigma=sigma_pix, mode="constant", cval=0.0)
    out = np.full_like(arr2d, np.nan, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def make_totani_fig11_like_with_extmask(
    data_counts,                  # 2D counts at energy bin k
    model_except_halo_nohalo_counts,  # 2D counts (all fitted comps except halo) from NO-HALO fit
    model_except_halo_halo_counts,    # 2D counts (all fitted comps except halo) from WITH-HALO fit
    halo_counts,                  # 2D counts for halo (template * best-fit coeff)
    expo_map,                     # 2D exposure [cm^2 s]
    omega_sr_map,                 # 2D solid angle per pixel [sr]
    dE_mev,                       # scalar energy-bin width [MeV]
    extmask2d=None,               # 2D bool mask: True = masked (greyed)
    wcs=None,
    disk_band_2d=None,
    sigma_smooth_deg=1.0,
    binsz_deg=1.0,
    vlim=2e-12,
    out_png="totani_fig11_like.png",
    energy_label="21 GeV",
):
    """
    2x2 Totani Fig.11-style plot at one energy bin.

    Panels are in flux units [cm^-2 s^-1 sr^-1 MeV^-1], smoothed with sigma=1 deg:
      TL: data - model_except_halo                      (no-halo fit residual)
      TR: data - model_except_halo                      (halo model + residual = observed minus all-but-halo)
      BL: halo                                          (halo model)
      BR: data - (model_except_halo + halo)             (fit residual with halo)

    Notes:
      - TR equals TL *if* model_except_halo is the same object used for both fits.
        If you have two separate fits (one w/out halo, one w/ halo) you should pass
        the appropriate model_except_halo for each case.
      - extmask2d True pixels are shown in grey (like the grey circles/regions).
    """
    # counts-space maps
    resid_nohalo_counts = data_counts - model_except_halo_nohalo_counts
    resid_withhalo_counts = data_counts - (model_except_halo_halo_counts + halo_counts)
    halo_plus_resid_counts = data_counts - model_except_halo_halo_counts  # = halo + resid_withhalo

    # convert to flux
    tl = counts_to_flux(resid_nohalo_counts, expo_map, omega_sr_map, dE_mev)
    tr = counts_to_flux(halo_plus_resid_counts, expo_map, omega_sr_map, dE_mev)
    bl = counts_to_flux(halo_counts, expo_map, omega_sr_map, dE_mev)
    br = counts_to_flux(resid_withhalo_counts, expo_map, omega_sr_map, dE_mev)

    # smooth
    tl = smooth_map_deg(tl, sigma_smooth_deg, binsz_deg)
    tr = smooth_map_deg(tr, sigma_smooth_deg, binsz_deg)
    bl = smooth_map_deg(bl, sigma_smooth_deg, binsz_deg)
    br = smooth_map_deg(br, sigma_smooth_deg, binsz_deg)

    # apply extended-source mask as NaN so it renders grey
    if extmask2d is not None:
        extmask2d = np.asarray(extmask2d, dtype=bool)
        if extmask2d.shape != tl.shape:
            raise ValueError(f"extmask2d shape {extmask2d.shape} must match maps {tl.shape}")
        for arr in (tl, tr, bl, br):
            arr[extmask2d] = np.nan

    # plotting setup
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    cmap = LinearSegmentedColormap.from_list(
        "cyan_blue_black_orange_yellow",
        ["cyan", "blue", "black", "orangered", "yellow"],
        N=256,
    ).copy()
    cmap.set_bad(color="0.6")  # masked regions grey

    disk_overlay = None
    if disk_band_2d is not None:
        disk_band_2d = np.asarray(disk_band_2d, dtype=bool)
        if disk_band_2d.shape != tl.shape:
            raise ValueError(f"disk_band_2d shape {disk_band_2d.shape} must match maps {tl.shape}")
        disk_overlay = np.where(disk_band_2d, 1.0, np.nan)

    def draw_panel(ax, img, text):
        im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        if disk_overlay is not None:
            ax.imshow(
                disk_overlay,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                alpha=0.85,
                interpolation="nearest",
            )

        ax.text(
            0.03, 0.93, text,
            transform=ax.transAxes,
            color="white", fontsize=10,
            ha="left", va="top",
        )

        ax.set_xlabel("Galactic longitude")
        ax.set_ylabel("Galactic latitude")

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06, location="top")
        cbar.set_label(r"flux  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
        return im

    if wcs is None:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(10.5, 9.0), constrained_layout=True)
        axes = np.array(
            [
                [fig.add_subplot(2, 2, 1, projection=wcs), fig.add_subplot(2, 2, 2, projection=wcs)],
                [fig.add_subplot(2, 2, 3, projection=wcs), fig.add_subplot(2, 2, 4, projection=wcs)],
            ]
        )

    draw_panel(axes[0, 0], tl, rf"{energy_label}, no-halo fit residual")
    draw_panel(axes[0, 1], tr, rf"{energy_label}, NFW-$\rho^2$ model+residual")
    draw_panel(axes[1, 0], bl, rf"{energy_label}, NFW-$\rho^2$ model")
    draw_panel(axes[1, 1], br, rf"{energy_label}, NFW-$\rho^2$ fit residual")

    fig.savefig(out_png, dpi=250)
    plt.close(fig)
    print("✓ wrote", out_png)


def make_totani_fig11_like_with_extmask_stacked(
    data_counts,
    model_except_halo_nohalo_counts,
    model_except_halo_halo_counts,
    halo_counts,
    denom_map,
    extmask2d=None,
    wcs=None,
    disk_band_2d=None,
    sigma_smooth_deg=1.0,
    binsz_deg=1.0,
    vlim=2e-12,
    out_png="totani_fig11_like_stacked.png",
    energy_label="stack",
):
    resid_nohalo_counts = data_counts - model_except_halo_nohalo_counts
    resid_withhalo_counts = data_counts - (model_except_halo_halo_counts + halo_counts)
    halo_plus_resid_counts = data_counts - model_except_halo_halo_counts

    tl = counts_to_flux_from_denom(resid_nohalo_counts, denom_map)
    tr = counts_to_flux_from_denom(halo_plus_resid_counts, denom_map)
    bl = counts_to_flux_from_denom(halo_counts, denom_map)
    br = counts_to_flux_from_denom(resid_withhalo_counts, denom_map)

    tl = smooth_map_deg(tl, sigma_smooth_deg, binsz_deg)
    tr = smooth_map_deg(tr, sigma_smooth_deg, binsz_deg)
    bl = smooth_map_deg(bl, sigma_smooth_deg, binsz_deg)
    br = smooth_map_deg(br, sigma_smooth_deg, binsz_deg)

    if extmask2d is not None:
        extmask2d = np.asarray(extmask2d, dtype=bool)
        for arr in (tl, tr, bl, br):
            arr[extmask2d] = np.nan

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    cmap = LinearSegmentedColormap.from_list(
        "cyan_blue_black_orange_yellow",
        ["cyan", "blue", "black", "orangered", "yellow"],
        N=256,
    ).copy()
    cmap.set_bad(color="0.6")


def _build_nonhalo_model_map_k(
    *,
    k: int,
    coeffs_by_label: dict,
    templates_dir: str,
    nE: int,
    ny: int,
    nx: int,
):
    components = {
        "ics": ["mu_ics_counts.fits"],
        "iso": ["mu_iso_counts.fits"],
        "gas": ["mu_gas_counts.fits"],
        "ps": ["mu_ps_counts.fits"],
    }

    if "loopI" in coeffs_by_label:
        components["loopI"] = ["mu_loopI_counts.fits"]
    elif ("loopA" in coeffs_by_label) and ("loopB" in coeffs_by_label):
        components["loopA"] = ["mu_loopA_counts.fits"]
        components["loopB"] = ["mu_loopB_counts.fits"]

    if "fb_flat" in coeffs_by_label:
        components["fb_flat"] = ["mu_fb_flat_counts.fits", "mu_bubbles_flat_binary_counts.fits"]
    elif ("fb_pos" in coeffs_by_label) and ("fb_neg" in coeffs_by_label):
        components["fb_pos"] = ["mu_fb_pos_counts.fits"]
        components["fb_neg"] = ["mu_fb_neg_counts.fits"]

    model = np.zeros((ny, nx), dtype=float)
    for lab, candidates in components.items():
        if lab not in coeffs_by_label:
            continue
        path = pick_existing(str(templates_dir), candidates)
        if path is None:
            raise FileNotFoundError(f"Missing template for {lab}; tried {candidates} in {templates_dir}")
        mu, _ = _read_cube(path, expected_shape=(nE, ny, nx))
        ak = float(np.asarray(coeffs_by_label[str(lab)], float).reshape(-1)[k])
        if np.isfinite(ak) and ak != 0.0:
            model += ak * np.asarray(mu[k], float)
    return model

def fit_per_bin_poisson_mle_cellwise_counts(
    counts,
    templates,   # list of arrays, each same shape as counts
    mask3d,
    lon,
    lat,
    roi_lon,
    roi_lat,
    cell_deg=10.0,
    nonneg=True,
    tiny=1e-30,
    maxiter=200,
    init="nnls",
):
    """
    Cell-wise, per-energy-bin template fit in TRUE COUNTS UNITS using Poisson MLE.

    For each cell and energy bin k, fits coefficients a_j such that:
        mu_pix = sum_j a_j * T_j_pix
    with y_pix ~ Poisson(mu_pix).

    Assumptions (enforced):
      - counts and every template are in counts-per-pixel (expected counts)
      - no per-cell normalisation or alternative parameterisations
      - supports optional non-negativity a_j >= 0 (default True)

    Parameters
    ----------
    counts : array, shape (nE, ...spatial...)
    templates : list of arrays, each shape == counts.shape
    mask3d : bool array, shape == counts.shape
    lon, lat : arrays, shape == counts.shape[1:]
    roi_lon, roi_lat : floats, ROI half-widths (deg)
    cell_deg : float, cell size (deg)
    nonneg : bool, enforce a_j >= 0
    tiny : float, floor for mu to avoid log(0)
    maxiter : int, optimizer cap
    init : {"nnls","lsq","ones"}, init strategy for MLE

    Returns
    -------
    cells : list of bool arrays, shape == spatial
        Cell masks.
    coeff_cells : array, shape (nCells, nE, nComp)
        Best-fit coefficients.
    info : dict
        success (nCells,nE), nll (nCells,nE), message (nCells,nE object)
    """
    counts = np.asarray(counts)
    mask3d = np.asarray(mask3d, dtype=bool)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if counts.ndim < 2:
        raise ValueError("counts must have shape (nE, ...spatial...)")
    if mask3d.shape != counts.shape:
        raise ValueError(f"mask3d must match counts shape {counts.shape}, got {mask3d.shape}")

    nE = counts.shape[0]
    spatial_shape = counts.shape[1:]

    if lon.shape != spatial_shape or lat.shape != spatial_shape:
        raise ValueError(f"lon/lat must match spatial shape {spatial_shape}, got lon {lon.shape}, lat {lat.shape}")

    if not isinstance(templates, (list, tuple)) or len(templates) == 0:
        raise ValueError("templates must be a non-empty list of arrays")
    ncomp = len(templates)

    for j, T in enumerate(templates):
        T = np.asarray(T)
        if T.shape != counts.shape:
            raise ValueError(f"templates[{j}] must match counts shape {counts.shape}, got {T.shape}")

    # Build lon/lat cell masks
    l_edges = np.arange(-roi_lon, roi_lon + 1e-9, cell_deg)
    b_edges = np.arange(-roi_lat, roi_lat + 1e-9, cell_deg)

    cells = []
    for l0 in l_edges[:-1]:
        l1 = l0 + cell_deg
        in_l = (lon >= l0) & (lon < l1)
        for b0 in b_edges[:-1]:
            b1 = b0 + cell_deg
            cell = in_l & (lat >= b0) & (lat < b1)
            if np.any(cell):
                cells.append(cell)

    nCells = len(cells)
    coeff_cells = np.zeros((nCells, nE, ncomp), dtype=float)
    success = np.zeros((nCells, nE), dtype=bool)
    nll_vals = np.full((nCells, nE), np.nan, dtype=float)
    messages = np.empty((nCells, nE), dtype=object)

    def initial_guess(y, X):
        if init == "ones":
            return np.full(ncomp, max(np.sum(y) / max(ncomp, 1), 1.0), dtype=float)
        if init == "lsq":
            a, *_ = np.linalg.lstsq(X, y, rcond=None)
            return np.clip(a, 0.0, None) if nonneg else a
        # default: nnls
        try:
            a0, _ = nnls(X, y)
            if not np.isfinite(a0).all() or np.all(a0 == 0):
                a0 = np.full(ncomp, 1e-3, dtype=float)
            return a0
        except Exception:
            return np.full(ncomp, 1e-3, dtype=float)

    bounds = [(0.0, None)] * ncomp if nonneg else [(None, None)] * ncomp

    for ci, cell2d in enumerate(cells):
        for k in range(nE):
            m = mask3d[k] & cell2d
            if not np.any(m):
                messages[ci, k] = "empty mask"
                continue

            y = counts[k][m].astype(float).ravel()
            X = np.stack([np.asarray(templates[j])[k][m].astype(float).ravel() for j in range(ncomp)], axis=1)

            good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            y = y[good]
            X = X[good]

            if y.size == 0:
                messages[ci, k] = "no finite pixels"
                continue

            # Poisson NLL and gradient
            def nll_and_grad(a):
                mu = X @ a
                mu = np.clip(mu, tiny, None)
                nll = np.sum(mu - y * np.log(mu))  # drop constant ln(y!)
                r = 1.0 - (y / mu)
                grad = X.T @ r
                return nll, grad

            def fun(a):
                v, _ = nll_and_grad(a)
                return v

            def jac(a):
                _, g = nll_and_grad(a)
                return g

            a0 = initial_guess(y, X)

            opt = minimize(
                fun,
                x0=a0,
                jac=jac,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": int(maxiter)},
            )

            coeff_cells[ci, k] = opt.x
            success[ci, k] = bool(opt.success)
            nll_vals[ci, k] = float(opt.fun)
            messages[ci, k] = str(opt.message)

    info = {"success": success, "nll": nll_vals, "message": messages}
    return cells, coeff_cells, info



# -------------------------
# Template auto-discovery
# -------------------------
def pick_existing(template_dir, candidates):
    for name in candidates:
        p = os.path.join(template_dir, name)
        if os.path.exists(p):
            return p
    return None


def resolve_templates(template_dir):
    """
    Prefer counts templates. Fall back to intensity templates if needed.
    """
    spec = {}

    spec["GAS"] = pick_existing(template_dir, [
        "mu_gas_counts.fits",
        "gas_dnde.fits",
    ])

    spec["ICS"] = pick_existing(template_dir, [
        "mu_ics_counts.fits",
        "ics_dnde.fits",
    ])

    spec["ISO"] = pick_existing(template_dir, [
        "mu_iso_counts.fits",
        "mu_iso.fits",
        "mu_iso_counts.fits.gz",
        "template_iso_intensity.fits",
        "iso_dnde.fits",
        "iso_E2dnde.fits",
    ])

    spec["PS"] = pick_existing(template_dir, [
        "mu_ps_counts.fits",
        "mu_ps.fits",
        "ps_dnde.fits",
        "ps_E2dnde.fits",
    ])

    spec["NFW"] = pick_existing(template_dir, [
        "mu_nfw_rho2.5_g1.25_counts.fits",
        "nfw_rho2.5_g1.25_dnde.fits",
        "nfw_rho2.5_g1.25_E2dnde.fits",
    ])

    spec["LOOPI"] = pick_existing(template_dir, [
        "mu_loopI_counts.fits",
        "mu_loopi_counts.fits",
        "loopI_dnde.fits",
        "loopI_E2dnde.fits",
    ])

    spec["BUB_POS"] = pick_existing(template_dir, [
        "mu_bubbles_pos_counts.fits",
        "bubbles_pos_dnde.fits",
        "bubbles_pos_E2dnde.fits",
    ])

    spec["BUB_NEG"] = pick_existing(template_dir, [
        "mu_bubbles_neg_counts.fits",
        "bubbles_neg_dnde.fits",
        "bubbles_neg_E2dnde.fits",
    ])

    spec["FB_FLAT"] = pick_existing(template_dir, [
        "mu_bubbles_flat_binary_counts.fits",
        "bubbles_flat_binary_dnde.fits",
        "bubbles_flat_binary_E2dnde.fits",
    ])

    missing = [k for k, v in spec.items() if v is None]
    if missing:
        raise RuntimeError(
            "Missing templates for: " + ", ".join(missing) +
            f"\nLooked in: {template_dir}\n"
            "Expected files like mu_gas_counts.fits, mu_ics_counts.fits, mu_iso_counts.fits, mu_ps_counts.fits, mu_nfw_counts.fits, "
            "mu_loopI_counts.fits, mu_bubbles_flat_counts.fits"
        )
    return spec


# -------------------------
# Plot makers
# -------------------------
def plot_raw_counts(Ectr_gev, C, Cerr, outpath, title):
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(Ectr_gev, C, yerr=Cerr, fmt="o", capsize=2)
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Counts per energy bin")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("✓ wrote", outpath)


def plot_halo_plus_residual_bins0to3(
    *,
    counts,
    expo,
    omega,
    dE_mev,
    coeffs_halo,
    templates_dir,
    wcs,
    roi2d,
    disk_keep,
    ext_keep3d,
    sigma_smooth_deg,
    binsz_deg,
    out_png,
    vlims_by_bin,
    Ectr_gev,
):
    nE, ny, nx = counts.shape
    disk_band_2d = (roi2d & (~disk_keep))

    maps = []
    extmasks = []
    for k in range(4):
        if k >= nE:
            raise ValueError(f"Requested bin {k} but only have nE={nE}")
        model_nonhalo_halo_k = _build_nonhalo_model_map_k(
            k=int(k),
            coeffs_by_label=coeffs_halo,
            templates_dir=templates_dir,
            nE=nE,
            ny=ny,
            nx=nx,
        )
        halo_plus_resid_counts = np.asarray(counts[int(k)], float) - np.asarray(model_nonhalo_halo_k, float)
        m = counts_to_flux(halo_plus_resid_counts, expo[int(k)], omega, dE_mev[int(k)])
        m = smooth_map_deg(m, float(sigma_smooth_deg), float(binsz_deg))

        extmask2d = (roi2d & disk_keep & (~np.asarray(ext_keep3d[int(k)], dtype=bool)))
        m[np.asarray(extmask2d, dtype=bool)] = np.nan

        maps.append(m)
        extmasks.append(extmask2d)

    cmap = LinearSegmentedColormap.from_list(
        "cyan_blue_black_orange_yellow",
        ["cyan", "blue", "black", "orangered", "yellow"],
        N=256,
    ).copy()
    cmap.set_bad(color="0.6")

    disk_overlay = np.where(np.asarray(disk_band_2d, dtype=bool), 1.0, np.nan)

    fig = plt.figure(figsize=(10.5, 9.0), constrained_layout=True)
    axes = np.array(
        [
            [fig.add_subplot(2, 2, 1, projection=wcs), fig.add_subplot(2, 2, 2, projection=wcs)],
            [fig.add_subplot(2, 2, 3, projection=wcs), fig.add_subplot(2, 2, 4, projection=wcs)],
        ]
    )

    def draw(ax, img, k):
        vlim = float(vlims_by_bin[int(k)])
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.imshow(
            disk_overlay,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=0.85,
            interpolation="nearest",
        )
        ax.text(
            0.03,
            0.93,
            rf"bin {int(k)} ({Ectr_gev[int(k)]:.2f} GeV) halo+residual",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="top",
        )
        ax.set_xlabel("Galactic longitude")
        ax.set_ylabel("Galactic latitude")
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06, location="top")
        cbar.set_label(r"flux  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")

    draw(axes[0, 0], maps[0], 0)
    draw(axes[0, 1], maps[1], 1)
    draw(axes[1, 0], maps[2], 2)
    draw(axes[1, 1], maps[3], 3)

    fig.savefig(out_png, dpi=250)
    plt.close(fig)
    print("✓ wrote", out_png)


def plot_fit_fig2(Ectr_gev, data_y, data_yerr, comp_specs, outpath, title):
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for lab, y in comp_specs.items():
        ax.plot(Ectr_gev, y, marker="o", label=lab)

    ax.errorbar(Ectr_gev, data_y, yerr=data_yerr, fmt="o", capsize=2, label="data")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("✓ wrote", outpath)

def main():
    ap = argparse.ArgumentParser()
    repo_dir_default = os.environ.get(
        "REPO_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    data_dir_default = os.path.join(repo_dir_default, "fermi_data", "totani")

    ap.add_argument(
        "--counts",
        default=os.path.join(data_dir_default, "processed", "counts_ccube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--expo",
        default=os.path.join(data_dir_default, "processed", "expcube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--templates-dir",
        default=os.path.join(data_dir_default, "processed", "templates"),
    )
    ap.add_argument(
        "--ext-mask",
        default=os.path.join(data_dir_default, "processed", "templates", "mask_extended_sources.fits"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig12"),
    )

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)
    ap.add_argument("--target-gev", type=float, default=21.0)
    ap.add_argument("--smoothing-deg", type=float, default=1.0)
    ap.add_argument("--vlim", type=float, default=2e-12)

    ap.add_argument("--cell-deg", type=float, default=10.0)
    ap.add_argument(
        "--mcmc-dir",
        default=None,
        help="If set alone, used for both no-halo and with-halo panels.",
    )
    ap.add_argument(
        "--mcmc-dir-nohalo",
        default=None,
        help="MCMC directory for the fit WITHOUT halo component.",
    )
    ap.add_argument(
        "--mcmc-dir-halo",
        default=None,
        help="MCMC directory for the fit WITH halo component.",
    )
    ap.add_argument(
        "--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"],
        default="f_ml",
        help="Which MCMC summary coefficient to use per bin",
    )
    ap.add_argument(
        "--halo-label",
        default="nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno",
        help="Halo component key in the with-halo MCMC outputs.",
    )
    ap.add_argument(
        "--combine-loopI",
        action="store_true",
        help="If loopA/loopB exist but not loopI, use loopI=loopA+loopB.",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    counts, hdr, _Emin_mev, _Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    print(_Emin_mev, _Emax_mev)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial
    Ectr_gev = Ectr_mev / 1000.0

    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    disk_keep = np.abs(lat) >= args.disk_cut

    k = int(np.argmin(np.abs(Ectr_gev - float(args.target_gev))))

    # Extended-source mask file convention: True=keep, False=masked
    ext_keep3d = np.ones((nE, ny, nx), dtype=bool)
    if args.ext_mask and os.path.exists(args.ext_mask):
        ext_keep3d = load_mask_any_shape(args.ext_mask, counts.shape)

    # For plotting, we grey-out the masked pixels at the target energy bin
    extended_sources_mask2d = roi2d & disk_keep & (~ext_keep3d[k])

    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=ext_keep3d,
        counts=counts,
        expo=expo,
        extra2d=disk_keep,
    )
    fit_mask2d = fit_mask3d[k]

    # Diagnostics: masked fraction inside ROI&disk
    region2d = (roi2d & disk_keep)
    frac_masked_k = float(np.mean((~ext_keep3d[k])[region2d])) if np.any(region2d) else float("nan")
    frac_masked_all = float(np.mean((~ext_keep3d)[:, region2d])) if np.any(region2d) else float("nan")
    print(f"[fig11] ext-mask masked frac in ROI&disk: k={k:02d} -> {frac_masked_k:.4f} ; avg over bins -> {frac_masked_all:.4f}")

    templates_dir = args.templates_dir

    tab_nohalo = load_mcmc_coeffs_by_label(mcmc_dir=str(args.mcmc_dir_nohalo), stat=str(args.mcmc_stat), nE=nE)
    coeffs_nohalo = dict(tab_nohalo.coeffs_by_label)
    tab_halo = load_mcmc_coeffs_by_label(mcmc_dir=str(args.mcmc_dir_halo), stat=str(args.mcmc_stat), nE=nE)
    coeffs_halo = dict(tab_halo.coeffs_by_label)

    if bool(args.combine_loopI):
        coeffs_nohalo = combine_loopI(coeffs_by_label=coeffs_nohalo, out_key="loopI", drop_inputs=True)
        coeffs_halo = combine_loopI(coeffs_by_label=coeffs_halo, out_key="loopI", drop_inputs=True)

    halo_label_used = str(args.halo_label)
    if halo_label_used not in coeffs_halo:
        nfw_keys = sorted([k for k in coeffs_halo.keys() if str(k).lower().startswith("nfw_")])
        if len(nfw_keys) == 1:
            halo_label_used = str(nfw_keys[0])
            print(f"[fig11] --halo-label not found; auto-using halo label from MCMC outputs: {halo_label_used}")
        else:
            raise SystemExit(
                f"Missing halo label in with-halo MCMC outputs: {str(args.halo_label)}\n"
                f"Available nfw_* labels: {nfw_keys}"
            )

    halo_path = pick_existing(
        str(templates_dir),
        [
            f"mu_{halo_label_used}_counts.fits",
            f"mu_{str(args.halo_label)}_counts.fits",
            "mu_nfw_rho2.5_g1.25_counts.fits",
        ],
    )
    if halo_path is None:
        raise FileNotFoundError(f"Missing halo template for label '{halo_label_used}' in {templates_dir}")
    halo_mu, _ = _read_cube(halo_path, expected_shape=(nE, ny, nx))

    model_nonhalo_nohalo = _build_nonhalo_model_map_k(
        k=k,
        coeffs_by_label=coeffs_nohalo,
        templates_dir=templates_dir,
        nE=nE,
        ny=ny,
        nx=nx,
    )
    model_nonhalo_halo = _build_nonhalo_model_map_k(
        k=k,
        coeffs_by_label=coeffs_halo,
        templates_dir=templates_dir,
        nE=nE,
        ny=ny,
        nx=nx,
    )

    halo_coeff_k = float(np.asarray(coeffs_halo[str(halo_label_used)], float).reshape(-1)[k])
    halo_bestfit = halo_coeff_k * np.asarray(halo_mu[k], float)

    out_png = os.path.join(args.outdir, f"totani_fig11_like_E{Ectr_gev[k]:.2f}GeV.png")
    make_totani_fig11_like_with_extmask(
        data_counts=counts[k],
        model_except_halo_nohalo_counts=model_nonhalo_nohalo,
        model_except_halo_halo_counts=model_nonhalo_halo,
        halo_counts=halo_bestfit,
        expo_map=expo[k],
        omega_sr_map=omega,
        dE_mev=dE_mev[k],
        extmask2d=extended_sources_mask2d,
        wcs=wcs,
        disk_band_2d=(roi2d & (~disk_keep)),
        sigma_smooth_deg=float(args.smoothing_deg),
        binsz_deg=float(args.binsz),
        vlim=float(args.vlim),
        out_png=out_png,
        energy_label=f"{Ectr_gev[k]:.2f} GeV",
    )

    out_png_fig12 = os.path.join(args.outdir, "totani_fig12_halo_plus_residual_bins0to3.png")
    vlims_by_bin = {0: 4e-10, 1: 1e-10, 2: 3e-11, 3: 5e-12}
    plot_halo_plus_residual_bins0to3(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        coeffs_halo=coeffs_halo,
        templates_dir=templates_dir,
        wcs=wcs,
        roi2d=roi2d,
        disk_keep=disk_keep,
        ext_keep3d=ext_keep3d,
        sigma_smooth_deg=float(args.smoothing_deg),
        binsz_deg=float(args.binsz),
        out_png=out_png_fig12,
        vlims_by_bin=vlims_by_bin,
        Ectr_gev=Ectr_gev,
    )

    halo_coeffs_all = np.asarray(coeffs_halo[str(halo_label_used)], float).reshape(-1)

    def _stack_plot(ks, tag, label):
        if len(ks) == 0:
            return
        data_sum = np.nansum(counts[ks], axis=0)
        denom_sum = np.nansum(expo[ks] * omega[None, :, :] * dE_mev[ks, None, None], axis=0)

        model_nohalo_sum = np.zeros((ny, nx), dtype=float)
        model_halo_nonhalo_sum = np.zeros((ny, nx), dtype=float)
        for kk in ks:
            model_nohalo_sum += _build_nonhalo_model_map_k(
                k=int(kk),
                coeffs_by_label=coeffs_nohalo,
                templates_dir=templates_dir,
                nE=nE,
                ny=ny,
                nx=nx,
            )
            model_halo_nonhalo_sum += _build_nonhalo_model_map_k(
                k=int(kk),
                coeffs_by_label=coeffs_halo,
                templates_dir=templates_dir,
                nE=nE,
                ny=ny,
                nx=nx,
            )

        halo_sum = np.zeros((ny, nx), dtype=float)
        for kk in ks:
            ak = float(halo_coeffs_all[kk])
            if np.isfinite(ak) and ak != 0.0:
                halo_sum += ak * np.asarray(halo_mu[kk], float)

        extmask_stack = (roi2d & disk_keep) & (~np.all(ext_keep3d[ks], axis=0))
        out_png_stack = os.path.join(args.outdir, f"totani_fig12_like_{tag}_E{Ectr_gev[k]:.2f}GeV.png")
        if tag == "below":
            vlim_stack = 1e-10
        elif tag == "above":
            vlim_stack = 5e-13
        else:
            vlim_stack = float(args.vlim)
        make_totani_fig11_like_with_extmask_stacked(
            data_counts=data_sum,
            model_except_halo_nohalo_counts=model_nohalo_sum,
            model_except_halo_halo_counts=model_halo_nonhalo_sum,
            halo_counts=halo_sum,
            denom_map=denom_sum,
            extmask2d=extmask_stack,
            wcs=wcs,
            disk_band_2d=(roi2d & (~disk_keep)),
            sigma_smooth_deg=float(args.smoothing_deg),
            binsz_deg=float(args.binsz),
            vlim=float(vlim_stack),
            out_png=out_png_stack,
            energy_label=label,
        )

    ks_lo = list(range(0, k))
    ks_hi = list(range(k + 1, nE))
    _stack_plot(ks_lo, "below", rf"E < {Ectr_gev[k]:.2f} GeV")
    _stack_plot(ks_hi, "above", rf"E > {Ectr_gev[k]:.2f} GeV")


if __name__ == "__main__":
    main()
