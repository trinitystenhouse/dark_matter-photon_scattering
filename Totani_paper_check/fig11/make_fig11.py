#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize, nnls

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (  # noqa: E402
    load_mask_any_shape,
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)
from totani_helpers.cellwise_fit import (
    fit_cellwise_poisson_mle_counts,
    per_bin_total_counts_from_cellwise_coeffs,
)


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
    model_except_halo_counts,      # 2D counts (all fitted comps except halo)
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
    resid_nohalo_counts   = data_counts - model_except_halo_counts
    resid_withhalo_counts = data_counts - (model_except_halo_counts + halo_counts)
    halo_plus_resid_counts = data_counts - model_except_halo_counts  # = halo + resid_withhalo

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
    cmap = plt.get_cmap("seismic").copy()
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
    model_except_halo_counts,
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
    resid_nohalo_counts = data_counts - model_except_halo_counts
    resid_withhalo_counts = data_counts - (model_except_halo_counts + halo_counts)
    halo_plus_resid_counts = data_counts - model_except_halo_counts

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
    cmap = plt.get_cmap("seismic").copy()
    cmap.set_bad(color="0.6")

    disk_overlay = None
    if disk_band_2d is not None:
        disk_band_2d = np.asarray(disk_band_2d, dtype=bool)
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
            0.03,
            0.93,
            text,
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


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Make BOTH: raw counts plot + Totani Fig2-style fitted component plot.")
    ap.add_argument(
        "--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"),
        help="counts_ccube_*.fits",
    )
    ap.add_argument(
        "--expo",
        default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"),
        help="expcube_*.fits",
    )
    ap.add_argument(
        "--templates-dir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
        help="directory containing templates (mu_*_counts.fits etc)",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig2_3"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--cell-deg", type=float, default=10.0)
    ap.add_argument(
        "--weighting",
        default="uniform",
        choices=["uniform", "poisson"],
        help="pixel weighting scheme inside each fit cell",
    )
    ap.add_argument(
        "--cell-normalize",
        action="store_true",
        help="normalize each template column to sum=1 within each (cell, energy) before NNLS (old behavior)",
    )
    ap.add_argument("--disk-cut-fit", type=float, default=10.0, help="disk cut |b|>=X applied ONLY in fit plot")
    ap.add_argument("--mask-fit", default=None, help="optional mask FITS for fit plot (2D or 3D), True=keep")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load counts + EBOUNDS
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    # Exposure
    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")

    # Solid angle map
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # Lon/lat and ROI box
    lon, lat = lonlat_grids(wcs, ny, nx)
    roi2d = build_roi_box_mask(lon, lat, args.roi_lon, args.roi_lat)

    # X axis in GeV (IMPORTANT)
    Ectr_gev = Ectr_mev / 1000.0

    # -------------------------
    # (A) Raw counts UNMASKED plot
    # "unmasked" here = ROI box only (no disk cut, no source mask)
    # -------------------------
    mask3d_raw = roi2d[None, :, :] & np.ones((nE, ny, nx), bool)
    C_raw, C_raw_err = raw_counts_spectrum(counts, mask3d_raw)

    out_counts = os.path.join(args.outdir, "raw_counts_unmasked_roi.png")
    plot_raw_counts(
        Ectr_gev, C_raw, C_raw_err, out_counts,
        title=rf"Raw counts per bin (ROI: |l|≤{args.roi_lon}°, |b|≤{args.roi_lat}°; no disk/source mask)"
    )

    # Optional source/extended mask (True=keep)
    if args.mask_fit is not None:
        srcmask = load_mask_any_shape(args.mask_fit, nE, ny, nx)
    else:
        srcmask = np.ones((nE, ny, nx), bool)

    # Resolve templates automatically
    tpl = resolve_templates(args.templates_dir)

    print("[templates] dir:", args.templates_dir)
    for k in ["GAS", "ICS", "ISO", "PS", "LOOPI", "FB_FLAT", "BUB_POS", "BUB_NEG", "NFW"]:
        if k in tpl:
            print(f"[templates] {k}: {tpl[k]}")
    labels = ["GAS", "ICS", "ISO", "PS", "LOOPI", "FB_FLAT", "NFW"]

    # Load templates
    labels = ["gas", "iso", "ps", "nfw_rho2.5_g1.25", "loopI", "ics", "bubbles_flat_binary"]

    mu_list, headers = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",  
        hdu=0,
    )

    assert_templates_match_counts(counts, mu_list, labels)


    def run_fit_and_plot(
        region_name,
        region2d,
        out_png,
        out_coeff_txt,
        cells,
        coeff_cells,
        counts,
        expo,
        omega,
        dE_mev,
        Ectr_mev,
        Ectr_gev,
        srcmask,
        mu_list,
        labels,
    ):
        """
        region2d: boolean keep-mask (ny,nx)
        Uses mask3d_plot = srcmask (3d) AND region2d (2d) for spectra + output sums.
        Assumes mu_list are TRUE COUNTS templates and coeff_cells multiply them directly.
        """

        # Final 3D mask for THIS region
        mask3d_plot = srcmask & region2d[None, :, :]

        # Data E^2 spectrum in this region
        E2_data, E2err_data = data_E2_spectrum_counts(
            counts=counts,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask3d=mask3d_plot,
        )

        # Model/component E^2 spectra in this region (from fitted cell coefficients)
        E2_comp, E2_model = model_E2_spectrum_from_cells_counts(
            coeff_cells=coeff_cells,
            cells=cells,
            templates=mu_list,   # TRUE counts templates
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask3d=mask3d_plot,
        )

        # Pack for plotting
        comp_specs = {lab: E2_comp[j] for j, lab in enumerate(labels)}
        comp_specs["MODEL_SUM"] = E2_model

        # Closure printout: (data - model)/data in each bin
        frac_resid = np.full_like(E2_data, np.nan, dtype=float)
        for k in range(len(E2_data)):
            if np.isfinite(E2_data[k]) and E2_data[k] != 0 and np.isfinite(E2_model[k]):
                frac_resid[k] = (E2_data[k] - E2_model[k]) / E2_data[k]

        print("[closure]", region_name)
        for k in range(len(E2_data)):
            if np.isfinite(frac_resid[k]):
                print(f"  k={k:02d} E={Ectr_gev[k]:.3g} GeV frac_resid={(100*frac_resid[k]):+.2f}%")

        # Plot (uses your plot_fit_fig2)
        plot_fit_fig2(
            Ectr_gev=Ectr_gev,
            data_y=E2_data,
            data_yerr=E2err_data,
            comp_specs=comp_specs,
            outpath=out_png,
            title=region_name,
        )

        # Save per-bin total COUNTS attributed to each component within this region
        nE = counts.shape[0]
        with open(out_coeff_txt, "w") as f:
            f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
            for k in range(nE):
                csum = np.zeros(len(labels), float)

                for ci, cell2d in enumerate(cells):
                    cm = mask3d_plot[k] & cell2d
                    if not np.any(cm):
                        continue

                    a = coeff_cells[ci, k, :]  # (nComp,)
                    if not np.all(np.isfinite(a)):
                        continue

                    for j in range(len(labels)):
                        s = float(np.nansum(mu_list[j][k][cm]))  # template counts in (cell ∩ region)
                        if np.isfinite(s) and s != 0.0:
                            csum[j] += float(a[j]) * s

                f.write(
                    f"{k:02d} {Ectr_gev[k]:.6g} " +
                    " ".join(f"{csum[j]:.6g}" for j in range(len(labels))) +
                    "\n"
                )

        print("✓ wrote", out_coeff_txt)

    # Define regions
    disk_cut = float(args.disk_cut_fit)
    fig2_region2d = roi2d                              # Fig 2: include disk
    fig3_region2d = roi2d & (np.abs(lat) >= disk_cut)  # Fig 3: exclude disk

    # Fit mask is the SAME for both figures: ROI including disk
    fit_mask3d = srcmask & roi2d[None, :, :]

    cells, coeff_cells, info = fit_per_bin_poisson_mle_cellwise_counts(
        counts=counts,
        templates=mu_list,      # your list of count templates
        mask3d=fit_mask3d,
        lon=lon, lat=lat,
        roi_lon=args.roi_lon, roi_lat=args.roi_lat,
        cell_deg=10.0,
        nonneg=True,
    )


    # Fig 2 outputs
    out_fig2 = os.path.join(args.outdir, "totani_fig11_nohalo_fit_components.png")
    out_c2   = os.path.join(args.outdir, "fit_coefficients_fig11_nohalo.txt")

    run_fit_and_plot(
        region_name=rf"|l|≤{args.roi_lon}°\n|b|≤{args.roi_lat}°",
        region2d=roi2d,
        out_png=out_fig2,
        out_coeff_txt=out_c2,
        cells=cells,
        coeff_cells=coeff_cells,
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        Ectr_gev=Ectr_gev,
        srcmask=srcmask,
        mu_list=mu_list,
        labels=labels,
    )


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
        default=os.path.join(os.path.dirname(__file__), "plots_fig11"),
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
        "--nohalo-coeff-out",
        default=None,
        help="Output coeff file for the internally-generated no-halo fit (default: <outdir>/fit_coefficients_fig11_nohalo.txt)",
    )
    ap.add_argument(
        "--reuse-nohalo",
        action="store_true",
        help="If set and --nohalo-coeff-out exists, reuse it instead of re-fitting.",
    )

    ap.add_argument(
        "--coeff-file",
        required=True,
        help="Per-bin total counts coefficients file (e.g. fig2_3/plots_fig2_3/fit_coefficients_fig2_highlat.txt)",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with fits.open(args.counts) as h:
        counts = np.array(h[0].data, dtype=float)
        hdr = h[0].header
        eb = h["EBOUNDS"].data

    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    Emin_mev = eb["E_MIN"].astype(float) / 1000.0
    Emax_mev = eb["E_MAX"].astype(float) / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    Ectr_gev = Ectr_mev / 1000.0

    with fits.open(args.expo) as h:
        expo_raw = np.array(h[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(h)
    expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon_w = ((lon + 180.0) % 360.0) - 180.0
    roi2d = (np.abs(lon_w) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    disk_keep = np.abs(lat) >= args.disk_cut

    k = int(np.argmin(np.abs(Ectr_gev - float(args.target_gev))))

    # Extended-source mask file convention: True=keep, False=masked
    ext_keep3d = np.ones((nE, ny, nx), dtype=bool)
    if args.ext_mask and os.path.exists(args.ext_mask):
        ext_keep3d = load_mask_any_shape(args.ext_mask, counts.shape)

    # For plotting, we grey-out the masked pixels at the target energy bin
    extended_sources_mask2d = roi2d & disk_keep & (~ext_keep3d[k])

    # Fit mask is energy-dependent because the extended-source mask is energy-dependent
    fit_mask3d = (roi2d & disk_keep)[None, :, :] & ext_keep3d
    fit_mask2d = fit_mask3d[k]

    # Diagnostics: masked fraction inside ROI&disk
    region2d = (roi2d & disk_keep)
    frac_masked_k = float(np.mean((~ext_keep3d[k])[region2d])) if np.any(region2d) else float("nan")
    frac_masked_all = float(np.mean((~ext_keep3d)[:, region2d])) if np.any(region2d) else float("nan")
    print(f"[fig11] ext-mask masked frac in ROI&disk: k={k:02d} -> {frac_masked_k:.4f} ; avg over bins -> {frac_masked_all:.4f}")

    templates_dir = args.templates_dir
    base_names = [
        "mu_ics_counts.fits",
        "mu_iso_counts.fits",
        "mu_gas_counts.fits",
        "mu_ps_counts.fits",
        "mu_loopI_counts.fits",
        "mu_bubbles_flat_binary_counts.fits",
    ]
    base_mu = []
    for fn in base_names:
        path = os.path.join(templates_dir, fn)
        d, _ = _read_cube(path, expected_shape=(nE, ny, nx))
        base_mu.append(d)

    halo_path = os.path.join(templates_dir, "mu_nfw_rho2.5_g1.25_counts.fits")
    halo_mu, _ = _read_cube(halo_path, expected_shape=(nE, ny, nx))

    coeff_path = args.coeff_file
    if not os.path.isabs(coeff_path):
        coeff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fig2_3", coeff_path))
    if not os.path.exists(coeff_path):
        raise SystemExit(f"Coefficient file not found: {coeff_path}")

    # ------------------------------------------------------------
    # (1) Generate a Fig2_3-like fit WITHOUT halo (cellwise Poisson MLE),
    #     save per-bin total counts, and build the no-halo model map at k.
    # ------------------------------------------------------------
    nohalo_out = args.nohalo_coeff_out
    if nohalo_out is None:
        nohalo_out = os.path.join(args.outdir, "fit_coefficients_fig11_nohalo.txt")

    nohalo_labels = ["ics", "iso", "gas", "ps", "loopI", "bubbles_flat_binary"]
    if (not args.reuse_nohalo) or (not os.path.exists(nohalo_out)):
        res_fit = fit_cellwise_poisson_mle_counts(
            counts=counts,
            templates=base_mu,
            mask3d=fit_mask3d,
            lon=lon_w,
            lat=lat,
            roi_lon=float(args.roi_lon),
            roi_lat=float(args.roi_lat),
            cell_deg=float(args.cell_deg),
            nonneg=True,
        )
        cells = res_fit["cells"]
        coeff_cells = res_fit["coeff_cells"]

        totals = per_bin_total_counts_from_cellwise_coeffs(
            cells=cells,
            coeff_cells=coeff_cells,
            templates=base_mu,
            mask3d=fit_mask3d,
        )
        with open(nohalo_out, "w") as f:
            f.write("# k  Ectr(GeV)  " + "  ".join(nohalo_labels) + "\n")
            for kk in range(nE):
                f.write(
                    f"{kk:02d} {Ectr_gev[kk]:.6g} "
                    + " ".join(f"{totals[kk, j]:.6g}" for j in range(len(nohalo_labels)))
                    + "\n"
                )
        print("✓ wrote", nohalo_out)
    else:
        print("[fig11] reusing no-halo coefficients:", nohalo_out)

    # Read no-halo total counts at target k and build the no-halo model map
    nohalo_counts = []
    for lab in nohalo_labels:
        _, c = read_coeff_counts(nohalo_out, lab)
        nohalo_counts.append(float(c[k]))
    model_nohalo = np.zeros((ny, nx), dtype=float)
    for Ck, mu in zip(nohalo_counts, base_mu):
        model_nohalo += counts_coeff_to_map(mu[k], Ck, fit_mask2d)

    # ------------------------------------------------------------
    # (2) Halo counts from your existing Fig2/3 coefficient file
    # ------------------------------------------------------------
    _, halo_counts_arr = read_coeff_counts(coeff_path, "nfw")
    halo_counts_k = float(halo_counts_arr[k])
    halo_bestfit = counts_coeff_to_map(halo_mu[k], halo_counts_k, fit_mask2d)

    out_png = os.path.join(args.outdir, f"totani_fig11_like_E{Ectr_gev[k]:.2f}GeV.png")
    make_totani_fig11_like_with_extmask(
        data_counts=counts[k],
        model_except_halo_counts=model_nohalo,
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

    nohalo_counts_all = [read_coeff_counts(nohalo_out, lab)[1] for lab in nohalo_labels]
    halo_counts_all = read_coeff_counts(coeff_path, "nfw")[1]

    def _stack_plot(ks, tag, label):
        if len(ks) == 0:
            return
        data_sum = np.nansum(counts[ks], axis=0)
        denom_sum = np.nansum(expo[ks] * omega[None, :, :] * dE_mev[ks, None, None], axis=0)

        model_nohalo_sum = np.zeros((ny, nx), dtype=float)
        for j, mu in enumerate(base_mu):
            c_arr = np.asarray(nohalo_counts_all[j], dtype=float)
            for kk in ks:
                model_nohalo_sum += counts_coeff_to_map(mu[kk], float(c_arr[kk]), fit_mask3d[kk])

        halo_sum = np.zeros((ny, nx), dtype=float)
        for kk in ks:
            halo_sum += counts_coeff_to_map(halo_mu[kk], float(halo_counts_all[kk]), fit_mask3d[kk])

        extmask_stack = (roi2d & disk_keep) & (~np.all(ext_keep3d[ks], axis=0))
        out_png_stack = os.path.join(args.outdir, f"totani_fig11_like_{tag}_E{Ectr_gev[k]:.2f}GeV.png")
        if tag == "below":
            vlim_stack = 1e-10
        elif tag == "above":
            vlim_stack = 5e-13
        else:
            vlim_stack = float(args.vlim)
        make_totani_fig11_like_with_extmask_stacked(
            data_counts=data_sum,
            model_except_halo_counts=model_nohalo_sum,
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
