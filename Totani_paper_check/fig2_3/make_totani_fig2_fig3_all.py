#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

# Optional NNLS (recommended)
try:
    from scipy.optimize import nnls
    HAVE_NNLS = True
except Exception:
    HAVE_NNLS = False


def build_roi_box_mask(lon, lat, roi_lon=60.0, roi_lat=60.0):
    return (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)


def load_mask_any_shape(mask_path, nE, ny, nx):
    m = fits.getdata(mask_path).astype(bool)
    if m.shape == (nE, ny, nx):
        return m
    if m.shape == (ny, nx):
        return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
    raise RuntimeError(f"Mask shape {m.shape} incompatible with {(nE, ny, nx)}")


def bunit_str(hdr):
    return str(hdr.get("BUNIT", "")).lower().replace("**", "")

import os
import glob
import numpy as np
from astropy.io import fits


def load_mu_templates_from_fits(
    template_dir,
    labels,
    filename_pattern="{label}.fits",   # or "{label}_template.fits"
    hdu=0,
    dtype=np.float32,
    memmap=True,
    require_same_shape=True,
):
    """
    Load TRUE-counts template cubes (mu) from FITS files into mu_list.

    Assumes each template FITS contains a data cube shaped like:
        (nE, ny, nx)   OR   (nE, npix)
    matching your counts cube.

    Parameters
    ----------
    template_dir : str
        Directory containing FITS templates.
    labels : list[str]
        Component labels, used to build filenames.
    filename_pattern : str
        How to build a filename from a label. Example:
            "{label}.fits"
            "mu_{label}.fits"
            "{label}_mapcube.fits"
    hdu : int
        FITS HDU index holding the data.
    dtype : numpy dtype
        Cast output arrays to this dtype (float32 is usually plenty).
    memmap : bool
        Use FITS memmap to avoid loading everything at once.
    require_same_shape : bool
        If True, raises if templates don't all share the same shape.

    Returns
    -------
    mu_list : list[np.ndarray]
        List of template arrays in counts units.
    headers : list[fits.Header]
        Corresponding FITS headers (for WCS / energy metadata if needed).
    """
    mu_list = []
    headers = []
    shapes = []

    for lab in labels:
        path = os.path.join(template_dir, filename_pattern.format(label=lab))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found for '{lab}': {path}")

        with fits.open(path, memmap=memmap) as hdul:
            data = hdul[hdu].data
            hdr = hdul[hdu].header

        if data is None:
            raise ValueError(f"No data in {path} (HDU {hdu})")

        arr = np.asarray(data, dtype=dtype)

        # Basic sanity: must be at least 2D with energy axis first
        if arr.ndim < 2:
            raise ValueError(f"Template '{lab}' has ndim={arr.ndim}, expected (nE, ...spatial...)")

        mu_list.append(arr)
        headers.append(hdr)
        shapes.append(arr.shape)

    if require_same_shape:
        s0 = shapes[0]
        for lab, s in zip(labels, shapes):
            if s != s0:
                raise ValueError(f"Shape mismatch: '{labels[0]}' {s0} vs '{lab}' {s}")

    return mu_list, headers


def load_mu_templates_by_glob(
    template_dir,
    glob_pattern="*.fits",
    sort_key=None,
    hdu=0,
    dtype=np.float32,
    memmap=True,
):
    """
    Load all FITS templates matching a glob. Labels inferred from filenames.

    Example:
        mu_list, labels, headers = load_mu_templates_by_glob("templates/", "mu_*.fits")

    Labels are inferred as the basename without extension.
    """
    paths = glob.glob(os.path.join(template_dir, glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No FITS files found: {template_dir}/{glob_pattern}")

    if sort_key is None:
        paths = sorted(paths)
    else:
        paths = sorted(paths, key=sort_key)

    labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    mu_list = []
    headers = []
    shapes = []

    for path in paths:
        with fits.open(path, memmap=memmap) as hdul:
            data = hdul[hdu].data
            hdr = hdul[hdu].header

        if data is None:
            raise ValueError(f"No data in {path} (HDU {hdu})")

        arr = np.asarray(data, dtype=dtype)
        mu_list.append(arr)
        headers.append(hdr)
        shapes.append(arr.shape)

    # Enforce same shape
    s0 = shapes[0]
    for lab, s in zip(labels, shapes):
        if s != s0:
            raise ValueError(f"Shape mismatch: '{labels[0]}' {s0} vs '{lab}' {s}")

    return mu_list, labels, headers


def assert_templates_match_counts(counts, mu_list, labels=None):
    """
    Quick check: every template has same shape as counts.
    """
    counts = np.asarray(counts)
    for j, mu in enumerate(mu_list):
        mu = np.asarray(mu)
        if mu.shape != counts.shape:
            lab = labels[j] if labels is not None else f"template[{j}]"
            raise ValueError(f"{lab} shape {mu.shape} does not match counts shape {counts.shape}")
    return True


# -------------------------
# Spectra (counts-units only)
# -------------------------
def data_E2_spectrum_counts(counts, expo, omega, dE_mev, Ectr_mev, mask3d):
    """
    ROI-averaged data spectrum in E^2 dN/dE using a counts-space estimator.

    For each energy bin k:
        C_k = sum_{ROI} counts[k]
        D_k = sum_{ROI} expo[k] * omega * dE_k
        I_k = C_k / D_k               [ph / (cm^2 s sr MeV)]
        E2I = Ectr_k^2 * I_k

    Poisson error:
        sigma(I_k) = sqrt(C_k) / D_k
        sigma(E2I) = Ectr_k^2 * sigma(I_k)

    Inputs must be aligned in shape:
      counts, expo, mask3d: (nE, ...spatial...)
      omega: (...spatial...)
      dE_mev, Ectr_mev: (nE,)
    """
    counts = np.asarray(counts, float)
    expo   = np.asarray(expo, float)
    omega  = np.asarray(omega, float)
    dE_mev = np.asarray(dE_mev, float)
    Ectr   = np.asarray(Ectr_mev, float)
    mask3d = np.asarray(mask3d, bool)

    if counts.shape != expo.shape or counts.shape != mask3d.shape:
        raise ValueError("counts, expo, mask3d must have identical shapes (nE, ...spatial...)")
    if omega.shape != counts.shape[1:]:
        raise ValueError("omega must have spatial shape matching counts[0]")
    if dE_mev.shape[0] != counts.shape[0] or Ectr.shape[0] != counts.shape[0]:
        raise ValueError("dE_mev and Ectr_mev must have length nE")

    nE = counts.shape[0]
    y = np.full(nE, np.nan)
    yerr = np.full(nE, np.nan)

    for k in range(nE):
        m = mask3d[k]
        if not np.any(m):
            continue

        C = float(np.nansum(counts[k][m]))
        D = float(np.nansum((expo[k] * omega * dE_mev[k])[m]))
        if not np.isfinite(D) or D <= 0:
            continue

        I = C / D
        Ierr = (np.sqrt(C) / D) if C > 0 else 0.0

        y[k] = I * (Ectr[k] ** 2)
        yerr[k] = Ierr * (Ectr[k] ** 2)

    return y, yerr


def model_E2_spectrum_from_cells_counts(
    coeff_cells,
    cells,
    templates,
    expo,
    omega,
    dE_mev,
    Ectr_mev,
    mask3d,
):
    """
    ROI-averaged model spectrum in E^2 dN/dE from cellwise fitted coefficients,
    assuming TRUE COUNTS templates with NO cell normalisation.

    Model counts in ROI for bin k:
        C_j,k = sum_cells  a_{cell,k,j} * sum_{ROI∩cell} T_{j,k}
        C_tot,k = sum_j C_j,k

    Then:
        I_k = C_tot,k / D_k, where D_k = sum_{ROI} expo*omega*dE
        E2I_k = Ectr_k^2 * I_k

    Returns
    -------
    y_comp : (nComp, nE) array
        Component-wise E^2 I spectra.
    y_model : (nE,) array
        Total E^2 I spectrum (sum over components).
    """
    coeff_cells = np.asarray(coeff_cells, float)
    expo   = np.asarray(expo, float)
    omega  = np.asarray(omega, float)
    dE_mev = np.asarray(dE_mev, float)
    Ectr   = np.asarray(Ectr_mev, float)
    mask3d = np.asarray(mask3d, bool)

    if not isinstance(templates, (list, tuple)) or len(templates) == 0:
        raise ValueError("templates must be a non-empty list of arrays")
    templates = [np.asarray(t, float) for t in templates]

    nComp = len(templates)
    nE = templates[0].shape[0]

    # Shape checks
    if any(t.shape != templates[0].shape for t in templates):
        raise ValueError("all templates must have the same shape (nE, ...spatial...)")
    if expo.shape != templates[0].shape or mask3d.shape != templates[0].shape:
        raise ValueError("expo and mask3d must match template shape (nE, ...spatial...)")
    if omega.shape != templates[0].shape[1:]:
        raise ValueError("omega must match spatial shape of templates[0]")
    if dE_mev.shape[0] != nE or Ectr.shape[0] != nE:
        raise ValueError("dE_mev and Ectr_mev must have length nE")

    if coeff_cells.ndim != 3 or coeff_cells.shape[1] != nE or coeff_cells.shape[2] != nComp:
        raise ValueError(f"coeff_cells must have shape (nCells, nE, nComp)=(*,{nE},{nComp}), got {coeff_cells.shape}")
    if len(cells) != coeff_cells.shape[0]:
        raise ValueError("len(cells) must equal coeff_cells.shape[0]")

    y_comp = np.full((nComp, nE), np.nan)
    y_model = np.full(nE, np.nan)

    for k in range(nE):
        m_roi = mask3d[k]
        if not np.any(m_roi):
            continue

        D = float(np.nansum((expo[k] * omega * dE_mev[k])[m_roi]))
        if not np.isfinite(D) or D <= 0:
            continue

        Cj = np.zeros(nComp, dtype=float)

        # accumulate counts per component from each cell
        for ci, cell2d in enumerate(cells):
            cm = m_roi & cell2d
            if not np.any(cm):
                continue

            a = coeff_cells[ci, k, :]  # (nComp,)
            if not np.all(np.isfinite(a)):
                continue

            # Sum template counts inside this (ROI ∩ cell)
            # then multiply by the cell coefficient
            for j in range(nComp):
                s = float(np.nansum(templates[j][k][cm]))
                if np.isfinite(s) and s != 0.0:
                    Cj[j] += a[j] * s

        Ctot = float(np.sum(Cj))

        y_model[k] = (Ctot / D) * (Ectr[k] ** 2)
        for j in range(nComp):
            y_comp[j, k] = (Cj[j] / D) * (Ectr[k] ** 2)

    return y_comp, y_model

def raw_counts_spectrum(counts, mask3d):
    nE = counts.shape[0]
    C = np.full(nE, np.nan)
    Cerr = np.full(nE, np.nan)
    for k in range(nE):
        ck = np.nansum(counts[k][mask3d[k]])
        C[k] = ck
        Cerr[k] = np.sqrt(ck) if ck >= 0 else np.nan
    return C, Cerr


def omega_weighted_mean(map2d, omega, mask2d):
    w = omega * mask2d
    wsum = np.nansum(w)
    if wsum <= 0:
        return np.nan
    return np.nansum(map2d * w) / wsum


def fit_per_bin_weighted_nnls(counts, mu_list, mask3d):
    """
    Per-energy-bin weighted NNLS:
      counts[k] ~ sum_j a_kj * mu_j[k]
    Returns coeff[nE, ncomp]
    """
    nE = counts.shape[0]
    ncomp = len(mu_list)
    coeff = np.zeros((nE, ncomp), float)

    for k in range(nE):
        m = mask3d[k]
        y = counts[k][m].ravel()
        if y.size == 0:
            continue

        X = np.vstack([mu[k][m].ravel() for mu in mu_list]).T  # (Np, ncomp)

        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        y = y[good]
        X = X[good]
        if y.size == 0:
            continue

        # Poisson-ish weights: downweight noisy high-count pixels slightly
        w = 1.0 / np.maximum(y, 1.0)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw

        if HAVE_NNLS:
            c, _ = nnls(Xw, yw)
        else:
            c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            c = np.clip(c, 0.0, None)

        coeff[k] = c

    return coeff

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
        "--ext-mask",
        default=os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits"),
        help="Extended-source keep mask FITS (True=keep, False=masked). Applied before fitting components.",
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

    # -------------------------
    # (B) Fit + Totani Fig.2/Fig.3-style plots
    #   Fig 2: |b| >= disk_cut
    #   Fig 3: |b| <  disk_cut
    # Both use the same templates + optional source mask
    # -------------------------

    # Optional source/extended mask (True=keep)
    if args.mask_fit is not None:
        srcmask = load_mask_any_shape(args.mask_fit, nE, ny, nx)
    else:
        srcmask = np.ones((nE, ny, nx), bool)

    # Extended sources mask is applied BEFORE fitting any components
    if args.ext_mask is not None and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), nE, ny, nx)
        srcmask = srcmask & ext_keep3d
        frac_masked = float(np.mean((~ext_keep3d)[:, roi2d])) if np.any(roi2d) else float("nan")
        print(f"[ext-mask] applying extended-source mask: {args.ext_mask} (masked frac in ROI={frac_masked:.4f})")
    else:
        print("[ext-mask] no extended-source mask applied")

    # Resolve templates automatically
    tpl = resolve_templates(args.templates_dir)

    print("[templates] dir:", args.templates_dir)
    for k in ["GAS", "ICS", "ISO", "PS", "LOOPI", "FB_FLAT", "BUB_POS", "BUB_NEG", "NFW"]:
        if k in tpl:
            print(f"[templates] {k}: {tpl[k]}")
    labels = ["GAS", "ICS", "ISO", "PS", "LOOPI", "FB_FLAT", "NFW"]

    # Load templates
    labels = ["gas", "iso", "ps", "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno", "loopI", "ics", "fb_flat"]

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

    res_fit = fit_cellwise_poisson_mle_counts(
        counts=counts,
        templates=mu_list,
        mask3d=fit_mask3d,
        lon=lon,
        lat=lat,
        roi_lon=float(args.roi_lon),
        roi_lat=float(args.roi_lat),
        cell_deg=float(args.cell_deg),
        nonneg=True,
    )
    cells = res_fit["cells"]
    coeff_cells = res_fit["coeff_cells"]
    info = res_fit["info"]


    # Fig 2 outputs
    out_fig2 = os.path.join(args.outdir, "totani_fig2_fit_components.png")
    out_c2   = os.path.join(args.outdir, "fit_coefficients_fig2_highlat.txt")

    run_fit_and_plot(
        region_name=rf"|l|≤{args.roi_lon}°\n|b|≤{args.roi_lat}°",
        region2d=fig2_region2d,
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

    # Fig 3 outputs
    out_fig3 = os.path.join(args.outdir, "totani_fig3_fit_components.png")
    out_c3   = os.path.join(args.outdir, "fit_coefficients_fig3_disk.txt")

    run_fit_and_plot(
        region_name=rf"|l|≤{args.roi_lon}°\n{disk_cut}°≤|b|≤{args.roi_lat}°\n(fit including |b|<{disk_cut}°)",
        region2d=fig3_region2d,
        out_png=out_fig3,
        out_coeff_txt=out_c3,
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


if __name__ == "__main__":
    main()