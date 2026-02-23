#!/usr/bin/env python3
"""
Reproduce Totani 2025 Figures 2 and 3 (spectra of best-fit templates).

Totani definitions (important):
- Fit is performed INCLUDING the Galactic disk (|b| < 10° is INCLUDED in the FIT).
- Figure 2: plotted mean background flux within ROI INCLUDING the disk.
- Figure 3: SAME fit as Fig 2, but plotted mean background flux EXCLUDING the disk (|b| < 10° removed).

This script:
- Loads counts cube + exposure cube
- Loads template cubes (either COUNTS or INTENSITY-like)
- Converts templates to expected COUNTS per bin (mu_j[k,y,x]) for coefficient=1
- Fits NNLS per energy bin (global or cellwise)
- Converts fitted component counts back to ROI-averaged intensity and plots E^2 dN/dE

Notes:
- Do NOT renormalize templates arbitrarily. That destroys the physical meaning of coefficients.
- If you want Totani-style visuals, set --plot-style totani (no Data/model/residual).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# your repo helper imports (as in your original)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts
from totani_helpers.fit_utils import build_fit_mask3d, component_counts_from_cellwise_fit

try:
    from scipy.optimize import nnls
    HAVE_NNLS = True
except Exception:
    HAVE_NNLS = False


def _load_mcmc_coeffs_per_energy(*, mcmc_dir: str, nE: int, stat: str = "f_ml"):
    coeffs_by_name: dict[str, np.ndarray] = {}
    labels_ref: list[str] | None = None

    for k in range(int(nE)):
        path = os.path.join(str(mcmc_dir), f"mcmc_results_k{k:02d}.npz")
        if not os.path.exists(path):
            continue

        npz = np.load(path, allow_pickle=True)
        labels = npz["labels"].tolist()
        labels = [str(x) for x in labels]
        if labels_ref is None:
            labels_ref = labels
        else:
            if labels != labels_ref:
                raise RuntimeError(
                    "MCMC output labels differ across energy bins; cannot build a consistent spectrum."
                )

        if stat not in npz:
            raise KeyError(f"MCMC file {path} missing key '{stat}'.")
        f = np.asarray(npz[stat], float).reshape(-1)
        if f.shape[0] != len(labels):
            raise RuntimeError(f"MCMC file {path} has {f.shape[0]} coeffs but {len(labels)} labels")

        for lab, val in zip(labels, f):
            if lab not in coeffs_by_name:
                coeffs_by_name[lab] = np.full(int(nE), np.nan, dtype=float)
            coeffs_by_name[lab][k] = float(val)

    return coeffs_by_name


def _pick_coeff_for_template_key(*, coeffs_by_label: dict[str, np.ndarray], template_key: str):
    if template_key in coeffs_by_label:
        return coeffs_by_label[template_key]

    if template_key == "nfw":
        nfw_keys = [k for k in coeffs_by_label.keys() if str(k).lower().startswith("nfw_")]
        if len(nfw_keys) == 1:
            return coeffs_by_label[nfw_keys[0]]
        if len(nfw_keys) > 1:
            raise RuntimeError(
                "Multiple MCMC labels start with 'nfw_'. Cannot map template key 'nfw' uniquely."
            )
        raise KeyError("Could not find an MCMC coefficient for template key 'nfw'.")

    raise KeyError(f"Could not find an MCMC coefficient for template '{template_key}'.")

def normalise_templates_counts(templates_counts: dict, fit_mask: np.ndarray):
    """
    For each template name, compute norm[k] = sum(template[k] in fit_mask[k] (if 3D) or fit_mask (if 2D))
    fit_mask may be 2D (ny,nx) or 3D (nE,ny,nx).

    Return:
      templates_norm[name][k] = template[name][k] / norm[k]
      norms[name] = array shape (nE,)
    """
    names = list(templates_counts.keys())
    nE = next(iter(templates_counts.values())).shape[0]

    norms = {}
    templates_norm = {}

    fit_mask = np.asarray(fit_mask, dtype=bool)
    for name in names:
        T = np.asarray(templates_counts[name], float)
        if T.ndim != 3:
            raise ValueError(f"{name}: expected 3D template (nE,ny,nx), got {T.shape}")
        norm = np.zeros(nE, dtype=float)
        Tn = T.copy()

        for k in range(nE):
            mk = fit_mask[k] if fit_mask.ndim == 3 else fit_mask
            s = float(np.nansum(T[k][mk]))
            norm[k] = s
            if s > 0:
                Tn[k] /= s
            else:
                Tn[k] *= 0.0  # empty template in fit region

        norms[name] = norm
        templates_norm[name] = Tn

    return templates_norm, norms

from scipy.optimize import nnls

def nnls_fit_per_energy(counts, templates_norm, fit_mask, eps=1.0):
    """
    Fit counts[k] ≈ Σ_i a_i[k] * templates_norm_i[k]  (NNLS)
    Returns:
      coeffs_norm[name] = array (nE,)
      model_norm = array (nE,ny,nx) in NORMALISED-template units
    """
    names = list(templates_norm.keys())
    nE, ny, nx = counts.shape

    coeffs = {n: np.zeros(nE, dtype=float) for n in names}
    model_norm = np.zeros((nE, ny, nx), dtype=float)

    fit_mask = np.asarray(fit_mask, dtype=bool)

    for k in range(nE):
        mk = (fit_mask[k] if fit_mask.ndim == 3 else fit_mask) & np.isfinite(counts[k])
        if mk.sum() < 10:
            continue
        y = counts[k][mk].astype(float)

        # optional Poisson weighting (recommended)
        w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
        yw = y * w

        A = np.vstack([templates_norm[n][k][mk].astype(float) for n in names]).T
        Aw = A * w[:, None]

        x, _ = nnls(Aw, yw)

        for ci, n in enumerate(names):
            coeffs[n][k] = x[ci]

        # model in normalised template space
        # (still useful; convert back to counts later)
        Mk = np.zeros((ny, nx), dtype=float)
        for ci, n in enumerate(names):
            Mk += x[ci] * templates_norm[n][k]
        model_norm[k] = Mk

    return coeffs, model_norm


def _iter_cells(lon_deg_2d, lat_deg_2d, *, roi_lon, roi_lat, cell_deg):
    """Yield (cell_name, cell_mask_2d) for a regular lon/lat grid inside ROI."""
    lon_deg_2d = np.asarray(lon_deg_2d, float)
    lat_deg_2d = np.asarray(lat_deg_2d, float)

    lon_edges = np.arange(-float(roi_lon), float(roi_lon) + float(cell_deg), float(cell_deg))
    lat_edges = np.arange(-float(roi_lat), float(roi_lat) + float(cell_deg), float(cell_deg))

    for i in range(len(lon_edges) - 1):
        l0, l1 = lon_edges[i], lon_edges[i + 1]
        for j in range(len(lat_edges) - 1):
            b0, b1 = lat_edges[j], lat_edges[j + 1]

            # include upper edge for last bin to avoid gaps
            if i == len(lon_edges) - 2:
                m_lon = (lon_deg_2d >= l0) & (lon_deg_2d <= l1)
            else:
                m_lon = (lon_deg_2d >= l0) & (lon_deg_2d < l1)

            if j == len(lat_edges) - 2:
                m_lat = (lat_deg_2d >= b0) & (lat_deg_2d <= b1)
            else:
                m_lat = (lat_deg_2d >= b0) & (lat_deg_2d < b1)

            cell = m_lon & m_lat
            name = f"lon[{l0:g},{l1:g}]_lat[{b0:g},{b1:g}]"
            yield name, cell


def build_model_counts_cellwise(
    *,
    counts_fit,
    templates_counts,
    norms_global,
    fit_mask3d,
    lon,
    lat,
    roi_lon,
    roi_lat,
    cell_deg,
    eps=1.0,
    min_pix=10,
):
    """Cellwise per-energy NNLS. Returns dict name->(nE,ny,nx) component model counts and total model counts."""
    names = list(templates_counts.keys())
    nE, ny, nx = counts_fit.shape

    # Precompute normalised templates using GLOBAL norms (stable) then fit per-cell.
    templates_norm = {}
    for name in names:
        T = np.asarray(templates_counts[name], float)
        nrm = np.asarray(norms_global[name], float)
        Tn = np.zeros_like(T, dtype=float)
        good = nrm > 0
        Tn[good] = T[good] / nrm[good][:, None, None]
        templates_norm[name] = Tn

    comp_counts = {name: np.zeros((nE, ny, nx), dtype=float) for name in names}
    model_total = np.zeros((nE, ny, nx), dtype=float)

    for _cell_name, cell2d in _iter_cells(lon, lat, roi_lon=roi_lon, roi_lat=roi_lat, cell_deg=cell_deg):
        cell2d = np.asarray(cell2d, dtype=bool)
        if not np.any(cell2d):
            continue

        for k in range(nE):
            mk = fit_mask3d[k] & cell2d & np.isfinite(counts_fit[k])
            if mk.sum() < min_pix:
                continue

            y = counts_fit[k][mk].astype(float)
            w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
            yw = y * w

            A = np.vstack([templates_norm[n][k][mk].astype(float) for n in names]).T
            Aw = A * w[:, None]
            x, _ = nnls(Aw, yw)

            # Convert to physical coeffs in counts space: a_phys = a_norm / norm
            for ci, n in enumerate(names):
                nrm = float(norms_global[n][k])
                if nrm <= 0:
                    continue
                a_phys = float(x[ci]) / nrm
                Tk = np.asarray(templates_counts[n][k], float)
                comp_counts[n][k][cell2d] += a_phys * Tk[cell2d]
                model_total[k][cell2d] += a_phys * Tk[cell2d]

    return comp_counts, model_total

def build_model_counts_from_norm(counts_shape, templates_counts, coeffs_norm, norms):
    nE, ny, nx = counts_shape
    model_counts = np.zeros((nE, ny, nx), dtype=float)

    for name, T in templates_counts.items():
        a_norm = np.asarray(coeffs_norm[name], float)      # (nE,)
        norm = np.asarray(norms[name], float)              # (nE,)
        a_phys = np.zeros_like(a_norm)
        good = norm > 0
        a_phys[good] = a_norm[good] / norm[good]

        model_counts += a_phys[:, None, None] * T

    return model_counts

import numpy as np
import matplotlib.pyplot as plt

def mean_E2_dnde_background(
    *,
    Ectr_mev,          # (nE,) bin centers in MeV
    dE_mev,            # (nE,) bin widths in MeV
    expo,              # (nE,ny,nx) exposure in cm^2 s
    omega,             # (ny,nx) pixel solid angle in sr
    mask2d,            # (ny,nx) True=keep pixels for averaging
    model_counts_bkg,  # (nE,ny,nx) background model counts (NOT including bubbles)
):
    E = np.asarray(Ectr_mev, float)
    dE = np.asarray(dE_mev, float)
    if dE.ndim == 0:
        dE = np.full_like(E, float(dE))

    nE, ny, nx = model_counts_bkg.shape
    assert expo.shape == (nE, ny, nx)
    assert omega.shape == (ny, nx)
    assert mask2d.shape == (ny, nx)
    assert E.shape == (nE,)
    assert dE.shape == (nE,)

    # denom (nE,ny,nx)
    denom = expo * omega[None, :, :] * dE[:, None, None]

    mean_dnde = np.full(nE, np.nan, dtype=float)

    for k in range(nE):
        good = mask2d & np.isfinite(denom[k]) & (denom[k] > 0) & np.isfinite(model_counts_bkg[k])
        if not np.any(good):
            continue
        dnde_map = model_counts_bkg[k][good] / denom[k][good]  # ph cm^-2 s^-1 sr^-1 MeV^-1
        mean_dnde[k] = float(np.mean(dnde_map))

    # E^2 dN/dE in MeV units: MeV^2 * (ph cm^-2 s^-1 sr^-1 MeV^-1) = MeV * ph cm^-2 s^-1 sr^-1
    E2_dnde_mev = (E**2) * mean_dnde

    return mean_dnde, E2_dnde_mev


def plot_E2_dnde(Ectr_mev, E2_dnde_mev, *, out_png=None, label="Background"):
    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0
    y = np.asarray(E2_dnde_mev, float)

    m = np.isfinite(Ectr_gev) & np.isfinite(y) & (Ectr_gev > 0)
    Ectr_gev = Ectr_gev[m]
    y = y[m]

    plt.figure(figsize=(7,5))
    plt.plot(Ectr_gev, y, marker="o", label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"$E^2 \,\langle \mathrm{d}N/\mathrm{d}E \rangle$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
        plt.close()
    else:
        plt.show()


def _build_counts_cubes_from_coeffs(*, templates_counts: dict, coeffs_by_label: dict[str, np.ndarray]):
    names = list(templates_counts.keys())
    first = templates_counts[names[0]]
    nE, ny, nx = first.shape

    comp_counts = {}
    model_total = np.zeros((nE, ny, nx), dtype=float)

    for name in names:
        T = np.asarray(templates_counts[name], float)
        a = _pick_coeff_for_template_key(coeffs_by_label=coeffs_by_label, template_key=name)
        a = np.asarray(a, float).reshape(-1)
        if a.shape[0] != nE:
            raise RuntimeError(f"Coeff for '{name}' has length {a.shape[0]} but nE={nE}")

        cube = a[:, None, None] * T
        comp_counts[name] = cube
        model_total += cube

    return comp_counts, model_total


def plot_E2_dnde_multi(Ectr_mev, curves, *, out_png=None, title=None):
    """curves: list of (label, E2_dnde_mev_array)."""
    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0

    plt.figure(figsize=(8, 6))
    for lab, y_in in curves:
        y = np.asarray(y_in, float)
        m = np.isfinite(Ectr_gev) & np.isfinite(y) & (Ectr_gev > 0)
        if not np.any(m):
            continue
        plt.plot(Ectr_gev[m], y[m], marker="o", label=str(lab))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"$E^2 \,\langle \mathrm{d}N/\mathrm{d}E \rangle$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    if title is not None:
        plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
        plt.close()
    else:
        plt.show()

# ----------------------------
# FITS + units helpers
# ----------------------------
def load_cube(path, expected_shape=None):
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header
        bunit = str(hdr.get("BUNIT", "")).strip()
    if expected_shape is not None and data.shape != expected_shape:
        raise ValueError(f"{os.path.basename(path)} shape {data.shape} != expected {expected_shape}")
    return data, hdr, bunit


def _parse_energy_unit(bunit: str):
    """Return 'mev', 'gev', or None."""
    bu = (bunit or "").lower().replace(" ", "")
    if "mev" in bu:
        return "mev"
    if "gev" in bu:
        return "gev"
    return None


def template_to_mu(template, bunit, Ectr_mev, dE_mev, expo, omega):
    """
    Convert template cube -> expected counts cube mu for coefficient = 1.

    Accepted template meanings:
    1) COUNTS templates:
       BUNIT contains 'count' or 'counts' -> already expected counts per bin.

    2) INTENSITY templates in one of these forms:
       - dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]  (or GeV^-1)
       - E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]    (or GeV cm^-2 s^-1 sr^-1)

    We infer E^2 dN/dE vs dN/dE by presence of energy^-1 in the unit string:
       if 'mev-1' or 'gev-1' present -> dN/dE
       else if looks like intensity (cm^-2 s^-1 sr^-1) -> assume E^2 dN/dE
    """
    bu = (bunit or "").lower().replace(" ", "").replace("**", "")

    # Case 1: counts-like
    if ("count" in bu) or ("counts" in bu):
        return template

    # Identify intensity-like
    is_intensity_like = ("cm-2" in bu) and ("s-1" in bu) and ("sr-1" in bu)
    has_per_energy = ("mev-1" in bu) or ("gev-1" in bu)

    if not is_intensity_like:
        # fall back: assume template is dN/dE in MeV^-1 if unknown
        has_per_energy = True

    eunit = _parse_energy_unit(bunit)
    if eunit is None:
        # default to MeV if not stated
        eunit = "mev"

    if eunit == "gev":
        # convert E and dE to GeV for unit-consistency if the template is in GeV units
        Ectr = Ectr_mev / 1e3
        dE = dE_mev / 1e3
    else:
        Ectr = Ectr_mev
        dE = dE_mev

    if has_per_energy:
        # template is dN/dE in (energy)^-1
        dnde = template
    else:
        # template is E^2 dN/dE in (energy) * (cm^-2 s^-1 sr^-1)
        # convert to dN/dE
        # guard against E=0
        dnde = template / (Ectr[:, None, None] ** 2)

    mu = dnde * expo * omega[None, :, :] * dE[:, None, None]
    return mu


# ----------------------------
# NNLS fitting
# ----------------------------
def fit_nnls_global(counts, mu_list, mask3d, verbose=False):
    """
    Global NNLS per energy bin over all masked pixels in the ROI.
    Returns coeff array: (nE, nComp)
    """
    nE = counts.shape[0]
    nComp = len(mu_list)
    coeff = np.zeros((nE, nComp), dtype=float)

    for k in range(nE):
        m = mask3d[k]
        y = counts[k][m].ravel()
        if y.size == 0:
            continue

        X = np.vstack([mu[k][m].ravel() for mu in mu_list]).T

        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        y = y[good]
        X = X[good]
        if y.size == 0:
            continue

        if HAVE_NNLS:
            c, res = nnls(X, y)
        else:
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            c = np.clip(c, 0.0, None)
            res = np.linalg.norm(y - X @ c)

        coeff[k] = c
        if verbose and k < 3:
            nz = np.sum(c > 1e-12)
            print(f"[global] bin {k}: nonzero={nz}/{nComp}, resid={res:.3e}")

    return coeff


def fit_nnls_cellwise(counts, mu_list, mask3d, lon, lat, roi_lon, roi_lat, cell_deg=10.0, verbose=False):
    """
    Cellwise NNLS: divide ROI into cells (in l,b) and fit each cell independently.
    Returns:
        cells: list of 2D boolean masks (ny,nx)
        coeff_cells: (nCells, nE, nComp)
    """
    nE = counts.shape[0]
    nComp = len(mu_list)

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

    coeff_cells = np.zeros((len(cells), nE, nComp), dtype=float)

    for ci, cell2d in enumerate(cells):
        for k in range(nE):
            m = mask3d[k] & cell2d
            y = counts[k][m].ravel()
            if y.size == 0:
                continue

            X = np.vstack([mu[k][m].ravel() for mu in mu_list]).T
            good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            y = y[good]
            X = X[good]
            if y.size == 0:
                continue

            if HAVE_NNLS:
                c, res = nnls(X, y)
            else:
                c, *_ = np.linalg.lstsq(X, y, rcond=None)
                c = np.clip(c, 0.0, None)
                res = np.linalg.norm(y - X @ c)

            coeff_cells[ci, k] = c

            if verbose and ci == 0 and k < 3:
                nz = np.sum(c > 1e-12)
                print(f"[cellwise] cell0 bin{k}: nonzero={nz}/{nComp}, resid={res:.3e}")

    return cells, coeff_cells


# ----------------------------
# Spectra computation
# ----------------------------
def compute_E2_spectra_from_fit(
    counts,
    mu_list,
    labels,
    expo,
    omega,
    dE_mev,
    Ectr_mev,
    plot_mask3d,
    fit_kind,
    coeff,         # global: (nE,nComp) ; cellwise: (nCells,nE,nComp)
    cells=None,    # list of (ny,nx) masks if cellwise
):
    """
    Produce ROI-averaged intensity spectra for:
      - data (and poisson err)
      - each component (from fitted counts)
      - model sum
    in units: E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]

    We compute:
      denom = sum(expo * omega * dE_mev) over plot region
      I = sum(counts) / denom
    then multiply by E^2 (MeV^2) to get E^2 dN/dE.
    """
    nE = counts.shape[0]
    nComp = len(mu_list)

    data_E2 = np.full(nE, np.nan)
    data_E2_err = np.full(nE, np.nan)
    comp_E2 = np.zeros((nComp, nE), dtype=float)
    model_E2 = np.zeros(nE, dtype=float)

    for k in range(nE):
        m = plot_mask3d[k]
        if not np.any(m):
            continue

        Csum = np.sum(counts[k][m])
        denom = np.sum(expo[k][m] * omega[m] * dE_mev[k])

        if denom <= 0 or not np.isfinite(denom):
            continue

        I_data = Csum / denom
        data_E2[k] = I_data * (Ectr_mev[k] ** 2)
        data_E2_err[k] = (np.sqrt(Csum) / denom) * (Ectr_mev[k] ** 2) if Csum >= 0 else np.nan

        # component sums in counts, then divide by same denom
        for j in range(nComp):
            if fit_kind == "global":
                mu_counts = np.sum((coeff[k, j] * mu_list[j][k])[m])
            else:
                # cellwise: sum over cells, each with its own coeff
                mu_counts = 0.0
                for ci, cell2d in enumerate(cells):
                    mm = m & cell2d
                    if np.any(mm):
                        mu_counts += np.sum((coeff[ci, k, j] * mu_list[j][k])[mm])

            I_comp = mu_counts / denom
            comp_E2[j, k] = I_comp * (Ectr_mev[k] ** 2)

        model_E2[k] = np.sum(comp_E2[:, k])

    return data_E2, data_E2_err, comp_E2, model_E2

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

# ----------------------------
# Plotting
# ----------------------------
def plot_fig(
    Ectr_gev,
    data_E2,
    data_E2_err,
    comp_E2,
    model_E2,
    labels,
    title,
    outpath,
    plot_style="diagnostic",
):
    """
    plot_style:
      - 'diagnostic' : shows Data + Model Sum + residual panel (your style)
      - 'totani'     : shows only component spectra (closer to Totani Fig2/3)
    """
    if plot_style == "totani":
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        colors = ["black", "green", "tab:purple", "tab:gray", "tab:orange", "tab:blue", "red", "tab:brown"]
        # plot components
        for j, lab in enumerate(labels):
            c = colors[j % len(colors)]
            ax.plot(Ectr_gev, comp_E2[j], marker="o", ms=4, lw=1.0, label=lab, color=c)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("photon energy [GeV]")
        ax.set_ylabel(r"mean background flux $E^2\,dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {outpath}")
        return

    # diagnostic style (Data + model sum + residual)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    ax1.errorbar(Ectr_gev, data_E2, yerr=data_E2_err, fmt="o", color="black",
                 label="Data", markersize=4, capsize=3, zorder=10)

    colors = ["blue", "green", "orange", "red", "purple", "brown", "pink", "gray"]
    for j, lab in enumerate(labels):
        ax1.plot(Ectr_gev, comp_E2[j], label=lab, color=colors[j % len(colors)], lw=1.5)

    ax1.plot(Ectr_gev, model_E2, "k--", label="Model Sum", lw=2, alpha=0.7)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$E^2\,dN/dE$ [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9, ncol=2)
    ax1.tick_params(labelbottom=False)

    # residual
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = (data_E2 - model_E2) / data_E2 * 100.0
        yerr = data_E2_err / data_E2 * 100.0

    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.errorbar(Ectr_gev, residual, yerr=yerr, fmt="o", color="black", ms=4, capsize=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("Energy [GeV]")
    ax2.set_ylabel("Residual [%]")
    ax2.set_ylim(-50, 50)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {outpath}")


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Reproduce Totani 2025 Figures 2 and 3 (fit incl. disk; plot incl/excl disk)")
    repo_dir = os.environ.get("REPO_PATH", os.path.expanduser("~/Documents/PhD/Year 2/DM_Photon_Scattering"))
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

    p.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    p.add_argument("--expo", default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))
    p.add_argument("--templates-dir", default=os.path.join(data_dir, "processed", "templates"))

    p.add_argument("--ext-mask", default=os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits"),
                   help="Extended source mask (True=keep). Can be (ny,nx) or (nE,ny,nx).")
    p.add_argument("--ps-mask", default=None,
                   help="Point source mask (True=keep). Can be (ny,nx) or (nE,ny,nx).")

    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig2_3"))

    p.add_argument(
        "--use-mcmc",
        action="store_true",
        help="Use saved MCMC outputs (mcmc_results_kXX.npz) to build spectra instead of refitting.",
    )
    p.add_argument(
        "--mcmc-dir",
        default=os.path.join(repo_dir, "Totani_paper_check", "mcmc", "mcmc_results"),
        help="Directory containing mcmc_results_kXX.npz files.",
    )
    p.add_argument(
        "--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"],
        default="f_ml",
        help="Which MCMC summary coefficient to use per bin.",
    )

    p.add_argument("--roi-lon", type=float, default=60.0)
    p.add_argument("--roi-lat", type=float, default=60.0)
    p.add_argument("--disk-cut", type=float, default=10.0, help="Disk cut: exclude |b|<disk_cut in Fig 3 plot")
    p.add_argument("--binsz", type=float, default=0.125)

    p.add_argument("--fit-mode", choices=["global", "cellwise"], default="cellwise")
    p.add_argument("--cell-deg", type=float, default=10.0)
    p.add_argument("--plot-style", choices=["diagnostic", "totani"], default="diagnostic")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    templates_dir = args.templates_dir

    print("=" * 70)
    print("Totani 2025 Fig 2/3 reproduction")
    print("Fit includes disk; Fig2 plot includes disk; Fig3 plot excludes disk")
    print("=" * 70)

    # Load counts + energy bins
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    Ectr_gev = Ectr_mev / 1000.0
    print(f"[counts] {counts.shape}, E: {Ectr_gev[0]:.3g}–{Ectr_gev[-1]:.3g} GeV ({nE} bins)")

    wcs = WCS(hdr).celestial

    # Exposure
    expo_raw, E_expo_mev = read_exposure(args.expo)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != counts.shape:
        raise ValueError(f"Exposure shape {expo.shape} != counts shape {counts.shape}")
    print(f"[expo]   {expo.shape}")

    # Geometry
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    print(f"[roi]   |l|<={args.roi_lon} deg, |b|<={args.roi_lat} deg")

    # Masks
    def _load_mask(path):
        if (path is None) or (not os.path.exists(path)):
            return np.ones((nE, ny, nx), dtype=bool)
        m = fits.getdata(path).astype(bool)
        if m.shape == (ny, nx):
            return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
        if m.shape == (nE, ny, nx):
            return m
        raise ValueError(f"Mask {os.path.basename(path)} has shape {m.shape}, expected {(ny,nx)} or {(nE,ny,nx)}")

    ext_mask3d = _load_mask(args.ext_mask)
    ps_mask3d = _load_mask(args.ps_mask)

    # Fit mask: ROI including disk, apply source masks + data coverage consistently.
    srcmask3d = ext_mask3d & ps_mask3d
    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=srcmask3d,
        counts=counts,
        expo=expo,
    )

    # For any post-fit averaging/plotting that expects a 2D mask, be conservative.
    fit_mask_2d = np.all(fit_mask3d, axis=0)

    labels = [
        "gas",
        "iso",
        "ps",
        "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        "loopI",
        "ics",
        "fb_flat",
    ]

    mu_list, _headers = load_mu_templates_from_fits(
        template_dir=templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )

    # Assemble templates_counts from the loaded FITS cubes.
    # Keep short keys for downstream code.
    templates_counts = {}
    for lab, mu in zip(labels, mu_list):
        key = "nfw" if lab.startswith("nfw_") else lab
        templates_counts[key] = mu

    # Fit (counts-space) using the same cellwise 10° convention as the other figure scripts.
    # This fit does not require template normalisation; coefficients multiply counts templates directly.
    args.fit_mode = "cellwise"
    args.cell_deg = 10.0

    if args.use_mcmc:
        coeffs_by_label = _load_mcmc_coeffs_per_energy(mcmc_dir=args.mcmc_dir, nE=nE, stat=args.mcmc_stat)
        comp_counts_dict, model_counts_total = _build_counts_cubes_from_coeffs(
            templates_counts=templates_counts,
            coeffs_by_label=coeffs_by_label,
        )
        bkg_names = list(templates_counts.keys())

        # Fig2: include disk (use fit mask's conservative 2D mask)
        curves2 = []
        for name in bkg_names:
            _mean_dnde, E2_comp = mean_E2_dnde_background(
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                expo=expo,
                omega=omega,
                mask2d=fit_mask_2d,
                model_counts_bkg=comp_counts_dict[name],
            )
            curves2.append((name, E2_comp))
        _mean_dnde, E2_tot = mean_E2_dnde_background(
            Ectr_mev=Ectr_mev,
            dE_mev=dE_mev,
            expo=expo,
            omega=omega,
            mask2d=fit_mask_2d,
            model_counts_bkg=model_counts_total,
        )
        curves2.append(("total", E2_tot))

        plot_E2_dnde_multi(
            Ectr_mev,
            curves2,
            out_png=os.path.join(args.outdir, "E2_dnde_background_components_mcmc_fig2.png"),
            title=f"MCMC background components ({args.mcmc_stat})",
        )

        # Fig3: exclude disk in the *plot*, keep same fit
        plot_mask2d = fit_mask_2d & (np.abs(lat) >= float(args.disk_cut))
        curves3 = []
        for name in bkg_names:
            _mean_dnde, E2_comp = mean_E2_dnde_background(
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                expo=expo,
                omega=omega,
                mask2d=plot_mask2d,
                model_counts_bkg=comp_counts_dict[name],
            )
            curves3.append((name, E2_comp))
        _mean_dnde, E2_tot3 = mean_E2_dnde_background(
            Ectr_mev=Ectr_mev,
            dE_mev=dE_mev,
            expo=expo,
            omega=omega,
            mask2d=plot_mask2d,
            model_counts_bkg=model_counts_total,
        )
        curves3.append(("total", E2_tot3))

        plot_E2_dnde_multi(
            Ectr_mev,
            curves3,
            out_png=os.path.join(args.outdir, "E2_dnde_background_components_mcmc_fig3.png"),
            title=f"MCMC background components ({args.mcmc_stat}), |b|>={float(args.disk_cut):g} deg",
        )

    else:
        component_order = list(templates_counts.keys())
        res_fit = fit_cellwise_poisson_mle_counts(
            counts=counts,
            templates=templates_counts,
            mask3d=fit_mask3d,
            lon=lon,
            lat=lat,
            roi_lon=float(args.roi_lon),
            roi_lat=float(args.roi_lat),
            cell_deg=float(args.cell_deg),
            component_order=component_order,
            nonneg=True,
        )

        comp_counts_dict, model_counts_total = component_counts_from_cellwise_fit(
            templates_counts=templates_counts,
            res_fit=res_fit,
            mask3d=fit_mask3d,
        )
        resid_counts = np.asarray(counts, float).copy()
        resid_counts[~fit_mask3d] = np.nan
        resid_counts = resid_counts - model_counts_total

        bkg_names = component_order

        # 5) plot each fitted component separately (in counts units -> mean dN/dE)
        curves = []
        for name in bkg_names:
            _mean_dnde, E2_comp = mean_E2_dnde_background(
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                expo=expo,
                omega=omega,
                mask2d=fit_mask_2d,
                model_counts_bkg=comp_counts_dict[name],
            )
            curves.append((name, E2_comp))

        _mean_dnde, E2_tot = mean_E2_dnde_background(
            Ectr_mev=Ectr_mev,
            dE_mev=dE_mev,
            expo=expo,
            omega=omega,
            mask2d=fit_mask_2d,
            model_counts_bkg=model_counts_total,
        )
        curves.append(("total", E2_tot))

        plot_E2_dnde_multi(
            Ectr_mev,
            curves,
            out_png=os.path.join(args.outdir, "E2_dnde_background_components.png"),
            title=f"Fitted background components ({args.fit_mode}, cell={args.cell_deg:g} deg)",
        )



if __name__ == "__main__":
    main()
