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

try:
    from scipy.optimize import nnls
    HAVE_NNLS = True
except Exception:
    HAVE_NNLS = False


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

    # Fit mask is ALWAYS full ROI including disk, with any source masks applied
    fit_mask3d = roi2d[None, :, :] & ext_mask3d & ps_mask3d

    # Templates
    # (keep your preferred filenames, but do NOT renormalize)
    template_files = {
        "GAS":      "gas_dnde.fits",
        "ICS":      "ics_dnde.fits",
        "ISO":      "iso_E2dnde.fits",
        "PS":       "ps_E2dnde.fits",
        "LOOPI":    "loopI_E2dnde.fits",
        "FB_FLAT":  "bubbles_flat_binary_E2dnde.fits",
        "NFW":      "nfw_rho2.5_g1.25_E2dnde.fits",
    }

    labels = []
    mu_list = []

    print("\n[templates]")
    for lab, fn in template_files.items():
        path = os.path.join(args.templates_dir, fn)
        if not os.path.exists(path):
            print(f"  - missing: {lab} -> {fn}")
            continue
        cube, _, bunit = load_cube(path, expected_shape=(nE, ny, nx))
        mu = template_to_mu(cube, bunit, Ectr_mev, dE_mev, expo, omega)
        labels.append(lab)
        mu_list.append(mu)
        print(f"  ✓ {lab:7s} {fn:35s}  BUNIT='{bunit}'")

    if len(mu_list) == 0:
        raise RuntimeError("No templates were loaded.")

    nComp = len(mu_list)
    print(f"\nLoaded {nComp} templates: {labels}")

    # Fit
    print(f"\n[fit] mode={args.fit_mode}")
    if args.fit_mode == "global":
        coeff = fit_nnls_global(counts, mu_list, fit_mask3d, verbose=args.verbose)
        fit_kind = "global"
        cells = None
        coeff_for_spectra = coeff
        print(f"coeff shape: {coeff.shape}")
    else:
        cells, coeff_cells = fit_nnls_cellwise(
            counts, mu_list, fit_mask3d, lon, lat, args.roi_lon, args.roi_lat,
            cell_deg=args.cell_deg, verbose=args.verbose
        )
        fit_kind = "cellwise"
        coeff_for_spectra = coeff_cells
        print(f"nCells: {len(cells)}   coeff shape: {coeff_cells.shape}")

    # Plot masks (Totani definitions)
    # Fig 2 plot includes disk
    fig2_plot_mask3d = roi2d[None, :, :] & ext_mask3d & ps_mask3d
    # Fig 3 plot excludes disk
    fig3_plot2d = roi2d & (np.abs(lat) >= args.disk_cut)
    fig3_plot_mask3d = fig3_plot2d[None, :, :] & ext_mask3d & ps_mask3d

    # Spectra (use SAME fitted coeffs for both!)
    print("\n[spectra] computing Fig 2 (plot incl. disk)")
    data2, data2e, comp2, model2 = compute_E2_spectra_from_fit(
        counts, mu_list, labels, expo, omega, dE_mev, Ectr_mev, fig2_plot_mask3d,
        fit_kind, coeff_for_spectra, cells=cells
    )

    print("[spectra] computing Fig 3 (plot excl. disk)")
    data3, data3e, comp3, model3 = compute_E2_spectra_from_fit(
        counts, mu_list, labels, expo, omega, dE_mev, Ectr_mev, fig3_plot_mask3d,
        fit_kind, coeff_for_spectra, cells=cells
    )

    # Plots
    fig2_title = f"Totani 2025 Figure 2 (fit incl. disk; plot incl. disk)\n|l| ≤ {args.roi_lon}°, |b| ≤ {args.roi_lat}°"
    fig3_title = f"Totani 2025 Figure 3 (fit incl. disk; plot excl. disk)\n|l| ≤ {args.roi_lon}°, {args.disk_cut}° ≤ |b| ≤ {args.roi_lat}°"

    fig2_path = os.path.join(args.outdir, "totani_fig2_reproduced.png")
    fig3_path = os.path.join(args.outdir, "totani_fig3_reproduced.png")

    plot_fig(Ectr_gev, data2, data2e, comp2, model2, labels, fig2_title, fig2_path, plot_style=args.plot_style)
    plot_fig(Ectr_gev, data3, data3e, comp3, model3, labels, fig3_title, fig3_path, plot_style=args.plot_style)

    # Save coefficients (optional summary)
    coeff_path = os.path.join(args.outdir, "fitted_coefficients.txt")
    with open(coeff_path, "w") as f:
        f.write("# NNLS coefficients per energy bin\n")
        f.write("# Fit region: full ROI incl disk, with masks applied\n")
        f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
        if fit_kind == "global":
            for k in range(nE):
                f.write(f"{k:02d}  {Ectr_gev[k]:.6g}  " + "  ".join(f"{coeff[k,j]:.6e}" for j in range(nComp)) + "\n")
        else:
            # cellwise: write pixel-weighted mean coeff across cells for convenience
            for k in range(nE):
                wsum = 0.0
                csum = np.zeros(nComp, float)
                for ci, cell2d in enumerate(cells):
                    cm = fit_mask3d[k] & cell2d
                    npx = int(np.count_nonzero(cm))
                    if npx <= 0:
                        continue
                    wsum += npx
                    csum += npx * coeff_for_spectra[ci, k]
                cmean = csum / wsum if wsum > 0 else np.zeros(nComp, float)
                f.write(f"{k:02d}  {Ectr_gev[k]:.6g}  " + "  ".join(f"{cmean[j]:.6e}" for j in range(nComp)) + "\n")

    print(f"\n✓ Saved coeff summary: {coeff_path}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
