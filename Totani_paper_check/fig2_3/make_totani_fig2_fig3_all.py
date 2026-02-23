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
    load_mask_any_shape,
)
from totani_helpers.cellwise_fit import *
from totani_helpers.fit_utils import *

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

def raw_counts_spectrum(counts, mask3d):
    nE = counts.shape[0]
    C = np.full(nE, np.nan)
    Cerr = np.full(nE, np.nan)
    for k in range(nE):
        ck = np.nansum(counts[k][mask3d[k]])
        C[k] = ck
        Cerr[k] = np.sqrt(ck) if ck >= 0 else np.nan
    return C, Cerr



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
def data_E2_spectrum_counts(*, counts, expo, omega, dE_mev, Ectr_mev, mask3d, tiny=1e-30):
    nE = counts.shape[0]
    E2 = np.full(nE, np.nan, float)
    E2err = np.full(nE, np.nan, float)

    for k in range(nE):
        m = mask3d[k]
        if not np.any(m):
            continue

        Omega_reg = float(np.nansum(omega[m]))
        if not np.isfinite(Omega_reg) or Omega_reg <= 0:
            continue

        denom = expo[k][m] * dE_mev[k]
        good = np.isfinite(denom) & (denom > tiny) & np.isfinite(counts[k][m])
        if not np.any(good):
            continue

        N = counts[k][m][good]
        denom = denom[good]

        I_mean = float(np.nansum(N / denom) / Omega_reg)   # ph cm^-2 s^-1 sr^-1 MeV^-1
        E2[k] = I_mean * (Ectr_mev[k] ** 2)

        # 1-sigma Poisson error propagated the same way:
        # Var(sum N/denom) = sum Var(N)/denom^2 = sum N/denom^2 (Poisson)
        var = float(np.nansum(N / (denom**2)) / (Omega_reg**2))
        E2err[k] = np.sqrt(var) * (Ectr_mev[k] ** 2)

    return E2, E2err

def model_E2_spectrum_from_cells_counts(
    *,
    coeff_cells,   # (nCells, nE, nComp)
    cells,         # list of 2D bool masks
    templates,     # list of mu_j counts templates, each (nE,ny,nx)
    expo,
    omega,
    dE_mev,
    Ectr_mev,
    mask3d,
    tiny=1e-30,
):
    nCells, nE, nComp = coeff_cells.shape
    E2_comp = np.full((nComp, nE), np.nan, float)
    E2_model = np.full(nE, np.nan, float)

    for k in range(nE):
        mreg = mask3d[k]
        if not np.any(mreg):
            continue

        Omega_reg = float(np.nansum(omega[mreg]))
        if not np.isfinite(Omega_reg) or Omega_reg <= 0:
            continue

        denom_reg = expo[k] * dE_mev[k]
        good_reg = mreg & np.isfinite(denom_reg) & (denom_reg > tiny)

        # accumulate per-component numerator: sum_p mu_pred/(expo*dE)
        num_comp = np.zeros(nComp, float)

        for ci, cell2d in enumerate(cells):
            cm = good_reg & cell2d
            if not np.any(cm):
                continue

            a = coeff_cells[ci, k, :]  # multipliers for counts templates
            if not np.all(np.isfinite(a)):
                continue

            denom = denom_reg[cm]
            for j in range(nComp):
                mu = templates[j][k][cm]
                num_comp[j] += float(a[j]) * float(np.nansum(mu / denom))

        I_comp = num_comp / Omega_reg
        E2_comp[:, k] = I_comp * (Ectr_mev[k] ** 2)
        E2_model[k] = float(np.nansum(I_comp)) * (Ectr_mev[k] ** 2)

    return E2_comp, E2_model


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

def print_template_sums_vs_data(counts, templates_all, fit_mask2d, k_list=(0,2,6,10,12)):
    names = list(templates_all.keys())
    for k in k_list:
        ysum = float(np.nansum(counts[k][fit_mask2d]))
        print(f"\n[k={k:02d}] data sum in fit mask: {ysum:.3e}")
        for n in names:
            tsum = float(np.nansum(templates_all[n][k][fit_mask2d]))
            ratio = tsum/ysum if ysum>0 else np.nan
            print(f"  {n:12s} template sum={tsum:.3e}  ratio_to_data={ratio:.3e}")

def predicted_component_counts_from_cellfit(*, counts, cells, coeff_cells, labels, templates_all, mask3d):
    nCells, nE, nComp = coeff_cells.shape
    out = {lab: np.zeros(nE, float) for lab in labels}
    out["DATA"] = np.zeros(nE, float)
    out["MODEL"] = np.zeros(nE, float)

    # build templates list in label order
    Tlist = [templates_all[lab] for lab in labels]

    for k in range(nE):
        out["DATA"][k] = float(np.nansum(mask3d[k] * np.nan_to_num(counts[k], nan=0.0)))

        for ci, cell2d in enumerate(cells):
            cm = mask3d[k] & cell2d
            if not np.any(cm):
                continue
            a = coeff_cells[ci, k, :]
            if not np.all(np.isfinite(a)):
                continue

            for j, lab in enumerate(labels):
                s = float(np.nansum(Tlist[j][k][cm]))
                out[lab][k] += float(a[j]) * s

        out["MODEL"][k] = sum(out[lab][k] for lab in labels)

    return out


def print_component_fractions(pred, Ectr_gev=None, k_list=(0,2,6,10,12)):
    labels = [k for k in pred.keys() if k not in ("DATA","MODEL")]
    for k in k_list:
        data = pred["DATA"][k]
        model = pred["MODEL"][k]
        frac = (data-model)/data if data>0 else np.nan
        tagE = f" E={Ectr_gev[k]:.3g} GeV" if Ectr_gev is not None else ""
        print(f"\n[k={k:02d}{tagE}] data={data:.3e} model={model:.3e} resid={(100*frac):+.2f}%")
        for lab in labels:
            f = pred[lab][k]/model if model>0 else np.nan
            print(f"  {lab:12s} pred={pred[lab][k]:.3e}  frac_of_model={f:.3f}")
def cell_component_fractions(*, k, cells, coeff_cells, labels, templates_all, mask3d, top=8):
    nComp = len(labels)
    Tlist = [templates_all[lab] for lab in labels]

    rows = []
    for ci, cell2d in enumerate(cells):
        cm = mask3d[k] & cell2d
        if not np.any(cm):
            continue
        a = coeff_cells[ci, k, :]
        if not np.all(np.isfinite(a)):
            continue

        pred = np.zeros(nComp, float)
        for j in range(nComp):
            s = float(np.nansum(Tlist[j][k][cm]))
            pred[j] = float(a[j]) * s

        model = float(np.sum(pred))
        if model <= 0:
            continue

        fracs = pred / model
        rows.append((ci, model, fracs))

    if not rows:
        print("No non-empty cells.")
        return

    nfw_idx = labels.index([x for x in labels if x.startswith("nfw")][0]) if any(l.startswith("nfw") for l in labels) else None
    fb_idx  = labels.index("fb_flat") if "fb_flat" in labels else None

    # sort by NFW frac descending
    if nfw_idx is not None:
        rows.sort(key=lambda r: r[2][nfw_idx], reverse=True)
        print(f"\nTop {top} cells by NFW fraction at k={k}:")
        for (ci, model, fr) in rows[:top]:
            msg = f"  cell {ci:03d}: model={model:.3e} nfw_frac={fr[nfw_idx]:.3f}"
            if fb_idx is not None:
                msg += f" fb_frac={fr[fb_idx]:.3f}"
            print(msg)
def print_cell_template_correlations(*, k, cell_idx, cells, labels, templates_all, mask3d, max_print=12):
    cell2d = cells[cell_idx]
    m = mask3d[k] & cell2d
    if not np.any(m):
        print("Empty cell mask.")
        return

    X = np.stack([templates_all[lab][k][m].ravel() for lab in labels], axis=1)
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    if X.shape[0] < 5:
        print("Not enough pixels.")
        return

    X = X - X.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(X*X, axis=0))
    denom = np.maximum(denom, 1e-300)
    C = (X.T @ X) / (denom[:,None] * denom[None,:])

    # print strongest correlations (off-diagonal)
    pairs = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            pairs.append((abs(C[i,j]), C[i,j], labels[i], labels[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])

    print(f"\n[k={k}] strongest template correlations in cell {cell_idx}:")
    for aabs, cij, li, lj in pairs[:max_print]:
        print(f"  corr({li},{lj}) = {cij:+.3f}")
def print_fb_support(mu_fb, mask3d, roi2d, lat, k_list=(0,2,6,10,12)):
    for k in k_list:
        m = mask3d[k] & roi2d
        nz = int(np.count_nonzero(mu_fb[k][m] > 0))
        tot = int(np.count_nonzero(m))
        frac = nz / tot if tot>0 else np.nan
        print(f"[k={k:02d}] fb_flat nonzero pixels in fit region: {nz}/{tot} = {frac:.3f}")

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

    if getattr(args, "cell_normalize", False):
        print("[W] --cell-normalize is ignored: cellwise Poisson MLE fits coefficients in the original counts-template units.")

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
        srcmask = load_mask_any_shape(args.mask_fit, counts.shape)
    else:
        srcmask = np.ones((nE, ny, nx), bool)

    # Extended sources mask is applied BEFORE fitting any components
    if args.ext_mask is not None and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), counts.shape)
        srcmask = srcmask & ext_keep3d
        frac_masked = float(np.mean((~ext_keep3d)[:, roi2d])) if np.any(roi2d) else float("nan")
        print(f"[ext-mask] applying extended-source mask: {args.ext_mask} (masked frac in ROI={frac_masked:.4f})")
    else:
        print("[ext-mask] no extended-source mask applied")

    # Load templates (counts-space; coefficients multiply these directly)
    labels = ["gas", "iso", "ps", "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno", "loopI", "ics", "fb_flat"]

    mu_list, headers = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",  
        hdu=0,
    )
    templates_all = {lab: mu_list[i] for i, lab in enumerate(labels)}


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

        # Final 3D mask for THIS region (include data coverage)
        mask3d_plot = build_fit_mask3d(
            roi2d=region2d,
            srcmask3d=srcmask,
            counts=counts,
            expo=expo,
        )

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

    def run_fit_and_plot_global(
        *,
        region_name,
        region2d,
        out_png,
        out_coeff_txt,
        counts,
        expo,
        omega,
        dE_mev,
        Ectr_mev,
        Ectr_gev,
        srcmask,
        mu_list,
        labels,
        aG,             # (nE, nComp) from fit_global_poisson_mle_counts
    ):
        """
        region2d: boolean keep-mask (ny,nx) for plotting/summing.
        srcmask: boolean (nE,ny,nx) keep-mask (ext sources + optional other masks)
        aG multiplies mu_list directly (counts templates).
        """

        # 3D mask for THIS plotted region (region geometry ∩ srcmask ∩ data coverage)
        mask3d_plot = build_fit_mask3d(
            roi2d=region2d,
            srcmask3d=srcmask,
            counts=counts,
            expo=expo,
        )

        # Data E^2 spectrum in this region
        E2_data, E2err_data = data_E2_spectrum_counts(
            counts=counts,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask3d=mask3d_plot,
        )

        # Build predicted counts maps for each component and model sum (restricted to region mask later)
        nE, ny, nx = counts.shape
        nComp = len(mu_list)

        pred_maps = []
        for j in range(nComp):
            pm = np.zeros((nE, ny, nx), float)
            for k in range(nE):
                pm[k] = aG[k, j] * np.asarray(mu_list[j][k], float)
            pred_maps.append(pm)

        model_map = np.sum(pred_maps, axis=0)

        # Convert to Totani-style E^2 <I> spectra using the SAME region2d
        # Use mask2d derived from mask3d_plot per-bin? We'll use a conservative mask2d:
        # require pixel is ever used in any bin; or simply region2d and expo finite each k inside E2 helper.
        mask2d = region2d.copy()

        E2_comp = np.zeros((nComp, nE), float)
        for j in range(nComp):
            E2_comp[j] = E2_from_pred_counts_maps(
                pred_counts_map=pred_maps[j],
                expo=expo,
                omega=omega,
                dE_mev=dE_mev,
                Ectr_mev=Ectr_mev,
                mask2d=mask2d,
            )

        E2_model = E2_from_pred_counts_maps(
            pred_counts_map=model_map,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=mask2d,
        )

        comp_specs = {lab: E2_comp[j] for j, lab in enumerate(labels)}
        comp_specs["MODEL_SUM"] = E2_model

        # Closure printout
        frac_resid = np.full_like(E2_data, np.nan, dtype=float)
        for k in range(nE):
            if np.isfinite(E2_data[k]) and E2_data[k] != 0 and np.isfinite(E2_model[k]):
                frac_resid[k] = (E2_data[k] - E2_model[k]) / E2_data[k]

        print("[closure]", region_name)
        for k in range(nE):
            if np.isfinite(frac_resid[k]):
                print(f"  k={k:02d} E={Ectr_gev[k]:.3g} GeV frac_resid={(100*frac_resid[k]):+.2f}%")

        # Plot
        plot_fit_fig2(
            Ectr_gev=Ectr_gev,
            data_y=E2_data,
            data_yerr=E2err_data,
            comp_specs=comp_specs,
            outpath=out_png,
            title=region_name,
        )

        # Save per-bin total COUNTS attributed to each component within this region
        with open(out_coeff_txt, "w") as f:
            f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
            for k in range(nE):
                m = mask3d_plot[k]
                if not np.any(m):
                    f.write(f"{k:02d} {Ectr_gev[k]:.6g} " + " ".join("nan" for _ in labels) + "\n")
                    continue
                csum = []
                for j in range(nComp):
                    s = float(np.nansum(np.asarray(mu_list[j][k], float)[m]))
                    csum.append(float(aG[k, j]) * s)
                f.write(f"{k:02d} {Ectr_gev[k]:.6g} " + " ".join(f"{x:.6g}" for x in csum) + "\n")

        print("✓ wrote", out_coeff_txt)

    # Define regions
    disk_cut = float(args.disk_cut_fit)
    fig2_region2d = roi2d                              # Fig 2: include disk
    fig3_region2d = roi2d & (np.abs(lat) >= disk_cut)  # Fig 3: exclude disk

    # Fit mask is the SAME for both figures: ROI including disk
    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=srcmask,
        counts=counts,
        expo=expo,
    )

    k=2
    m = fit_mask3d[k]
    ics = mu_list[labels.index("ics")][k][m]
    iso = mu_list[labels.index("iso")][k][m]
    print("ICS std/mean:", np.std(ics)/np.mean(ics))
    print("ISO std/mean:", np.std(iso)/np.mean(iso))

    # Cellwise Poisson MLE fit in TRUE counts-template units.
    # Any internal column scaling is undone before returning coeffs.
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
        column_scale="l2",
        drop_tol=0.0,
        ridge=0.0,
    )

    cells = res_fit["cells"]
    coeff_cells = res_fit["coeff_cells"]
    info = res_fit["info"]


    # Fig 2/3 outputs
    out_fig2 = os.path.join(args.outdir, "totani_fig2_fit_components.png")
    out_c2   = os.path.join(args.outdir, "fit_coefficients_fig2_highlat.txt")
    out_fig3 = os.path.join(args.outdir, "totani_fig3_fit_components.png")
    out_c3   = os.path.join(args.outdir, "fit_coefficients_fig3_disk.txt")
    out_fig2_glob = os.path.join(args.outdir, "totani_fig2_fit_components_global.png")
    out_c2_glob   = os.path.join(args.outdir, "fit_coefficients_fig2_highlat_global.txt")
    out_fig3_glob = os.path.join(args.outdir, "totani_fig3_fit_components_global.png")
    out_c3_glob   = os.path.join(args.outdir, "fit_coefficients_fig3_disk_global.txt")
    # # Fig 2 outputs
    # run_fit_and_plot(
    #     region_name=rf"|l|≤{args.roi_lon}°\n|b|≤{args.roi_lat}°",
    #     region2d=fig2_region2d,
    #     out_png=out_fig2,
    #     out_coeff_txt=out_c2,
    #     cells=cells,
    #     coeff_cells=coeff_cells,
    #     counts=counts,
    #     expo=expo,
    #     omega=omega,
    #     dE_mev=dE_mev,
    #     Ectr_mev=Ectr_mev,
    #     Ectr_gev=Ectr_gev,
    #     srcmask=srcmask,
    #     mu_list=mu_list,
    #     labels=labels,
    # )

    # # Fig 3 outputs
    # run_fit_and_plot(
    #     region_name=rf"|l|≤{args.roi_lon}°\n{disk_cut}°≤|b|≤{args.roi_lat}°\n(fit including |b|<{disk_cut}°)",
    #     region2d=fig3_region2d,
    #     out_png=out_fig3,
    #     out_coeff_txt=out_c3,
    #     cells=cells,
    #     coeff_cells=coeff_cells,
    #     counts=counts,
    #     expo=expo,
    #     omega=omega,
    #     dE_mev=dE_mev,
    #     Ectr_mev=Ectr_mev,
    #     Ectr_gev=Ectr_gev,
    #     srcmask=srcmask,
    #     mu_list=mu_list,
    #     labels=labels,
    # )

    #   # 2D fit region (your fit mask for the solve)
    # expo_safe = np.array(expo, dtype=float, copy=True)
    # expo_safe[~np.isfinite(expo_safe) | (expo_safe <= 0)] = np.nan
    # data_ok3d = np.isfinite(counts) & np.isfinite(expo_safe)
    # data_ok2d = np.any(data_ok3d, axis=0)
    # srcmask2d = np.any(srcmask, axis=0)
    # fit_mask2d = roi2d & data_ok2d & srcmask2d

    # # 3D mask used in fit (same as you pass to fit_cellwise_poisson_mle_counts)
    # fit_mask3d = build_fit_mask3d(roi2d=roi2d, srcmask3d=srcmask, counts=counts, expo=expo)

    # # 1) scales
    # print_template_sums_vs_data(counts, templates_all, fit_mask2d)

    # # 2) allocations
    # pred = predicted_component_counts_from_cellfit(counts=counts,
    #     cells=cells, coeff_cells=coeff_cells, labels=labels,
    #     templates_all=templates_all, mask3d=fit_mask3d
    # )
    # print_component_fractions(pred, Ectr_gev=Ectr_gev)

    # # 3) where it happens
    # cell_component_fractions(k=2, cells=cells, coeff_cells=coeff_cells, labels=labels, templates_all=templates_all, mask3d=fit_mask3d)

    # # 4) degeneracy in a chosen cell
    # print_cell_template_correlations(k=2, cell_idx=0, cells=cells, labels=labels, templates_all=templates_all, mask3d=fit_mask3d)

    # # 5) fb support
    # print_fb_support(templates_all["fb_flat"], fit_mask3d, roi2d, lat)

    # j = labels.index("nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno")
    # print("NFW coeff median by bin:", [np.nanmedian(coeff_cells[:,k,j]) for k in range(nE)])
    # for k in range(nE):
    #     v = coeff_cells[:,k,j]
    #     v = v[np.isfinite(v)]
    #     print(k, np.percentile(v,[16,50,84]))

    templates_all = {lab: mu for lab, mu in zip(labels, mu_list)}
    resG = fit_global_poisson_mle_counts(counts=counts, templates=templates_all, mask3d=fit_mask3d, ridge=None)
    aG = resG["coeff"]

    k = 2
    m = fit_mask3d[k]
    mu_iso = mu_list[labels.index("iso")][k][m].ravel()

    # build model without iso from your fitted global coeffs aG
    pred_wo_iso = np.zeros_like(mu_iso)
    for j, lab in enumerate(labels):
        if lab == "iso": 
            continue
        pred_wo_iso += aG[k, j] * mu_list[j][k][m].ravel()

    resid = counts[k][m].ravel() - pred_wo_iso
    # if resid is mostly <=0 where mu_iso>0, iso will go to 0
    print("resid sum:", resid.sum(), "resid median:", np.median(resid))
    print("corr(resid, mu_iso):", np.corrcoef(resid, mu_iso)[0,1])

    run_fit_and_plot_global(
        region_name=rf"|l|≤{args.roi_lon}°\n|b|≤{args.roi_lat}°",
        region2d=fig2_region2d,
        out_png=out_fig2_glob,
        out_coeff_txt=out_c2_glob,
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        Ectr_gev=Ectr_gev,
        srcmask=srcmask,
        mu_list=mu_list,
        labels=labels,
        aG=aG,
    )

    run_fit_and_plot_global(
        region_name=rf"|l|≤{args.roi_lon}°\n{disk_cut}°≤|b|≤{args.roi_lat}°\n(fit including |b|<{disk_cut}°)",
        region2d=fig3_region2d,
        out_png=out_fig3_glob,
        out_coeff_txt=out_c3_glob,
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        Ectr_gev=Ectr_gev,
        srcmask=srcmask,
        mu_list=mu_list,
        labels=labels,
        aG=aG,
    )

    j_iso = labels.index("iso")
    for k in [0,2,6,10]:
        m = fit_mask3d[k]
        print(f"k={k:02d} iso sum={np.nansum(mu_list[j_iso][k][m]):.3e}  "
            f"iso median={np.nanmedian(mu_list[j_iso][k][m]):.3e}  "
            f"iso max={np.nanmax(mu_list[j_iso][k][m]):.3e}")

    k = 2
    m = fit_mask3d[k]
    x = mu_list[labels.index("ics")][k][m].ravel()
    y = mu_list[labels.index("iso")][k][m].ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    x -= x.mean(); y -= y.mean()
    corr = (x@y)/((np.sqrt((x@x)*(y@y)))+1e-30)
    print("corr(ics, iso) =", corr)

    

if __name__ == "__main__":
    main()