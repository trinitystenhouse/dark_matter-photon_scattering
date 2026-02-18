#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import (  # noqa: E402
    lonlat_grids,
    load_mask_any_shape,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts  # noqa: E402
from totani_helpers.fit_utils import build_fit_mask3d  # noqa: E402

# Reuse plotting/spectrum utilities from fig2_3
from fig2_3.make_totani_fig2_fig3_all import (  # noqa: E402
    assert_templates_match_counts,
    data_E2_spectrum_counts,
    load_mu_templates_from_fits,
    model_E2_spectrum_from_cells_counts,
    plot_fit_fig2,
)


REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def main():
    ap = argparse.ArgumentParser(description="Reproduce Totani Fig.4: best-fit spectra excluding disk, using bubbles pos/neg, no GC excess.")
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig4"))

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0, help="Exclude disk: fit uses |b|>=disk-cut")
    ap.add_argument("--cell-deg", type=float, default=10.0)

    ap.add_argument(
        "--ext-mask",
        default=os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits"),
        help="Extended-source keep mask FITS (True=keep, False=masked). Applied before fitting.",
    )
    ap.add_argument(
        "--mask-fit",
        default=None,
        help="Optional additional keep mask FITS (2D or 3D) applied before fitting (True=keep)",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")

    omega = pixel_solid_angle_map(wcs, ny, nx, float(args.binsz))
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon) <= float(args.roi_lon)) & (np.abs(lat) <= float(args.roi_lat))
    hilat2d = np.abs(lat) >= float(args.disk_cut)

    # Keep masks
    if args.mask_fit is not None:
        srcmask = load_mask_any_shape(args.mask_fit, counts.shape)
    else:
        srcmask = np.ones_like(counts, dtype=bool)

    if args.ext_mask is not None and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), counts.shape)
        srcmask = srcmask & ext_keep3d

    # Fit mask: ROI and disk removed
    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=srcmask,
        counts=counts,
        expo=expo,
        extra2d=hilat2d,
    )

    # Components: no NFW, no flat bubbles; use bubbles_pos/neg
    labels = ["gas", "iso", "ps", "loopI", "ics", "fb_neg_norm", "fb_pos_norm"]
    mu_list, _hdrs = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )
    assert_templates_match_counts(counts, mu_list, labels)

    Ectr_gev = Ectr_mev / 1000.0

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

    # Plot region is the same as fit region for Fig.4
    mask3d_plot = fit_mask3d

    E2_data, E2err_data = data_E2_spectrum_counts(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_plot,
    )

    E2_comp, E2_model = model_E2_spectrum_from_cells_counts(
        coeff_cells=coeff_cells,
        cells=cells,
        templates=mu_list,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_plot,
    )

    comp_specs = {lab: E2_comp[j] for j, lab in enumerate(labels)}
    # Invert sign of negative bubbles component for plotting so it appears positive.
    if "bubbles_neg" in comp_specs:
        comp_specs["bubbles_neg"] = -comp_specs["bubbles_neg"]
    comp_specs["MODEL_SUM"] = E2_model

    out_png = os.path.join(args.outdir, "totani_fig4_fit_components.png")
    title = rf"|l|≤{args.roi_lon}°\n{args.disk_cut}°≤|b|≤{args.roi_lat}° (fit excludes disk; no NFW; bubbles pos/neg)"
    plot_fit_fig2(
        Ectr_gev=Ectr_gev,
        data_y=E2_data,
        data_yerr=E2err_data,
        comp_specs=comp_specs,
        outpath=out_png,
        title=title,
    )

    out_coeff = os.path.join(args.outdir, "fit_coefficients_fig4_disk_removed_posneg.txt")
    with open(out_coeff, "w") as f:
        f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
        for k in range(nE):
            csum = np.zeros(len(labels), float)
            for ci, cell2d in enumerate(cells):
                cm = mask3d_plot[k] & cell2d
                if not np.any(cm):
                    continue
                a = coeff_cells[ci, k, :]
                if not np.all(np.isfinite(a)):
                    continue
                for j in range(len(labels)):
                    s = float(np.nansum(mu_list[j][k][cm]))
                    if np.isfinite(s) and s != 0.0:
                        csum[j] += float(a[j]) * s

            f.write(
                f"{k:02d} {Ectr_gev[k]:.6g} "
                + " ".join(f"{csum[j]:.6g}" for j in range(len(labels)))
                + "\n"
            )

    print("✓ wrote", out_png)
    print("✓ wrote", out_coeff)


if __name__ == "__main__":
    main()
