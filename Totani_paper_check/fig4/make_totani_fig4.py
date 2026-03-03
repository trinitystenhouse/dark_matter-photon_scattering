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
from totani_helpers.fit_utils import E2_from_pred_counts_maps, build_fit_mask3d  # noqa: E402
from totani_helpers.mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, pick_coeff

# Reuse plotting/spectrum utilities from fig2_3
from fig2_3.make_totani_fig2_fig3_all import (  # noqa: E402
    assert_templates_match_counts,
    data_E2_spectrum_counts,
    load_mu_templates_from_fits,
    plot_fit_fig2,
)


REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def main():
    ap = argparse.ArgumentParser(description="Reproduce Totani Fig.4: best-fit spectra excluding disk, using bubbles pos/neg, no GC excess.")
    ap.add_argument(
        "--mcmc-dir",
        default=os.path.join(REPO_DIR, "Totani_paper_check", "mcmc", "mcmc_results"),
        help="Directory containing mcmc_results_kXX.npz files.",
    )
    ap.add_argument(
        "--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"],
        default="f_ml",
        help="Which MCMC summary coefficient to use per bin.",
    )
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
    labels = ["gas", "iso", "ps", "loopI", "ics", "fb_neg", "fb_pos"]
    mu_list, _hdrs = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )
    assert_templates_match_counts(counts, mu_list, labels)

    Ectr_gev = Ectr_mev / 1000.0

    tab = load_mcmc_coeffs_by_label(mcmc_dir=str(args.mcmc_dir), stat=str(args.mcmc_stat), nE=nE)
    coeffs_plot = combine_loopI(coeffs_by_label=dict(tab.coeffs_by_label), out_key="loopI", drop_inputs=False)

    mask3d_plot = fit_mask3d
    mask2d_plot = np.all(mask3d_plot, axis=0)

    E2_data, E2err_data = data_E2_spectrum_counts(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_plot,
    )

    comp_specs = {}
    E2_model = np.zeros_like(Ectr_mev, dtype=float)
    for lab, mu in zip(labels, mu_list):
        if str(lab) == "loopI" and ("loopA" in coeffs_plot) and ("loopB" in coeffs_plot):
            aA = pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopA")
            aB = pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopB")
            muA = load_mu_templates_from_fits(
                template_dir=args.templates_dir,
                labels=["loopA"],
                filename_pattern="mu_{label}_counts.fits",
                hdu=0,
            )[0][0]
            muB = load_mu_templates_from_fits(
                template_dir=args.templates_dir,
                labels=["loopB"],
                filename_pattern="mu_{label}_counts.fits",
                hdu=0,
            )[0][0]
            pred = np.asarray(aA, float)[:, None, None] * np.asarray(muA, float) + np.asarray(aB, float)[:, None, None] * np.asarray(muB, float)
        else:
            a = pick_coeff(coeffs_by_label=coeffs_plot, template_key=str(lab))
            pred = np.asarray(a, float)[:, None, None] * np.asarray(mu, float)

        E2 = E2_from_pred_counts_maps(
            pred_counts_map=pred,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=mask2d_plot,
        )
        comp_specs[str(lab)] = E2
        E2_model += E2

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
            for j, lab in enumerate(labels):
                if str(lab) == "loopI" and ("loopA" in coeffs_plot) and ("loopB" in coeffs_plot):
                    aA = float(np.asarray(coeffs_plot["loopA"], float).reshape(-1)[k])
                    aB = float(np.asarray(coeffs_plot["loopB"], float).reshape(-1)[k])
                    muA = load_mu_templates_from_fits(
                        template_dir=args.templates_dir,
                        labels=["loopA"],
                        filename_pattern="mu_{label}_counts.fits",
                        hdu=0,
                    )[0][0]
                    muB = load_mu_templates_from_fits(
                        template_dir=args.templates_dir,
                        labels=["loopB"],
                        filename_pattern="mu_{label}_counts.fits",
                        hdu=0,
                    )[0][0]
                    csum[j] = float(aA) * float(np.nansum(muA[k][mask3d_plot[k]])) + float(aB) * float(np.nansum(muB[k][mask3d_plot[k]]))
                else:
                    a = float(np.asarray(pick_coeff(coeffs_by_label=coeffs_plot, template_key=str(lab)), float).reshape(-1)[k])
                    csum[j] = a * float(np.nansum(mu_list[j][k][mask3d_plot[k]]))

            f.write(
                f"{k:02d} {Ectr_gev[k]:.6g} "
                + " ".join(f"{csum[j]:.6g}" for j in range(len(labels)))
                + "\n"
            )

    print("✓ wrote", out_png)
    print("✓ wrote", out_coeff)


if __name__ == "__main__":
    main()
