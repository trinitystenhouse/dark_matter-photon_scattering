#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from astropy.wcs import WCS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import (  # noqa: E402
    load_mask_any_shape,
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts  # noqa: E402

from fig2_3.make_totani_fig2_fig3_all import (  # noqa: E402
    assert_templates_match_counts,
    load_mu_templates_from_fits,
    model_E2_spectrum_from_cells_counts,
)


REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def _fit_and_load_templates(*, labels, templates_dir, counts, lon, lat, fit_mask3d, roi_lon, roi_lat, cell_deg):
    mu_list, _hdrs = load_mu_templates_from_fits(
        template_dir=templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )
    assert_templates_match_counts(counts, mu_list, labels)

    res_fit = fit_cellwise_poisson_mle_counts(
        counts=counts,
        templates=mu_list,
        mask3d=fit_mask3d,
        lon=lon,
        lat=lat,
        roi_lon=float(roi_lon),
        roi_lat=float(roi_lat),
        cell_deg=float(cell_deg),
        nonneg=True,
    )

    return mu_list, res_fit


def _components_E2_for_mask(*, labels, mu_list, res_fit, expo, omega, Ectr_mev, dE_mev, mask3d_eval):
    E2_comp, E2_model = model_E2_spectrum_from_cells_counts(
        coeff_cells=res_fit["coeff_cells"],
        cells=res_fit["cells"],
        templates=mu_list,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_eval,
    )
    return E2_comp, E2_model


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Totani Fig.8: linear plots of NFW component spectra for rho^2.5, rho^2, rho^1 fits; plus diffuse model flux at the Galactic poles."
    )
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig8"))

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)
    ap.add_argument("--cell-deg", type=float, default=10.0)

    ap.add_argument(
        "--ext-mask",
        default=os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits"),
        help="Extended-source keep mask FITS (True=keep, False=masked). Applied before fitting.",
    )
    ap.add_argument("--mask-fit", default=None, help="Optional additional keep mask FITS (2D or 3D), True=keep")

    ap.add_argument("--pole-cap-deg", type=float, default=80.0, help="Evaluate pole flux over |b| >= pole-cap-deg")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    counts, hdr, _Emin, _Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
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

    if args.mask_fit is not None:
        srcmask = load_mask_any_shape(args.mask_fit, counts.shape)
    else:
        srcmask = np.ones_like(counts, dtype=bool)

    if args.ext_mask is not None and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), counts.shape)
        srcmask = srcmask & ext_keep3d

    fit_mask3d = srcmask & (roi2d & hilat2d)[None, :, :]

    pole2d = roi2d & (np.abs(lat) >= float(args.pole_cap_deg))
    pole_mask3d = srcmask & pole2d[None, :, :]

    variants = [
        ("rho2.5", "nfw_rho2.5_g1.25_pheno"),
        ("rho2", "nfw_rho2_g1.25_pheno"),
        ("rho1", "nfw_rho1_g1.25_pheno"),
    ]

    Ectr_gev = Ectr_mev / 1000.0

    # (A) Linear plot: NFW component spectra from fits
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")

    # (B) Diffuse pole flux (MODEL diffuse sum) from each fit
    figp = plt.figure(figsize=(7.2, 5.2))
    axp = figp.add_subplot(111)
    axp.set_xscale("log")

    for tag, nfw_label in variants:
        labels = ["gas", "iso", "ps", "loopI", "ics", nfw_label, "bubbles_pos", "bubbles_neg"]

        mu_list, res_fit = _fit_and_load_templates(
            labels=labels,
            templates_dir=args.templates_dir,
            counts=counts,
            lon=lon,
            lat=lat,
            fit_mask3d=fit_mask3d,
            roi_lon=float(args.roi_lon),
            roi_lat=float(args.roi_lat),
            cell_deg=float(args.cell_deg),
        )

        E2_comp_roi, _E2_model_roi = _components_E2_for_mask(
            labels=labels,
            mu_list=mu_list,
            res_fit=res_fit,
            expo=expo,
            omega=omega,
            Ectr_mev=Ectr_mev,
            dE_mev=dE_mev,
            mask3d_eval=fit_mask3d,
        )
        j_nfw = labels.index(nfw_label)
        ax.plot(Ectr_gev, E2_comp_roi[j_nfw], marker="o", label=f"NFW {tag}")

        E2_comp_pole, _E2_model_pole = _components_E2_for_mask(
            labels=labels,
            mu_list=mu_list,
            res_fit=res_fit,
            expo=expo,
            omega=omega,
            Ectr_mev=Ectr_mev,
            dE_mev=dE_mev,
            mask3d_eval=pole_mask3d,
        )

        # Diffuse pole flux = sum of diffuse components (exclude PS, NFW, bubbles)
        diffuse_labels = ["gas", "iso", "loopI", "ics"]
        pole_diffuse = np.zeros_like(Ectr_gev, dtype=float)
        for lab in diffuse_labels:
            pole_diffuse += E2_comp_pole[labels.index(lab)]

        axp.plot(Ectr_gev, pole_diffuse, marker="o", label=f"Diffuse poles (fit {tag})")

    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title("Halo (NFW) component spectra (linear y)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_nfw = os.path.join(args.outdir, "totani_fig8_nfw_linear.png")
    fig.savefig(out_nfw, dpi=200)
    plt.close(fig)
    print("✓ wrote", out_nfw)

    axp.set_xlabel("Energy (GeV)")
    axp.set_ylabel(r"$E^2\,dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    axp.set_title(rf"Diffuse model flux at Galactic poles (|b|≥{args.pole_cap_deg:g}°), linear y")
    axp.grid(True, which="both", alpha=0.25)
    axp.legend(frameon=False)
    figp.tight_layout()
    out_pole = os.path.join(args.outdir, "totani_fig8_poles_diffuse_linear.png")
    figp.savefig(out_pole, dpi=200)
    plt.close(figp)
    print("✓ wrote", out_pole)


if __name__ == "__main__":
    main()
