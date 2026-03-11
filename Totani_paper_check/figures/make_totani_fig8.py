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
    lonlat_grids,
    load_mask_any_shape,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.fit_utils import (  # noqa: E402
    E2_from_pred_counts_maps,
    build_fit_mask3d,
    load_mu_templates_from_fits,
)
from totani_helpers.mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, pick_coeff


def _assert_templates_match_counts(counts: np.ndarray, mu_list: list[np.ndarray], labels: list[str]) -> None:
    if not isinstance(mu_list, (list, tuple)) or len(mu_list) != len(labels):
        raise ValueError("mu_list must align with labels")
    for lab, mu in zip(labels, mu_list):
        mu = np.asarray(mu)
        if mu.shape != counts.shape:
            raise ValueError(f"Template '{lab}' has shape {mu.shape}, expected counts shape {counts.shape}")


REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def _E2_for_component(*, mu: np.ndarray, coeff: np.ndarray, expo, omega, Ectr_mev, dE_mev, mask2d) -> np.ndarray:
    pred = np.asarray(coeff, float).reshape(-1)[:, None, None] * np.asarray(mu, float)
    return E2_from_pred_counts_maps(
        pred_counts_map=pred,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask2d=mask2d,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Totani Fig.8: linear plots of NFW component spectra for rho^2.5, rho^2, rho^1 fits; plus diffuse model flux at the Galactic poles."
    )
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig8"))

    ap.add_argument(
        "--mcmc-dir-rho25",
        default=os.path.join(REPO_DIR, "Totani_paper_check", "mcmc", "mcmc_results_fig5"),
        help="MCMC directory for rho=2.5 variant (mcmc_results_kXX.npz files).",
    )
    ap.add_argument(
        "--mcmc-dir-rho2",
        default=os.path.join(REPO_DIR, "Totani_paper_check", "mcmc", "mcmc_results_fig6"),
        help="MCMC directory for rho=2 variant (mcmc_results_kXX.npz files).",
    )
    ap.add_argument(
        "--mcmc-dir-rho1",
        default=os.path.join(REPO_DIR, "Totani_paper_check", "mcmc", "mcmc_results_fig7"),
        help="MCMC directory for rho=1 variant (mcmc_results_kXX.npz files).",
    )
    ap.add_argument(
        "--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"],
        default="f_ml",
        help="Which MCMC summary coefficient to use per bin.",
    )

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

    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=srcmask,
        counts=counts,
        expo=expo,
        extra2d=hilat2d,
    )

    pole2d = roi2d & (np.abs(lat) >= float(args.pole_cap_deg))
    pole_mask3d = build_fit_mask3d(
        roi2d=pole2d,
        srcmask3d=srcmask,
        counts=counts,
        expo=expo,
    )

    variants = [
        ("rho2.5", str(args.mcmc_dir_rho25), "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno"),
        ("rho2", str(args.mcmc_dir_rho2), "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno"),
        ("rho1", str(args.mcmc_dir_rho1), "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno"),
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

    roi_mask2d = np.all(fit_mask3d, axis=0)
    pole_mask2d = np.all(pole_mask3d, axis=0)

    for tag, mcmc_dir, nfw_label in variants:
        # For loopI combine, load loopA/loopB explicitly and build loopI contribution in code.
        labels_load = ["gas", "iso", "ps", "loopA", "loopB", "ics", nfw_label, "fb_pos", "fb_neg"]
        mu_list, _hdrs = load_mu_templates_from_fits(
            template_dir=args.templates_dir,
            labels=labels_load,
            filename_pattern="mu_{label}_counts.fits",
            hdu=0,
        )
        _assert_templates_match_counts(counts, mu_list, labels_load)

        tab = load_mcmc_coeffs_by_label(mcmc_dir=str(mcmc_dir), stat=str(args.mcmc_stat), nE=nE)
        coeffs_plot = combine_loopI(coeffs_by_label=dict(tab.coeffs_by_label), out_key="loopI", drop_inputs=False)

        mu_by_label = {lab: mu for lab, mu in zip(labels_load, mu_list)}

        # (A) NFW component in ROI
        mu_nfw = mu_by_label[nfw_label]
        a_nfw = pick_coeff(coeffs_by_label=coeffs_plot, template_key=nfw_label)
        E2_nfw_roi = _E2_for_component(
            mu=mu_nfw,
            coeff=a_nfw,
            expo=expo,
            omega=omega,
            Ectr_mev=Ectr_mev,
            dE_mev=dE_mev,
            mask2d=roi_mask2d,
        )
        ax.plot(Ectr_gev, E2_nfw_roi, marker="o", label=f"NFW {tag}")

        # (B) Diffuse pole flux = gas + iso + (loopA+loopB) + ics over poles
        pole_diffuse = np.zeros_like(Ectr_mev, dtype=float)
        for lab in ["gas", "iso", "ics"]:
            pole_diffuse += _E2_for_component(
                mu=mu_by_label[lab],
                coeff=pick_coeff(coeffs_by_label=coeffs_plot, template_key=lab),
                expo=expo,
                omega=omega,
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                mask2d=pole_mask2d,
            )

        # loopI = loopA + loopB
        if ("loopA" in mu_by_label) and ("loopB" in mu_by_label):
            pole_diffuse += _E2_for_component(
                mu=mu_by_label["loopA"],
                coeff=pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopA"),
                expo=expo,
                omega=omega,
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                mask2d=pole_mask2d,
            )
            pole_diffuse += _E2_for_component(
                mu=mu_by_label["loopB"],
                coeff=pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopB"),
                expo=expo,
                omega=omega,
                Ectr_mev=Ectr_mev,
                dE_mev=dE_mev,
                mask2d=pole_mask2d,
            )

        axp.plot(Ectr_gev, pole_diffuse, marker="o", label=f"Diffuse poles ({tag})")

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
