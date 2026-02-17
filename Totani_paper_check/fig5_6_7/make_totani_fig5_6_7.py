#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.totani_io import (  # noqa: E402
    load_mask_any_shape,
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)

from make_fig import make_fig


REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def main():
    ap = argparse.ArgumentParser(description="Reproduce Totani Figs.5-7: best-fit spectra excluding disk, using bubbles pos/neg.")
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig5_6_7"))

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
    fit_mask3d = srcmask & (roi2d & hilat2d)[None, :, :]

    common = dict(
        templates_dir=args.templates_dir,
        counts=counts,
        expo=expo,
        lon=lon,
        lat=lat,
        fit_mask3d=fit_mask3d,
        Ectr_mev=Ectr_mev,
        dE_mev=dE_mev,
        omega=omega,
        roi_lon=float(args.roi_lon),
        roi_lat=float(args.roi_lat),
        cell_deg=float(args.cell_deg),
    )

    labels_5 = ["gas", "iso", "ps", "loopI", "ics", "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno", "bubbles_pos", "bubbles_neg"]
    make_fig(
        labels=labels_5,
        out_png=os.path.join(args.outdir, "totani_fig5_fit_components.png"),
        out_coeff=os.path.join(args.outdir, "fit_coefficients_fig5_disk_removed_posneg.txt"),
        title=rf"|l|≤{args.roi_lon}°\n{args.disk_cut}°≤|b|≤{args.roi_lat}° (fit excludes disk; NFW rho=2.5)",
        **common,
    )

    labels_6 = ["gas", "iso", "ps", "loopI", "ics", "nfw_NFW_g1_rho2_rs21_R08_rvir402_ns2048_normpole_pheno", "bubbles_pos", "bubbles_neg"]
    make_fig(
        labels=labels_6,
        out_png=os.path.join(args.outdir, "totani_fig6_fit_components.png"),
        out_coeff=os.path.join(args.outdir, "fit_coefficients_fig6_disk_removed_posneg.txt"),
        title=rf"|l|≤{args.roi_lon}°\n{args.disk_cut}°≤|b|≤{args.roi_lat}° (fit excludes disk; NFW rho=2)",
        **common,
    )

    labels_7 = ["gas", "iso", "ps", "loopI", "ics", "nfw_NFW_g1_rho1_rs21_R08_rvir402_ns2048_normpole_pheno", "bubbles_pos", "bubbles_neg"]
    make_fig(
        labels=labels_7,
        out_png=os.path.join(args.outdir, "totani_fig7_fit_components.png"),
        out_coeff=os.path.join(args.outdir, "fit_coefficients_fig7_disk_removed_posneg.txt"),
        title=rf"|l|≤{args.roi_lon}°\n{args.disk_cut}°≤|b|≤{args.roi_lat}° (fit excludes disk; NFW rho=1)",
        **common,
    )


if __name__ == "__main__":
    main()
