#!/usr/bin/env python3
"""Utilities for exporting MCMC coefficients.

This script intentionally contains *only* MCMC-output loading + light post-processing.
All Totani Fig2/3 fitting/plotting logic has been moved elsewhere.
"""

import os
import sys
import argparse
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, save_coeff_table_txt
from totani_helpers.fig23_mcmc import make_fig2_fig3_plots_from_mcmc


def main():
    p = argparse.ArgumentParser(description="Export MCMC coefficients to a txt table.")
    repo_dir = os.environ.get("REPO_PATH", os.path.expanduser("~/Documents/PhD/Year 2/DM_Photon_Scattering"))

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

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
    p.add_argument(
        "--out-txt",
        default=os.path.join(os.path.dirname(__file__), "mcmc_coefficients.txt"),
        help="Output txt path.",
    )

    p.add_argument(
        "--make-plots",
        action="store_true",
        help="Also make the Fig2/Fig3-style component spectra plots from the MCMC coefficients.",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig2_3"),
        help="Output directory for plots/tables when --make-plots is set.",
    )
    p.add_argument(
        "--counts",
        default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"),
        help="Counts cube FITS used for E-binning and WCS when making plots.",
    )
    p.add_argument(
        "--expo",
        default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"),
        help="Exposure cube FITS used when making plots.",
    )
    p.add_argument(
        "--templates-dir",
        default=os.path.join(data_dir, "processed", "templates"),
        help="Directory containing mu_{label}_counts.fits templates.",
    )
    p.add_argument(
        "--ext-mask",
        default=os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits"),
        help="Extended source mask (True=keep). Can be (ny,nx) or (nE,ny,nx).",
    )
    p.add_argument("--roi-lon", type=float, default=60.0)
    p.add_argument("--roi-lat", type=float, default=60.0)
    p.add_argument("--disk-cut", type=float, default=10.0)
    p.add_argument("--binsz", type=float, default=0.125)
    p.add_argument("--plot-style", choices=["diagnostic", "totani"], default="diagnostic")
    p.add_argument(
        "--nE",
        type=int,
        default=None,
        help="Optional number of energy bins (if omitted, inferred from available mcmc_results_kXX files).",
    )
    p.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Optional list of component keys/labels to save (default: all labels found in the MCMC output, after loopI combine).",
    )
    p.add_argument(
        "--combine-loopI",
        action="store_true",
        help="If present and loopA/loopB exist, add loopI=loopA+loopB (and drop loopA/loopB).",
    )

    args = p.parse_args()

    tab = load_mcmc_coeffs_by_label(mcmc_dir=args.mcmc_dir, stat=args.mcmc_stat, nE=args.nE)
    coeffs = tab.coeffs_by_label

    if args.combine_loopI:
        coeffs = combine_loopI(coeffs_by_label=coeffs, out_key="loopI", drop_inputs=True)

    if args.keys is None:
        # Preserve the original label order where possible.
        keys = [k for k in tab.labels if k in coeffs]
        for k in coeffs.keys():
            if k not in keys:
                keys.append(k)
    else:
        keys = [str(k) for k in args.keys]

    x = np.arange(int(np.nanmax(tab.bins_present)) + 1, dtype=int)
    save_coeff_table_txt(out_txt=args.out_txt, x=x, coeffs_by_label=coeffs, keys=keys, x_label="k")
    print(f"✓ Wrote {args.out_txt}")

    if args.make_plots:
        make_fig2_fig3_plots_from_mcmc(
            counts_path=args.counts,
            expo_path=args.expo,
            templates_dir=args.templates_dir,
            mcmc_dir=args.mcmc_dir,
            outdir=args.outdir,
            mcmc_stat=args.mcmc_stat,
            plot_style=args.plot_style,
            ext_mask_path=args.ext_mask,
            roi_lon=float(args.roi_lon),
            roi_lat=float(args.roi_lat),
            disk_cut=float(args.disk_cut),
            binsz=float(args.binsz),
        )
        print(f"✓ Wrote plots to {args.outdir}")



if __name__ == "__main__":
    main()
