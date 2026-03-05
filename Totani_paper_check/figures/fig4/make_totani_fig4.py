#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, save_coeff_table_txt
from totani_helpers.fig23_mcmc import make_plots_from_mcmc


REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def main():
    ap = argparse.ArgumentParser(description="Reproduce Totani Fig.4: best-fit spectra excluding disk, using bubbles pos/neg, no GC excess.")
    ap.add_argument(
        "--mcmc-dir",
        default=os.path.join(REPO_DIR, "Totani_paper_check", "mcmc", "mcmc_results_fig4"),
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
        "--nE",
        type=int,
        default=None,
        help="Optional number of energy bins (if omitted, inferred from available mcmc_results_kXX files).",
    )
    ap.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Optional list of component keys/labels to save (default: all labels found in the MCMC output, after loopI combine).",
    )
    ap.add_argument(
        "--combine-loopI",
        action="store_true",
        help="If present and loopA/loopB exist, add loopI=loopA+loopB (and drop loopA/loopB).",
    )
    ap.add_argument("--fig", type=str, default="fig4")
    ap.add_argument(
        "--out-txt",
        default=os.path.join(os.path.dirname(__file__), f"mcmc_coefficients_fig4.txt"),
        help="Output txt path.",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

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


    # Components: no NFW, no flat bubbles; use bubbles_pos/neg
    labels = ["gas", "iso", "ps", "loopA", "loopB", "ics", "fb_neg", "fb_pos"]

    make_plots_from_mcmc(
        fig=args.fig,
        counts_path=args.counts,
        expo_path=args.expo,
        templates_dir=args.templates_dir,
        mcmc_dir=args.mcmc_dir,
        outdir=args.outdir,
        mcmc_stat=args.mcmc_stat,
        plot_style="totani",
        ext_mask_path=args.ext_mask,
        roi_lon=args.roi_lon,
        roi_lat=args.roi_lat,
        disk_cut=args.disk_cut,
        binsz=args.binsz,
        labels=labels)

    print(f"✓ Wrote plots to {args.outdir}")



if __name__ == "__main__":
    main()
