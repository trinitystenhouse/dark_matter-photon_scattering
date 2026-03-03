#!/usr/bin/env python3

import argparse
import os

from check_component import run_component_check
from totani_helpers.mcmc_io import add_flux_scaling_args


def main():
    repo_dir = os.environ.get("REPO_PATH")
    if repo_dir is None:
        raise SystemExit("REPO_PATH not set")

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    template = os.path.join(data_dir, "processed", "templates", "mu_ics_counts.fits")
    plot_dir = os.path.join(os.path.dirname(__file__), "plots_check_ics")

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=counts)
    ap.add_argument("--template", default=template)
    ap.add_argument("--plot-dir", default=plot_dir)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    add_flux_scaling_args(
        ap,
        default_expo=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"),
        default_coeff_file=os.path.join(repo_dir, "Totani_paper_check", "fig2_3", "mcmc_coefficients_fig2_3_quick.txt"),
        default_binsz=0.125,
        include_mcmc_component=True,
    )
    args = ap.parse_args()

    return run_component_check(
        label="ICS",
        template_path=str(args.template),
        counts_path=str(args.counts),
        roi_lon=float(args.roi_lon),
        roi_lat=float(args.roi_lat),
        plot_dir=str(args.plot_dir) if args.plot_dir is not None else None,
        scale_flux=bool(args.scale_flux),
        expo_path=getattr(args, "expo", None),
        binsz_deg=float(getattr(args, "binsz", 0.125)),
        mcmc_dir=str(args.mcmc_dir) if args.mcmc_dir is not None else None,
        mcmc_stat=str(args.mcmc_stat),
        mcmc_component=str(args.mcmc_component) if args.mcmc_component is not None else None,
        coeff_file=str(args.coeff_file) if getattr(args, "coeff_file", None) is not None else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
