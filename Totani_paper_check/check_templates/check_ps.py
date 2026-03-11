#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from check_component import run_component_check
from totani_helpers.mcmc_io import add_flux_scaling_args


def main():
    repo_dir = os.environ.get("REPO_PATH")
    if repo_dir is None:
        raise SystemExit("REPO_PATH not set")

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    template = os.path.join(data_dir, "processed", "templates", "mu_ps_counts.fits")
    plot_dir = os.path.join(os.path.dirname(__file__), "plots_check_ps")

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

    with fits.open(str(args.counts)) as hc:
        counts_cube = np.asarray(hc[0].data, float)
        hdr = hc[0].header
        eb = hc["EBOUNDS"].data
        Ectr_mev = np.sqrt((eb["E_MIN"].astype(float) / 1000.0) * (eb["E_MAX"].astype(float) / 1000.0))

    with fits.open(str(args.template)) as ht:
        mu_ps = np.asarray(ht[0].data, float)
        bunit = str(ht[0].header.get("BUNIT", "")).strip()

    nE, ny, nx = counts_cube.shape
    if mu_ps.shape != (nE, ny, nx):
        raise SystemExit(f"PS template shape {mu_ps.shape} != counts shape {(nE, ny, nx)}")

    fin = np.isfinite(mu_ps)
    neg = np.isfinite(mu_ps) & (mu_ps < 0)
    if not fin.all():
        raise SystemExit(f"PS template contains non-finite values: finite_frac={fin.mean():.6f}")
    if np.any(neg):
        raise SystemExit(f"PS template contains negative values: neg_frac={neg.mean():.6f}")
    if float(np.nansum(mu_ps)) <= 0:
        raise SystemExit("PS template total sum is <= 0; expected positive counts")

    os.makedirs(plot_dir, exist_ok=True)

    # ROI sums (same ROI as generic checker)
    from astropy.wcs import WCS

    wcs = WCS(hdr).celestial
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    roi2d = (np.abs(lon) <= 60.0) & (np.abs(lat) <= 60.0)

    data_sum = np.nansum(counts_cube[:, roi2d], axis=1)
    ps_sum = np.nansum(mu_ps[:, roi2d], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ps_sum / np.maximum(data_sum, 1.0)

    os.makedirs(str(args.plot_dir), exist_ok=True)
    summary_path = os.path.join(str(args.plot_dir), "summary_ps.txt")
    with open(summary_path, "w") as f:
        f.write(f"template: {str(args.template)}\n")
        f.write(f"counts:   {str(args.counts)}\n")
        f.write(f"BUNIT:    {bunit}\n")
        f.write(f"shape:    {mu_ps.shape}\n")
        f.write(f"total_mu_ps: {float(np.nansum(mu_ps)):.6e}\n")
        f.write(f"roi_total_mu_ps: {float(np.nansum(ps_sum)):.6e}\n")
        f.write(f"roi_total_counts: {float(np.nansum(data_sum)):.6e}\n")
        f.write(f"roi_ps_over_counts_mean: {float(np.nanmean(ratio)):.6e}\n")
        f.write(f"roi_ps_over_counts_median: {float(np.nanmedian(ratio)):.6e}\n")
        f.write("\n# k  Ectr_GeV  data_sum  ps_sum  ps_over_data\n")
        for k in range(nE):
            f.write(f"{k:3d} {Ectr_mev[k]/1000.0:12.6g} {data_sum[k]:.6e} {ps_sum[k]:.6e} {ratio[k]:.6e}\n")

    # Plot ratio vs energy
    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(Ectr_mev / 1000.0, ratio, lw=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ectr [GeV]")
    ax.set_ylabel("PS sum / data sum (ROI)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(str(args.plot_dir), "ps_over_data_roi_sum.png"), dpi=200)
    plt.close(fig)

    return run_component_check(
        label="PS",
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
