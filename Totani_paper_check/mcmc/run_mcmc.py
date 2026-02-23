#!/usr/bin/env python3
import os
import sys
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    load_mask_any_shape,
)
from totani_helpers.fit_utils import (
    build_fit_mask3d,
    load_mu_templates_from_fits,
)
from mcmc_helper import (
    totani_bounds,
    totani_mcmc_fit,
)
from get_norms import report_template_normalisations_from_mu_list

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.expanduser("~/Documents/PhD/Year 2/DM_Photon_Scattering"),
)
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def build_roi_box_mask(lon, lat, roi_lon=60.0, roi_lat=60.0):
    return (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)


# -----------------------------------------------------------------------------
# Totani-style init for YOUR parameterisation (mu are counts-space templates)
# -----------------------------------------------------------------------------
def _infer_E2dNdE_for_iso_at_f1(mu_iso_vec, denom_vec, Ectr_mev):
    """
    mu_iso_vec : counts vector for iso template at f_iso=1 (masked pixels)
    denom_vec  : expo*omega*dE for the same pixels [cm^2 s sr MeV]
    Returns:
      (E2I, I_med) where I_med is dN/dE [MeV^-1 cm^-2 s^-1 sr^-1]
    """
    mu_iso_vec = np.asarray(mu_iso_vec, float)
    denom_vec = np.asarray(denom_vec, float)
    good = np.isfinite(mu_iso_vec) & np.isfinite(denom_vec) & (denom_vec > 0)
    if not np.any(good):
        raise RuntimeError("Cannot infer isotropic normalization: denom_vec has no valid entries.")
    I_med = float(np.median(mu_iso_vec[good] / denom_vec[good]))
    E2I = float((Ectr_mev**2) * I_med)
    return E2I, I_med


def build_totani_init_for_mu_counts(
    *,
    labels,
    mu,             # (ncomp, npix) counts-space templates for this energy bin in the fit mask
    denom_vec,      # (npix,) expo*omega*dE for same pixels
    Ectr_mev,
    iso_target_E2=1e-4,
):
    """
    Totani text says:
      - PS + GALPROP start at original normalization  -> for you, f=1.0
      - isotropic initial E^2 dN/dE = 1e-4          -> for you, convert to dimensionless f_iso
      - others start at 0
    """
    labels_l = [s.lower() for s in labels]
    ncomp = len(labels)
    f0 = np.zeros(ncomp, float)

    # PS + GALPROP: start at their baseline normalization
    for j, lab in enumerate(labels_l):
        if lab in ("ps", "point_sources", "pointsources", "gas", "ics"):
            f0[j] = 1.0

    # isotropic: convert physical target into your dimensionless scaling
    if "iso" in labels_l or "isotropic" in labels_l:
        jiso = labels_l.index("iso") if "iso" in labels_l else labels_l.index("isotropic")
        E2I, I_med = _infer_E2dNdE_for_iso_at_f1(mu[jiso], denom_vec, Ectr_mev)
        f0[jiso] = iso_target_E2 / E2I

        print("\nIsotropic init conversion:")
        print(f"  iso at f=1 implies E^2 dN/dE = {E2I:.6e}  [MeV cm^-2 s^-1 sr^-1]")
        print(f"  target E^2 dN/dE           = {iso_target_E2:.6e}")
        print(f"  ==> set f_iso_init         = {f0[jiso]:.6e}")

    # all other components: start at 0 (already)

    return f0


def tighten_negative_bound_for_single_halo(*, mu_halo, Cbase, safety=0.999):
    """
    If you allow f_halo < 0, you *still* must keep Cexp = Cbase + f_halo*mu_halo > 0.
    This computes a safe lower bound:
        f_halo >= -min(Cbase/mu_halo) over mu_halo>0
    """
    mu_halo = np.asarray(mu_halo, float)
    Cbase = np.asarray(Cbase, float)

    good = np.isfinite(mu_halo) & np.isfinite(Cbase) & (mu_halo > 0) & (Cbase > 0)
    if not np.any(good):
        # if we can't compute it, don't tighten
        return -np.inf

    lo = -float(np.min(Cbase[good] / mu_halo[good]))
    return safety * lo


# -----------------------------------------------------------------------------
# Plotting (same as yours, but burn uses args.burn)
# -----------------------------------------------------------------------------
def plot_mcmc_results(res, outdir, energy_bin, Ectr_mev, burn_for_plots=None):
    labels = res.labels
    ncomp = len(labels)
    chain = res.chain  # (nsteps, nwalkers, ncomp)
    nsteps, nwalkers, _ = chain.shape

    burn_idx = int(burn_for_plots) if burn_for_plots is not None else int(0.25 * nsteps)
    burn_idx = max(0, min(burn_idx, nsteps - 1))
    post_samples = chain[burn_idx:, :, :].reshape(-1, ncomp)

    # 1) Trace
    fig_trace, axes = plt.subplots(ncomp, 1, figsize=(12, 2 * ncomp), sharex=True)
    if ncomp == 1:
        axes = [axes]

    for j in range(ncomp):
        ax = axes[j]
        for w in range(nwalkers):
            ax.plot(chain[:, w, j], alpha=0.3, lw=0.5)
        ax.set_ylabel(labels[j], fontsize=8)
        ax.axhline(res.f_ml[j], color="red", ls="--", lw=1, alpha=0.7, label="ML")
        ax.axhline(res.f_p50[j], color="orange", ls="--", lw=1, alpha=0.7, label="median")
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Step", fontsize=10)
    fig_trace.suptitle(f"MCMC Trace Plots (k={energy_bin}, E={Ectr_mev/1000:.3f} GeV)", fontsize=12)
    fig_trace.tight_layout()
    trace_file = os.path.join(outdir, f"mcmc_trace_k{energy_bin:02d}.png")
    fig_trace.savefig(trace_file, dpi=150, bbox_inches="tight")
    plt.close(fig_trace)
    print(f"  Trace plot saved to: {trace_file}")

    # 2) Hist
    ncols = min(3, ncomp)
    nrows = int(np.ceil(ncomp / ncols))
    fig_hist, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for j in range(ncomp):
        ax = axes[j]
        samples = post_samples[:, j]
        ax.hist(samples, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(res.f_ml[j], color="red", ls="--", lw=2, label=f"ML: {res.f_ml[j]:.3e}")
        ax.axvline(res.f_p50[j], color="orange", ls="--", lw=2, label=f"p50: {res.f_p50[j]:.3e}")
        ax.axvline(res.f_p16[j], color="green", ls=":", lw=1.5, alpha=0.7, label=f"p16: {res.f_p16[j]:.3e}")
        ax.axvline(res.f_p84[j], color="green", ls=":", lw=1.5, alpha=0.7, label=f"p84: {res.f_p84[j]:.3e}")
        ax.set_title(labels[j], fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    for j in range(ncomp, len(axes)):
        axes[j].axis("off")

    fig_hist.suptitle(f"Posterior Distributions (k={energy_bin}, E={Ectr_mev/1000:.3f} GeV)", fontsize=12)
    fig_hist.tight_layout()
    hist_file = os.path.join(outdir, f"mcmc_posteriors_k{energy_bin:02d}.png")
    fig_hist.savefig(hist_file, dpi=150, bbox_inches="tight")
    plt.close(fig_hist)
    print(f"  Posterior plot saved to: {hist_file}")

    # 3) Corner (optional)
    try:
        import corner

        short_labels = []
        for lab in labels:
            if len(lab) > 20:
                short_labels.append("NFW" if "nfw" in lab.lower() else lab[:20])
            else:
                short_labels.append(lab)

        fig_corner = corner.corner(
            post_samples,
            labels=short_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3e",
            title_kwargs={"fontsize": 9},
            label_kwargs={"fontsize": 9},
            truths=res.f_ml,
            truth_color="red",
        )
        fig_corner.suptitle(f"Corner Plot (k={energy_bin}, E={Ectr_mev/1000:.3f} GeV)", fontsize=12, y=1.0)
        corner_file = os.path.join(outdir, f"mcmc_corner_k{energy_bin:02d}.png")
        fig_corner.savefig(corner_file, dpi=150, bbox_inches="tight")
        plt.close(fig_corner)
        print(f"  Corner plot saved to: {corner_file}")
    except ImportError:
        print("  Corner package not available, skipping corner plot")

    # 4) Acceptance
    fig_acc, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(res.acceptance_fraction, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(res.acceptance_fraction.mean(), color="red", ls="--", lw=2,
               label=f"Mean: {res.acceptance_fraction.mean():.3f}")
    ax.set_xlabel("Acceptance Fraction", fontsize=10)
    ax.set_ylabel("Number of Walkers", fontsize=10)
    ax.set_title(f"Walker Acceptance Fractions (k={energy_bin}, E={Ectr_mev/1000:.3f} GeV)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig_acc.tight_layout()
    acc_file = os.path.join(outdir, f"mcmc_acceptance_k{energy_bin:02d}.png")
    fig_acc.savefig(acc_file, dpi=150, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"  Acceptance plot saved to: {acc_file}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run MCMC fit on Fermi data (Totani-style init, counts-space mu).")
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--ext-mask",
        default=os.path.join(DATA_DIR, "processed", "templates", "mask_extended_sources.fits"),
        help="Extended-source keep mask FITS (True=keep, False=masked).",
    )
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--energy-bin", type=int, default=2, help="Energy bin index to fit (0-based)")
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=1500)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "mcmc_results"))
    ap.add_argument("--iso-target-e2", type=float, default=1e-4,
                    help="Totani isotropic init target E^2 dN/dE [MeV cm^-2 s^-1 sr^-1].")
    ap.add_argument("--tighten-halo-neg-bound", action="store_true",
                    help="Tighten negative bound for halo so Cexp stays >0 given the baseline components.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Component labels (keep your ordering)
    labels = [
        "gas",
        "iso",
        "ps",
        "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        "loopI",
        "ics",
        "fb_flat",
    ]
    halo_labels = ("nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",)

    print("=" * 60)
    print("MCMC Fit Configuration")
    print("=" * 60)
    print(f"Components: {labels}")
    print(f"Energy bin: {args.energy_bin}")
    print(f"MCMC: {args.nwalkers} walkers, {args.nsteps} steps, burn={args.burn}, thin={args.thin}")
    print(f"iso init target: E^2 dN/dE = {args.iso_target_e2:.3e}")
    print("=" * 60)

    # ---- Load counts + EBOUNDS
    print("\nLoading data...")
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial
    print(f"  Counts cube: {counts.shape} (nE={nE}, ny={ny}, nx={nx})")
    print(f"  Energy bins: {len(Ectr_mev)}")
    print(f"  Energy range: {Emin[0]:.1f} - {Emax[-1]:.1f} MeV")

    # ---- Exposure
    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")
    print(f"  Exposure: {expo.shape}")

    # ---- Solid angle map
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    print(f"  Solid angle map: {omega.shape}")

    # ---- Lon/lat and ROI
    lon, lat = lonlat_grids(wcs, ny, nx)
    roi2d = build_roi_box_mask(lon, lat, args.roi_lon, args.roi_lat)
    print(f"  ROI: |l|<={args.roi_lon}°, |b|<={args.roi_lat}°  (pixels {roi2d.sum()}/{roi2d.size})")

    # ---- Optional extended-source keep mask
    srcmask = np.ones((nE, ny, nx), bool)
    if args.ext_mask is not None and os.path.exists(str(args.ext_mask)):
        ext_keep3d = load_mask_any_shape(str(args.ext_mask), counts.shape)
        srcmask &= ext_keep3d
        frac_masked = float(np.mean((~ext_keep3d)[:, roi2d])) if np.any(roi2d) else float("nan")
        print(f"  Extended-source mask applied (masked frac in ROI={frac_masked:.4f})")

    # ---- Load templates
    print("\nLoading templates...")
    mu_list, headers = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )
    print(f"  Loaded {len(mu_list)} templates")
    for i, lab in enumerate(labels):
        print(f"    {lab}: {mu_list[i].shape}")

    # ---- Fit mask
    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=srcmask,
        counts=counts,
        expo=expo,
    )

    # ---- Select energy bin
    k = args.energy_bin
    if k < 0 or k >= nE:
        raise ValueError(f"Energy bin {k} out of range [0, {nE-1}]")
    Ectr_k = float(np.asarray(Ectr_mev).reshape(-1)[k])
    dE_k = float(np.asarray(dE_mev).reshape(-1)[k])

    print(f"\nFitting energy bin k={k}, E_ctr={Ectr_k:.1f} MeV ({Ectr_k/1000:.3f} GeV)")
    mask_k = fit_mask3d[k]
    npix_k = int(mask_k.sum())
    print(f"  Valid pixels in fit mask: {npix_k}")
    if npix_k == 0:
        raise RuntimeError(f"No valid pixels in energy bin {k}")

    # ---- Norm report (your existing diagnostic)
    print("\nTemplate normalisations (from mu_list):")
    report_template_normalisations_from_mu_list(
        counts_cube=counts,
        mu_list=mu_list,
        labels=labels,
        fit_mask3d=fit_mask3d,
        energy_bin=k,
        Ectr_mev=Ectr_k,
        expo_cube=expo,
        omega_map=omega,
        dE_mev=dE_k,
    )

    # ---- Observed counts vector over masked pixels
    Cobs = counts[k][mask_k].ravel().astype(float)
    print(f"\n  Cobs: shape={Cobs.shape}, sum={Cobs.sum():.1f}, mean={Cobs.mean():.3f}")

    # ---- Build mu matrix in counts-space over masked pixels
    ncomp = len(labels)
    mu = np.zeros((ncomp, npix_k), dtype=float)
    for j, lab in enumerate(labels):
        mu[j, :] = mu_list[j][k][mask_k].ravel().astype(float)
        print(f"  mu[{lab}]: sum={mu[j,:].sum():.3e}, mean={mu[j,:].mean():.3e}")

    # ---- denom vector for isotropic conversion (same pixels)
    denom_vec = (expo[k][mask_k].ravel().astype(float) *
                 omega[mask_k].ravel().astype(float) *
                 dE_k)

    # ---- Initialize parameters (THIS is the critical change)
    print("\nInitializing MCMC (Totani-style, consistent with your mu counts templates)...")
    f0 = build_totani_init_for_mu_counts(
        labels=labels,
        mu=mu,
        denom_vec=denom_vec,
        Ectr_mev=Ectr_k,
        iso_target_E2=args.iso_target_e2,
    )
    print(f"  Initial values: {dict(zip(labels, f0))}")

    # ---- Bounds
    bnds = totani_bounds(labels, halo_keys=halo_labels)

    # Optional: tighten halo negative bound to avoid Cexp<=0 catastrophes
    if args.tighten_halo_neg_bound:
        # Build baseline counts from non-halo components at f0
        halo_idx = labels.index(halo_labels[0])
        non_halo = [j for j in range(ncomp) if j != halo_idx]
        Cbase = np.zeros(npix_k, float)
        for j in non_halo:
            Cbase += f0[j] * mu[j]

        lo_safe = tighten_negative_bound_for_single_halo(mu_halo=mu[halo_idx], Cbase=Cbase, safety=0.999)
        lo, hi = bnds[halo_idx]
        bnds[halo_idx] = (max(lo, lo_safe), hi)
        print(f"\n  Tightened halo lower bound: {labels[halo_idx]} lo={bnds[halo_idx][0]:.6e} (was {lo})")

    print("\n  Bounds:")
    for lab, (lo, hi) in zip(labels, bnds):
        lo_str = f"{lo:.3e}" if np.isfinite(lo) else "-inf"
        hi_str = f"{hi:.3e}" if np.isfinite(hi) else "+inf"
        print(f"    {lab}: [{lo_str}, {hi_str}]")

    # ---- Run MCMC
    print("\nRunning MCMC...")
    t0 = time.time()
    res = totani_mcmc_fit(
        Cobs=Cobs,
        mu=mu,
        labels=labels,
        f_init=f0,
        bounds=bnds,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burn=args.burn,
        thin=args.thin,
    )
    dt = time.time() - t0
    print(f"MCMC wall time: {dt:.2f} s  ({dt/60:.2f} min)")

    print("\n" + "=" * 60)
    print("MCMC Results")
    print("=" * 60)
    print(f"Used emcee: {res.used_emcee}")
    print(f"Acceptance fraction: mean={res.acceptance_fraction.mean():.3f}, "
          f"min={res.acceptance_fraction.min():.3f}, max={res.acceptance_fraction.max():.3f}")
    print(f"Log-likelihood (ML): {res.loglike_ml:.3f}")

    print("\nMaximum Likelihood:")
    for lab, val in zip(res.labels, res.f_ml):
        print(f"  {lab:50s}: {val:+.6e}")

    print("\nMedian (50th percentile):")
    for lab, val in zip(res.labels, res.f_p50):
        print(f"  {lab:50s}: {val:+.6e}")

    print("\n16th percentile:")
    for lab, val in zip(res.labels, res.f_p16):
        print(f"  {lab:50s}: {val:+.6e}")

    print("\n84th percentile:")
    for lab, val in zip(res.labels, res.f_p84):
        print(f"  {lab:50s}: {val:+.6e}")
    print("=" * 60)

    # ---- Save results
    outfile = os.path.join(args.outdir, f"mcmc_results_k{k:02d}.npz")
    np.savez(
        outfile,
        labels=res.labels,
        f_ml=res.f_ml,
        f_p16=res.f_p16,
        f_p50=res.f_p50,
        f_p84=res.f_p84,
        loglike_ml=res.loglike_ml,
        chain=res.chain,
        logprob=res.logprob,
        acceptance_fraction=res.acceptance_fraction,
        used_emcee=res.used_emcee,
        energy_bin=k,
        Ectr_mev=Ectr_k,
        iso_target_e2=args.iso_target_e2,
    )
    print(f"\nResults saved to: {outfile}")

    # ---- Save text summary
    txtfile = os.path.join(args.outdir, f"mcmc_results_k{k:02d}.txt")
    with open(txtfile, "w") as f:
        f.write("MCMC Fit Results\n")
        f.write(f"Energy bin: k={k}, E_ctr={Ectr_k:.1f} MeV ({Ectr_k/1000:.3f} GeV)\n")
        f.write(f"Components: {', '.join(labels)}\n")
        f.write(f"MCMC: {args.nwalkers} walkers, {args.nsteps} steps, burn={args.burn}, thin={args.thin}\n")
        f.write(f"Used emcee: {res.used_emcee}\n")
        f.write(f"iso init target E^2 dN/dE: {args.iso_target_e2:.6e}\n")
        f.write(f"Acceptance fraction: mean={res.acceptance_fraction.mean():.3f}\n")
        f.write(f"Log-likelihood (ML): {res.loglike_ml:.3f}\n\n")
        f.write(f"{'Component':<50s}  {'ML':>12s}  {'p16':>12s}  {'p50':>12s}  {'p84':>12s}\n")
        f.write("-" * 100 + "\n")
        for i, lab in enumerate(res.labels):
            f.write(f"{lab:<50s}  {res.f_ml[i]:+12.6e}  {res.f_p16[i]:+12.6e}  "
                    f"{res.f_p50[i]:+12.6e}  {res.f_p84[i]:+12.6e}\n")
    print(f"Summary saved to: {txtfile}\n")

    # ---- Plots
    print("Generating diagnostic plots...")
    plot_mcmc_results(res, args.outdir, k, Ectr_k, burn_for_plots=args.burn)
    print("\nAll done!")


if __name__ == "__main__":
    main()