#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import time

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:
    from helpers.heartbar import heart_tqdm as _heart_tqdm  # type: ignore
except Exception:  # pragma: no cover
    _heart_tqdm = None

_tqdm = _heart_tqdm if _heart_tqdm is not None else tqdm

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

def rescale_templates_to_data_sum(mu, labels, Cobs_vec, *, targets=("loopa","loopb","fb_flat","fb","nfw")):
    """
    Rescale selected templates so sum(mu_j) ~= sum(data) in the current bin/mask.
    Returns (mu_scaled, scale_factors) where:
      mu_scaled[j] = mu[j] * scale_factors[label]
    """
    mu = np.asarray(mu, float).copy()
    labels_l = [str(x).lower() for x in labels]
    y = float(np.sum(Cobs_vec))
    if y <= 0 or not np.isfinite(y):
        raise ValueError(f"Bad data sum {y}; cannot rescale.")

    scales = {}
    for j, lab in enumerate(labels_l):
        hit = any(t in lab for t in targets)
        if not hit:
            continue
        s = float(np.sum(mu[j]))
        if (not np.isfinite(s)) or (s <= 0):
            continue
        fac = y / s
        mu[j] *= fac
        scales[labels[j]] = fac

    return mu, scales

def build_roi_box_mask(lon, lat, roi_lon=60.0, roi_lat=60.0):
    return (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)


def build_cell_id_map(*, lon, lat, roi_lon=60.0, roi_lat=60.0, cell_deg=10.0):
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    ny, nx = lon.shape

    nlon = int(round((2.0 * float(roi_lon)) / float(cell_deg)))
    nlat = int(round((2.0 * float(roi_lat)) / float(cell_deg)))
    if nlon != 12 or nlat != 12:
        raise ValueError(f"Expected 12x12 cells for roi_lon=roi_lat=60 and cell_deg=10, got {nlon}x{nlat}.")

    in_roi = (np.abs(lon) <= float(roi_lon)) & (np.abs(lat) <= float(roi_lat))
    cell_id = np.full((ny, nx), -1, dtype=int)

    # Map lon/lat to cell indices in [0..11]
    ix = np.floor((lon + float(roi_lon)) / float(cell_deg)).astype(int)
    iy = np.floor((lat + float(roi_lat)) / float(cell_deg)).astype(int)
    ix = np.clip(ix, 0, nlon - 1)
    iy = np.clip(iy, 0, nlat - 1)

    cell_id[in_roi] = iy[in_roi] * nlon + ix[in_roi]
    return cell_id


def aggregate_to_cells(*, x2d, mask2d, cell_id_map, ncells=144):
    x2d = np.asarray(x2d, float)
    mask2d = np.asarray(mask2d, bool)
    cell_id_map = np.asarray(cell_id_map, int)

    if x2d.shape != mask2d.shape or x2d.shape != cell_id_map.shape:
        raise ValueError("x2d, mask2d, and cell_id_map must have the same 2D shape")

    valid = mask2d & (cell_id_map >= 0) & np.isfinite(x2d)
    cid = cell_id_map[valid].ravel()
    w = x2d[valid].ravel()

    x_cells = np.bincount(cid, weights=w, minlength=int(ncells)).astype(float)
    npix_cells = np.bincount(cid, weights=np.ones_like(w), minlength=int(ncells)).astype(int)
    return x_cells, npix_cells


def template_scale_report(k, Cobs_vec, mu, labels):
    y = float(np.sum(Cobs_vec))
    print(f"bin k={k}: data sum={y:.3e}")
    for j, lab in enumerate(labels):
        s = float(np.sum(mu[j]))
        med = float(np.median(mu[j]))
        mx = float(np.max(mu[j]))
        print(
            f"  {str(lab):12s}  sum(mu@f=1)={s:.3e}  median={med:.3e}  max={mx:.3e}  "
            f"ratio_to_data={s / y if y > 0 else np.nan:.3e}"
        )


import numpy as np

# -----------------------------------------------------------------------------
# Totani-style init for YOUR parameterisation (mu are counts-space templates)
# -----------------------------------------------------------------------------
def infer_iso_E2dNdE_at_f1(mu_iso_vec, denom_vec, Ectr_mev, *, require_mu_positive=True):
    """
    Infer isotropic intensity from a counts-space iso template evaluated at f_iso=1.

    Inputs
    ------
    mu_iso_vec : (npix,) expected counts per pixel for isotropic template at f_iso=1
                (already includes exposure*Omega*dE)
    denom_vec  : (npix,) exposure*Omega*dE per pixel  [cm^2 s sr MeV]
    Ectr_mev   : scalar, bin center energy [MeV]

    Returns
    -------
    E2I   : scalar, E^2 dN/dE  [MeV cm^-2 s^-1 sr^-1]
    I_med : scalar, dN/dE      [MeV^-1 cm^-2 s^-1 sr^-1]
    """
    mu_iso_vec = np.asarray(mu_iso_vec, float).reshape(-1)
    denom_vec  = np.asarray(denom_vec,  float).reshape(-1)

    good = np.isfinite(mu_iso_vec) & np.isfinite(denom_vec) & (denom_vec > 0)
    if require_mu_positive:
        good &= (mu_iso_vec > 0)

    if not np.any(good):
        raise RuntimeError(
            "Cannot infer isotropic normalization: need finite denom>0"
            + (" and mu_iso>0" if require_mu_positive else "")
            + " on at least one pixel."
        )

    I_med = float(np.median(mu_iso_vec[good] / denom_vec[good]))
    E2I   = float((float(Ectr_mev) ** 2) * I_med)

    return E2I, I_med


def build_totani_init_for_mu_counts(
    *,
    labels,
    mu,              # (ncomp, npix) counts-space templates within the fit mask
    denom_vec,       # (npix,) expo*omega*dE for the same masked pixels
    Ectr_mev,
    iso_target_E2=1e-4,          # set None to skip Totani iso target
    iso_keys=("iso", "isotropic"),
    start_at_one_keys=("ps", "point_sources", "pointsources", "gas", "ics"),
    require_mu_positive_for_iso=True,
    verbosity: int = 2,
):
    """
    Totani-style initialization for coefficients f (dimensionless scalings):

      - PS + GALPROP-like components start at f=1.0 (your mu are already in counts space)
      - isotropic starts at E^2 dN/dE = 1e-4 (MeV cm^-2 s^-1 sr^-1) by rescaling f_iso
      - all other components start at 0

    Returns
    -------
    f0 : (ncomp,) initial coefficient vector
    info : dict with iso diagnostic numbers (or empty if no iso found / disabled)
    """
    labels = list(labels)
    labels_l = [str(s).lower() for s in labels]

    mu = np.asarray(mu, float)
    if mu.ndim != 2:
        raise ValueError("mu must be 2D (ncomp, npix).")
    ncomp, npix = mu.shape

    denom_vec = np.asarray(denom_vec, float).reshape(-1)
    if denom_vec.shape[0] != npix:
        raise ValueError(f"denom_vec length {denom_vec.shape[0]} does not match mu npix={npix}.")

    f0 = np.zeros(ncomp, dtype=float)

    # Start-at-one components
    start_at_one = {k.lower() for k in start_at_one_keys}
    for j, lab in enumerate(labels_l):
        if lab in start_at_one:
            f0[j] = 1.0

    info = {}

    # Optional: rescale iso to target E^2 dN/dE
    if iso_target_E2 is not None:
        jiso = None
        for k in iso_keys:
            kk = str(k).lower()
            if kk in labels_l:
                jiso = labels_l.index(kk)
                break

        if jiso is not None:
            E2_at_f1, I_med = infer_iso_E2dNdE_at_f1(
                mu[jiso], denom_vec, Ectr_mev, require_mu_positive=require_mu_positive_for_iso
            )

            if (not np.isfinite(E2_at_f1)) or (E2_at_f1 <= 0):
                raise RuntimeError(f"Inferred iso E^2 dN/dE at f=1 is invalid: {E2_at_f1}")

            f0[jiso] = float(iso_target_E2) / float(E2_at_f1)

            if int(verbosity) >= 2:
                iso_counts_sum_f1   = float(np.sum(mu[jiso]))
                iso_counts_sum_init = float(f0[jiso] * iso_counts_sum_f1)

                print("\nIsotropic init conversion (Totani-style):")
                print(f"  iso label                 = {labels[jiso]}")
                print(f"  iso at f=1 => E^2 dN/dE    = {E2_at_f1:.6e}  [MeV cm^-2 s^-1 sr^-1]")
                print(f"  target E^2 dN/dE           = {float(iso_target_E2):.6e}")
                print(f"  ==> f_iso_init             = {f0[jiso]:.6e}")
                print(f"  iso sum counts (f=1)       = {iso_counts_sum_f1:.3e}")
                print(f"  iso sum counts (init)      = {iso_counts_sum_init:.3e}")

            info = dict(
                jiso=int(jiso),
                iso_label=str(labels[jiso]),
                E2_at_f1=float(E2_at_f1),
                I_med=float(I_med),
                f_iso_init=float(f0[jiso]),
            )
        else:
            if int(verbosity) >= 2:
                print("No isotropic label found; skipping iso target init.")

    return f0, info
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
def plot_mcmc_results(res, outdir, energy_bin, Ectr_mev, burn_for_plots=None, *, verbosity: int = 2):
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
    if int(verbosity) >= 2:
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
    if int(verbosity) >= 2:
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
        if int(verbosity) >= 2:
            print(f"  Corner plot saved to: {corner_file}")
    except ImportError:
        if int(verbosity) >= 2:
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
    if int(verbosity) >= 2:
        print(f"  Acceptance plot saved to: {acc_file}")


def cancellation_check(k, counts, tpl_list, labels, coeff_k, mask_k, *, verbosity: int = 2):
    y = float(np.nansum(counts[k][mask_k]))

    comp_sums = {}
    for j, lab in enumerate(labels):
        comp_sums[str(lab)] = float(np.nansum(float(coeff_k[j]) * tpl_list[j][k][mask_k]))

    model = float(np.sum(list(comp_sums.values())))
    pos = float(np.sum([v for v in comp_sums.values() if v > 0]))
    neg = float(np.sum([v for v in comp_sums.values() if v < 0]))

    mu_pix = np.zeros_like(tpl_list[0][k], dtype=float)
    for j in range(len(labels)):
        mu_pix += float(coeff_k[j]) * tpl_list[j][k]
    mu_min = float(np.nanmin(mu_pix[mask_k]))
    mu_med = float(np.nanmedian(mu_pix[mask_k]))

    lines = []
    lines.append(f"[CANCELLATION CHECK] bin k={k}:")
    lines.append(f"  data={y:.3e}  model={model:.3e}  model/data={model / y if y > 0 else np.nan:.3g}")
    lines.append(
        f"  sum(pos comps)={pos:.3e}  sum(neg comps)={neg:.3e}  "
        f"cancellation frac={(pos + neg) / pos if pos != 0 else np.nan:.3g}"
    )
    lines.append(f"  mu_pix min={mu_min:.3e}  median={mu_med:.3e}")
    lines.append("  top components by |counts|:")
    for lab, v in sorted(comp_sums.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]:
        lines.append(f"    {lab:20s}  {v:+.3e}")

    if int(verbosity) >= 2:
        print("\n" + "\n".join(lines))

    return {"comp_sums": comp_sums, "lines": lines}


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
    ap.add_argument(
        "--exclude-disk",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Exclude |b|<disk_cut from the ROI. Default is to include the disk in the fit.",
    )
    ap.add_argument("--disk-cut", type=float, default=10.0, help="Disk cut in degrees used when --exclude-disk is enabled")
    ap.add_argument("--cell-deg", type=float, default=10.0, help="Cell size in degrees (Totani uses 10°)")
    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--energy-bin", type=int, default=2, help="Energy bin index to fit (0-based)")
    ap.add_argument("--nwalkers", type=int, default=64)
    ap.add_argument("--nsteps", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=1500)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "mcmc_results"))
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Component labels to fit. Each label maps to a template file via "
            "--templates-dir/mu_{label}_counts.fits. If omitted, uses the built-in default list."
        ),
    )
    ap.add_argument(
        "--halo-label",
        default="nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        help=(
            "Halo label (template key) used for bounds/diagnostics when a single halo component is present. "
            "Ignored if --halo-labels is provided."
        ),
    )
    ap.add_argument(
        "--halo-labels",
        nargs="+",
        default=None,
        help=(
            "Explicit list of halo component keys (e.g. one nfw_* label). "
            "If omitted, halo labels are inferred from --labels (keys starting with 'nfw_'), "
            "or fall back to --halo-label if none found."
        ),
    )
    ap.add_argument(
        "--negative-keys",
        nargs="+",
        default=None,
        help=(
            "Substrings that identify components allowed to go negative. "
            "Default is ['nfw', 'fb_neg'] (applied as substring matches)."
        ),
    )
    ap.add_argument("--iso-target-e2", type=float, default=1e-4,
                    help="Optional: rescale iso init to this target E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]. If not set, iso starts at f=1 (recommended for physical templates).")
    ap.add_argument(
        "--iso-anchor-e2",
        type=float,
        default=1e-4,
        help=(
            "Anchor the isotropic prior center to this E^2 dN/dE value [MeV cm^-2 s^-1 sr^-1] by converting "
            "to the corresponding f_iso for each energy bin."
        ),
    )
    ap.add_argument(
        "--iso-anchor",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable anchoring the isotropic prior center to --iso-anchor-e2 (default: enabled).",
    )
    ap.add_argument(
        "--iso-floor-e2",
        type=float,
        default=0.0,
        help=(
            "Isotropic floor in E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]. Converted per energy bin to a lower bound on f_iso. "
            "Set <=0 to disable."
        ),
    )
    ap.add_argument(
        "--iso-prior-sigma-dex",
        type=float,
        default=0.5,
        help="Gaussian prior on log10(f_iso / f_iso0) with this sigma (dex). Set to 0 to disable.",
    )
    ap.add_argument(
        "--iso-prior-mode",
        choices=["f", "f_upper", "f_lower"],
        default="f",
        help=(
            "Iso prior mode. 'f' is a symmetric log-prior centered on Totani init f0. "
            "'f_upper' anchors the starting value but only penalizes excursions above f0 (allows drifting down). "
            "'f_lower' anchors the starting value but only penalizes excursions below f0 (allows drifting up)."
        ),
    )
    ap.add_argument(
        "--nonstable-prior-sigma",
        type=float,
        default=None,
        help=(
            "Optional Gaussian priors (dex) in log10(f/f0) for gas/ics/ps/iso, centered on Totani init f0. "
            "Useful to stabilize fits."
        ),
    )
    ap.add_argument(
        "--nonstable-prior-centers",
        default=None,
        help=(
            "Optional JSON dict mapping label->center for the nonstable prior (shared sigma set by --nonstable-prior-sigma). "
            "If provided, these values override/extend the default centers (which are taken from the Totani init f0 for gas/ics/ps/iso when present)."
        ),
    )
    ap.add_argument(
        "--component-priors",
        default=None,
        help=(
            "Optional JSON dict mapping label -> {center, sigma_dex, mode}. "
            "mode in {'f','f_upper','f_lower'} where 'f_lower' only penalizes being below center and 'f_upper' only penalizes being above."
        ),
    )
    ap.add_argument(
        "--require-autocorr",
        action="store_true",
        help="If set, fail the run if autocorr-time convergence check is not satisfied/available.",
    )
    ap.add_argument(
        "--early-stop",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If enabled, run MCMC in chunks and stop early once autocorr convergence is good enough.",
    )
    ap.add_argument(
        "--autocorr-target",
        type=float,
        default=50.0,
        help="Early-stop (and/or convergence) target for N/tau (Totani rule of thumb is 50).",
    )
    ap.add_argument(
        "--autocorr-check-every",
        type=int,
        default=1000,
        help="When --early-stop is enabled, check autocorr convergence every N steps.",
    )
    ap.add_argument(
        "--autocorr-min-steps",
        type=int,
        default=2000,
        help="When --early-stop is enabled, do not attempt autocorr checks before this many steps.",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plot generation (useful for long batch runs).",
    )
    ap.add_argument(
        "--progress",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show a tqdm progress bar during the emcee MCMC run (if emcee is available).",
    )
    ap.add_argument(
        "--cancellation-check",
        action="store_true",
        help="Print per-bin diagnostic showing positive/negative component cancellations and mu_pix positivity using ML coefficients.",
    )
    ap.add_argument(
        "--tighten-negative-bounds",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Tighten negative lower bounds for any component allowed to go negative "
            "(identified by --negative-keys, default: nfw and fb_neg), so Cexp stays >0 "
            "given the baseline components at their Totani init values."
        ),
    )
    ap.add_argument(
        "--tighten-halo-neg-bound",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Backward-compat alias. If set, overrides --tighten-negative-bounds."
        ),
    )
    ap.add_argument("--save-chain", action="store_true",
                    help="Save MCMC chain array (can be large; off by default to save disk space).")
    ap.add_argument("--save-logprob", action="store_true",
                    help="Save MCMC logprob array (can be large; off by default to save disk space).")
    ap.add_argument("--thin-save", type=int, default=10,
                    help="Thinning factor for saved chain/logprob (default 10 to reduce file size).")

    ap.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=(
            "Verbosity level: 0 prints nothing; 1 prints only wall time + fit config + output paths (default); "
            "2 prints all diagnostics."
        ),
    )
    args = ap.parse_args()

    verbosity = int(args.verbosity)

    def vprint(level: int, *p, **k):
        if verbosity >= int(level):
            print(*p, **k)

    os.makedirs(args.outdir, exist_ok=True)

    # Component labels
    if args.labels is None:
        labels = [
            "gas",
            "iso",
            "ps",
            "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
            "loopA",
            "loopB",
            "ics",
            "fb_flat",
        ]
    else:
        labels = [str(x) for x in args.labels]

    if args.halo_labels is not None:
        halo_labels = tuple(str(x) for x in args.halo_labels)
    else:
        # Self-identify halo-like labels by substring match (not necessarily full 'nfw_*' prefix)
        inferred = [lab for lab in labels if ("nfw" in str(lab).lower())]
        if len(inferred) > 0:
            halo_labels = tuple(inferred)
        else:
            halo_labels = (str(args.halo_label),)

    if args.negative_keys is None:
        negative_keys = ["nfw", "fb_neg"]
    else:
        negative_keys = [str(x) for x in args.negative_keys]

    # De-duplicate while preserving order
    _seen = set()
    negative_keys = [k for k in negative_keys if (k not in _seen and not _seen.add(k))]

    if verbosity >= 2:
        print("=" * 60)
        print("MCMC Fit Configuration")
        print("=" * 60)
        print(f"Components: {labels}")
        print(f"Negative keys: {negative_keys}")
        print(f"Energy bin: {args.energy_bin}")
        print(f"MCMC: {args.nwalkers} walkers, {args.nsteps} steps, burn={args.burn}, thin={args.thin}")
        print(f"iso init target: E^2 dN/dE = {args.iso_target_e2:.3e}")
        print("=" * 60)

    # ---- Load counts + EBOUNDS
    pbar = None
    if (verbosity > 0) and (_tqdm is not None):
        pbar = _tqdm(total=4, desc="Loading FITS")

    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial
    if pbar is not None:
        pbar.update(1)
    if verbosity >= 2:
        print(f"  Counts cube: {counts.shape} (nE={nE}, ny={ny}, nx={nx})")
    if verbosity >= 2:
        print(f"  Energy bins: {len(Ectr_mev)}")
        print(f"  Energy range: {Emin[0]:.1f} - {Emax[-1]:.1f} MeV")

    # ---- Exposure
    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")
    if pbar is not None:
        pbar.update(1)
    vprint(2, f"  Exposure: {expo.shape}")

    # ---- Solid angle map
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    vprint(2, f"  Solid angle map: {omega.shape}")

    # ---- Lon/lat and ROI
    lon, lat = lonlat_grids(wcs, ny, nx)
    roi2d = build_roi_box_mask(lon, lat, args.roi_lon, args.roi_lat)
    if args.exclude_disk:
        roi2d = roi2d & (np.abs(lat) >= float(args.disk_cut))
    disk_tag = f"excluded |b|<{float(args.disk_cut):g}°" if args.exclude_disk else "disk included"
    vprint(2, f"  ROI: |l|<={args.roi_lon}°, |b|<={args.roi_lat}° ({disk_tag})  (pixels {roi2d.sum()}/{roi2d.size})")

    # ---- Cell IDs for Totani-style cellwise likelihood
    cell_id_map = build_cell_id_map(lon=lon, lat=lat, roi_lon=args.roi_lon, roi_lat=args.roi_lat, cell_deg=args.cell_deg)

    # ---- Optional extended-source keep mask
    srcmask = np.ones((nE, ny, nx), bool)
    if args.ext_mask is not None and os.path.exists(str(args.ext_mask)):
        try:
            ext_keep3d = load_mask_any_shape(str(args.ext_mask), counts.shape)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read --ext-mask as a FITS file: {args.ext_mask}. "
                f"Original error: {e}. "
                "If you want to run without an extended-source mask, pass --ext-mask '' (or a non-existent path)."
            ) from e
        srcmask &= ext_keep3d
        frac_masked = float(np.mean((~ext_keep3d)[:, roi2d])) if np.any(roi2d) else float("nan")
        vprint(2, f"  Extended-source mask applied (masked frac in ROI={frac_masked:.4f})")

    if pbar is not None:
        pbar.update(1)

    # ---- Load templates
    mu_list, headers = load_mu_templates_from_fits(
        template_dir=args.templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
        progress=bool(verbosity > 0),
        progress_desc="Loading templates",
    )
    if pbar is not None:
        pbar.update(1)
        pbar.close()

    vprint(2, f"\nLoaded {len(mu_list)} templates")
    if verbosity >= 2:
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

    if verbosity >= 1:
        print("\nFit config:")
        print(f"  outdir: {args.outdir}")
        print(f"  templates_dir: {args.templates_dir}")
        print(f"  energy bin: k={k}, E_ctr={Ectr_k:.1f} MeV ({Ectr_k/1000:.3f} GeV)")
        print(f"  labels: {', '.join(labels)}")
        print(f"  nwalkers={args.nwalkers} nsteps={args.nsteps} burn={args.burn} thin={args.thin}")
        print(f"  early_stop={bool(args.early_stop)} require_autocorr={bool(args.require_autocorr)}")
        print(f"  iso_anchor={bool(args.iso_anchor)} iso_anchor_e2={args.iso_anchor_e2}")
        print(f"  iso_prior_mode={str(args.iso_prior_mode)} iso_prior_sigma_dex={args.iso_prior_sigma_dex}")
    mask_k = fit_mask3d[k]

    # Ensure we also exclude pixels where any template is non-finite in this bin.
    # (build_fit_mask3d already enforces finite counts/expo and expo>0.)
    tmpl_ok = np.ones((ny, nx), dtype=bool)
    for j in range(len(labels)):
        tmpl_ok &= np.isfinite(mu_list[j][k])
    n_before = int(np.count_nonzero(mask_k))
    mask_k = mask_k & tmpl_ok
    n_after = int(np.count_nonzero(mask_k))
    if n_after != n_before:
        vprint(2, f"  Mask tightened by template finiteness: {n_before} -> {n_after} pixels")

    npix_k = int(mask_k.sum())
    vprint(2, f"  Valid pixels in fit mask: {npix_k}")
    if npix_k == 0:
        raise RuntimeError(f"No valid pixels in energy bin {k}")

    # ---- Norm report (your existing diagnostic)
    if verbosity >= 2:
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

    # ---- Cellwise vectors for Totani-style likelihood
    # Observed counts aggregated into 10°x10° cells
    Cobs_cells, npix_cells = aggregate_to_cells(
        x2d=counts[k],
        mask2d=mask_k,
        cell_id_map=cell_id_map,
        ncells=144,
    )
    cell_keep = npix_cells > 0
    if verbosity >= 2:
        print("\nCell occupancy (npix per cell):")
        print(npix_cells.reshape(12, 12))

    Cobs = Cobs_cells[cell_keep].astype(float)
    vprint(2, f"\n  Cobs_cells: shape={Cobs.shape}, sum={Cobs.sum():.1f}, mean={Cobs.mean():.3f}")

    # Build mu_cells (ncomp, ncells_kept)
    ncomp = len(labels)
    mu_cells_full = np.zeros((ncomp, 144), dtype=float)
    for j, lab in enumerate(labels):
        mu_j_cells, _npix = aggregate_to_cells(
            x2d=mu_list[j][k],
            mask2d=mask_k,
            cell_id_map=cell_id_map,
            ncells=144,
        )
        mu_cells_full[j, :] = mu_j_cells
        vprint(2, f"  mu_cells[{lab}]: sum={mu_j_cells.sum():.3e}")

    mu = mu_cells_full[:, cell_keep].astype(float)

    # denom_vec: (expo * omega * dE) aggregated into cells and restricted to kept cells.
    denom_map = expo[k] * omega * dE_k
    denom_cells, _npix_cells_denom = aggregate_to_cells(
        x2d=denom_map,
        mask2d=mask_k,
        cell_id_map=cell_id_map,
        ncells=144,
    )
    denom_vec = denom_cells[cell_keep].astype(float)

    # Note on isotropic anchoring:
    # We do NOT rescale the iso template here. Instead, --iso-anchor only sets the isotropic
    # prior center in E^2 dN/dE-space (converted into f-space for this bin). This keeps
    # f_iso interpretable as a scale factor on the *input template* (so f_iso_init can be
    # ~O(10-100) at low energies if mu_iso@f=1 corresponds to a much lower E^2 dN/dE).
    labels_l = [str(x).lower() for x in labels]

    if verbosity >= 2:
        print("\nTemplate scales (cell space, f=1):")
        template_scale_report(k=k, Cobs_vec=Cobs, mu=mu, labels=labels)

    # ---- Initialize parameters
    f0, _init_info = build_totani_init_for_mu_counts(
        labels=labels,
        mu=mu,
        denom_vec=denom_vec,
        Ectr_mev=Ectr_k,
        iso_target_E2=args.iso_target_e2,
        verbosity=verbosity,
    )
    vprint(2, "\nInitializing MCMC (Totani-style, consistent with your mu counts templates)...")
    vprint(2, "f_init by label:", {lab: float(f0[j]) for j, lab in enumerate(labels)})

    # Prior centers (Totani init values)
    nonstable_centers = {}
    for key in ("gas", "ics", "ps"):
        if key in labels_l:
            nonstable_centers[key] = float(f0[labels_l.index(key)])
    iso_center = None
    # for key in ("iso", "isotropic"):
    #     if key in labels_l:
    #         iso_center = float(f0[labels_l.index(key)])
    #         break

    if bool(args.iso_anchor) and (args.iso_anchor_e2 is not None):
        jiso = None
        for key in ("iso", "isotropic"):
            if key in labels_l:
                jiso = labels_l.index(key)
                break
        if jiso is not None:
            # Convert target E^2 dN/dE to the corresponding f_iso center for this energy bin.
            E2I, _I_med = infer_iso_E2dNdE_at_f1(mu[jiso], denom_vec, Ectr_k)
            if not np.isfinite(E2I) or (E2I <= 0):
                raise RuntimeError("Cannot anchor iso prior: inferred E^2 dN/dE at f=1 is non-finite or non-positive.")
            iso_center = float(args.iso_anchor_e2) / float(E2I)
            vprint(2, f"[iso anchor] E2(f=1)={E2I:.6e}  anchor={float(args.iso_anchor_e2):.6e}  => iso_prior_center={iso_center:.6e}")

    # ---- Bounds
    bnds = totani_bounds(labels, negative_keys=tuple(negative_keys))

    # Optional: isotropic floor in E2-space -> enforce as a lower bound in f-space for this energy bin.
    if (args.iso_floor_e2 is not None) and (float(args.iso_floor_e2) > 0):
        jiso = None
        for key in ("iso", "isotropic"):
            if key in labels_l:
                jiso = labels_l.index(key)
                break
        if jiso is not None:
            E2I, _I_med = infer_iso_E2dNdE_at_f1(mu[jiso], denom_vec, Ectr_k)
            if not np.isfinite(E2I) or (E2I <= 0):
                raise RuntimeError("Cannot apply iso floor: inferred E^2 dN/dE at f=1 is non-finite or non-positive.")
            f_floor = float(args.iso_floor_e2) / float(E2I)
            lo, hi = bnds[jiso]
            bnds[jiso] = (max(float(lo), float(f_floor)), hi)
            vprint(1, f"[iso floor] E2(f=1)={E2I:.6e}  floor={float(args.iso_floor_e2):.6e}  => f_iso>={float(f_floor):.6e}")

    tighten_negative_bounds = bool(args.tighten_negative_bounds)
    if args.tighten_halo_neg_bound is not None:
        tighten_negative_bounds = bool(args.tighten_halo_neg_bound)

    if tighten_negative_bounds:
        neg_idxs = []
        for j, lab in enumerate(labels):
            key = str(lab).lower()
            if any(str(nk).lower() in key for nk in negative_keys):
                neg_idxs.append(j)

        if len(neg_idxs) == 0:
            vprint(2, "\n  [INFO] No negative-allowed components identified; skipping negative-bound tightening.")
        else:
            vprint(2, "\n  Tightening negative bounds (Cexp>0 safety)...")
            for jneg in neg_idxs:
                # mu is in cell space (ncomp, ncells_kept). Tighten bounds in the same space.
                Cbase = np.zeros(int(mu.shape[1]), float)
                for j in range(ncomp):
                    if j == jneg:
                        continue
                    Cbase += f0[j] * mu[j]

                lo_safe = tighten_negative_bound_for_single_halo(mu_halo=mu[jneg], Cbase=Cbase, safety=0.999)
                lo, hi = bnds[jneg]
                bnds[jneg] = (max(lo, lo_safe), hi)
                vprint(2, f"    {labels[jneg]}: lo={bnds[jneg][0]:.6e} (was {lo})")

    if verbosity >= 2:
        print("\n  Bounds:")
        for lab, (lo, hi) in zip(labels, bnds):
            lo_str = f"{lo:.3e}" if np.isfinite(lo) else "-inf"
            hi_str = f"{hi:.3e}" if np.isfinite(hi) else "+inf"
            print(f"    {lab}: [{lo_str}, {hi_str}]")
    


    # ---- Run MCMC
    vprint(2, "\nRunning MCMC...")
    t0 = time.time()

    if args.nonstable_prior_centers is not None:
        try:
            extra_centers = json.loads(str(args.nonstable_prior_centers))
        except Exception as e:
            raise SystemExit(f"--nonstable-prior-centers must be valid JSON: {repr(e)}")
        if not isinstance(extra_centers, dict):
            raise SystemExit("--nonstable-prior-centers must decode to a JSON object/dict")
        for k, v in extra_centers.items():
            lk = str(k).lower()
            if lk in ("iso", "isotropic"):
                continue
            nonstable_centers[lk] = float(v)

    component_priors = None
    if args.component_priors is not None:
        try:
            component_priors = json.loads(str(args.component_priors))
        except Exception as e:
            raise SystemExit(f"--component-priors must be valid JSON: {repr(e)}")
        if not isinstance(component_priors, dict):
            raise SystemExit("--component-priors must decode to a JSON object/dict")

    # Normalize component_priors keys for consistent matching/printing (labels are matched case-insensitively downstream).
    component_priors_l = None
    if component_priors is not None:
        component_priors_l = {str(k).lower(): v for k, v in component_priors.items()}

    if verbosity >= 1:
        print("\nPriors (per label):")
        for j, lab in enumerate(labels):
            lk = str(lab).lower()
            lo, hi = bnds[j]
            lo_str = f"{float(lo):.6e}" if np.isfinite(lo) else "-inf"
            hi_str = f"{float(hi):.6e}" if np.isfinite(hi) else "+inf"

            pieces = [f"bounds=[{lo_str}, {hi_str}]"
            ]

            if lk in ("iso", "isotropic"):
                if (args.iso_prior_sigma_dex is not None) and (float(args.iso_prior_sigma_dex) > 0):
                    ctr = iso_center if (iso_center is not None) else float(f0[j])
                    pieces.append(
                        f"iso_log10_ratio_gauss(mode={str(args.iso_prior_mode)}, center={float(ctr):.6e}, sigma_dex={float(args.iso_prior_sigma_dex):.6g})"
                    )

            if (args.nonstable_prior_sigma is not None) and (float(args.nonstable_prior_sigma) > 0):
                if (lk not in ("iso", "isotropic")) and (lk in nonstable_centers):
                    pieces.append(
                        f"nonstable_log10_ratio_gauss(center={float(nonstable_centers[lk]):.6e}, sigma_dex={float(args.nonstable_prior_sigma):.6g})"
                    )

            if component_priors is not None:
                spec = component_priors_l.get(lk, None) if (component_priors_l is not None) else None
                if isinstance(spec, dict):
                    if ("center" in spec) and ("sigma_dex" in spec):
                        try:
                            ctr = float(spec.get("center"))
                            sig = float(spec.get("sigma_dex"))
                            mode = spec.get("mode", "f")
                            if np.isfinite(ctr) and np.isfinite(sig) and (sig > 0):
                                pieces.append(
                                    f"component_log10_ratio_gauss(mode={str(mode)}, center={ctr:.6e}, sigma_dex={sig:.6g})"
                                )
                        except Exception:
                            pass

            print(f"  {lab}: " + " | ".join(pieces))

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
        require_autocorr=bool(args.require_autocorr),
        early_stop=bool(args.early_stop),
        autocorr_target=float(args.autocorr_target),
        autocorr_check_every=int(args.autocorr_check_every),
        autocorr_min_steps=int(args.autocorr_min_steps),
        iso_prior_sigma_dex=args.iso_prior_sigma_dex,
        iso_prior_mode=str(args.iso_prior_mode),
        iso_prior_center=iso_center,
        nonstable_prior_sigma_dex=args.nonstable_prior_sigma,
        nonstable_prior_centers=nonstable_centers,
        component_priors=component_priors_l,
        init_jitter_frac=1e-3,
        progress=bool(args.progress) and (verbosity > 0),
        verbosity=verbosity,
    )
    dt = time.time() - t0
    if verbosity >= 1:
        print(f"MCMC wall time: {dt:.2f} s  ({dt/60:.2f} min)")
    vprint(2, "acc per walker:", res.acceptance_fraction)
    vprint(2, "n dead:", np.sum(res.acceptance_fraction < 0.02))

    if verbosity >= 1:
        labels_l_post = [str(x).lower() for x in res.labels]
        jiso_post = None
        for key in ("iso", "isotropic"):
            if key in labels_l_post:
                jiso_post = labels_l_post.index(key)
                break
        if jiso_post is not None:
            E2_f1, _I_med_f1 = infer_iso_E2dNdE_at_f1(mu[jiso_post], denom_vec, Ectr_k)
            f_iso_ml = float(res.f_ml[jiso_post])
            f_iso_p50 = float(res.f_p50[jiso_post])
            E2_ml = f_iso_ml * float(E2_f1)
            E2_p50 = f_iso_p50 * float(E2_f1)
            target = float(args.iso_anchor_e2) if (bool(args.iso_anchor) and (args.iso_anchor_e2 is not None)) else None
            print(
                f"[iso] Inferred E^2 dN/dE (this bin, k={k}, Ectr={Ectr_k/1000:.3f} GeV): "
                f"f=1 -> {float(E2_f1):.6e};  f_ML -> {float(f_iso_ml):.6e};  f_p50 -> {float(f_iso_p50):.6e};  "
                f"E2_ML -> {float(E2_ml):.6e};  E2_p50 -> {float(E2_p50):.6e} "
                "[MeV cm^-2 s^-1 sr^-1]"
            )
            if target is not None:
                print(f"[iso] Anchor target: E^2 dN/dE = {float(target):.6e} at f_iso=1 (fit is free to choose f_iso != 1).")

    if verbosity >= 2:
        print("\n" + "=" * 60)
        print("MCMC Results")
        print("=" * 60)
        print(f"Used emcee: {res.used_emcee}")
        print(
            f"Acceptance fraction: mean={res.acceptance_fraction.mean():.3f}, "
            f"min={res.acceptance_fraction.min():.3f}, max={res.acceptance_fraction.max():.3f}"
        )
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

    cancellation_diag = None
    if bool(args.cancellation_check):
        cancellation_diag = cancellation_check(
            k=k,
            counts=counts,
            tpl_list=mu_list,
            labels=labels,
            coeff_k=res.f_ml,
            mask_k=mask_k,
            verbosity=verbosity,
        )

    # ---- Save results (compressed, minimal by default)
    outfile = os.path.join(args.outdir, f"mcmc_results_k{k:02d}.npz")
    # NOTE: np.savez_compressed appends '.npz' if the provided path doesn't end with '.npz'.
    # Make the tmp path end with '.npz' so os.replace can find it.
    outfile_tmp = outfile + ".tmp.npz"
    
    # Build save dict with essential arrays (downcast to float32 to save space)
    save_dict = {
        "labels": res.labels,
        "f_ml": res.f_ml.astype(np.float32),
        "f_p16": res.f_p16.astype(np.float32),
        "f_p50": res.f_p50.astype(np.float32),
        "f_p84": res.f_p84.astype(np.float32),
        "loglike_ml": np.float32(res.loglike_ml),
        "acceptance_fraction": res.acceptance_fraction.astype(np.float32),
        "used_emcee": res.used_emcee,
        "energy_bin": k,
        "Ectr_mev": np.float32(Ectr_k),
        "iso_target_e2": np.float32(args.iso_target_e2) if args.iso_target_e2 is not None else None,
    }
    
    # Optionally save chain/logprob (thinned to reduce file size)
    if args.save_chain:
        chain_thinned = res.chain[::args.thin_save, :, :].astype(np.float32)
        save_dict["chain"] = chain_thinned
        vprint(2, f"  Saving chain: {chain_thinned.shape} (thinned by {args.thin_save})")
    
    if args.save_logprob:
        logprob_thinned = res.logprob[::args.thin_save, :].astype(np.float32)
        save_dict["logprob"] = logprob_thinned
        vprint(2, f"  Saving logprob: {logprob_thinned.shape} (thinned by {args.thin_save})")
    
    try:
        if os.path.exists(outfile_tmp):
            os.remove(outfile_tmp)
        np.savez_compressed(outfile_tmp, **save_dict)
        os.replace(outfile_tmp, outfile)
    finally:
        if os.path.exists(outfile_tmp):
            try:
                os.remove(outfile_tmp)
            except Exception:
                pass
    if verbosity >= 1:
        print(f"Results saved to: {outfile}")

    # ---- Save text summary
    txtfile = os.path.join(args.outdir, f"mcmc_results_k{k:02d}.txt")
    txtfile_tmp = txtfile + ".tmp"
    try:
        if os.path.exists(txtfile_tmp):
            os.remove(txtfile_tmp)
        with open(txtfile_tmp, "w") as f:
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

            if cancellation_diag is not None:
                f.write("\n")
                for line in cancellation_diag.get("lines", []):
                    f.write(str(line) + "\n")
        os.replace(txtfile_tmp, txtfile)
    finally:
        if os.path.exists(txtfile_tmp):
            try:
                os.remove(txtfile_tmp)
            except Exception:
                pass
    if verbosity >= 1:
        print(f"Summary saved to: {txtfile}\n")

    # ---- Plots
    if bool(args.no_plots):
        vprint(2, "Skipping diagnostic plots (--no-plots).")
    else:
        vprint(2, "Generating diagnostic plots...")
        plot_mcmc_results(res, args.outdir, k, Ectr_k, burn_for_plots=args.burn, verbosity=verbosity)
    vprint(2, "\nAll done!")


if __name__ == "__main__":
    main()