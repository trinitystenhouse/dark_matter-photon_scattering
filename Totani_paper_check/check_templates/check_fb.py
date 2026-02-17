#!/usr/bin/env python3
"""
check_templates/check_templates.py

End-to-end validation for:
  1) total best-fit model counts cube (mu_modelsum_counts.fits)
  2) per-bin NNLS coefficients (fit_coeffs_per_bin.npz or .txt optional)
  3) flat bubbles template (bubbles_flat_* or a mu_flat you built)
  4) Totani-style FB residual templates (mu_fbpos..., mu_fbneg...)

It performs:
  A) Data vs Model sanity per energy bin:
       - sumM/sumD in mask
       - chi2/dof with Poisson-ish weights (D+1)
  B) Reconstruction identity:
       - sum_i A[k,i] * mu_i[k] == modelsum[k] (on mask)
  C) Flat FB geometry checks:
       - binary-ish values, nonzero pixels, symmetry-ish quick stats
  D) FB residual templates checks (at ~4.3 GeV bin):
       - non-negativity of fbpos/fbneg
       - identity: fbpos - fbneg == (data - model)  (within mask; if norm=none)
  E) Quick diagnostic plots saved to OUTDIR:
       - coefficients vs energy
       - sum ratio and chi2/dof vs energy
       - 4.3 GeV maps: data, model, residual, fbpos, fbneg

Usage example:

python check_templates/check_templates.py \
  --counts  fermi_data/totani/processed/counts_ccube_1000to1000000.fits \
  --modelsum plots_fig1/mu_modelsum_counts.fits \
  --mask     fermi_data/totani/processed/templates/mask_extended_sources.fits \
  --expo     fermi_data/totani/processed/expcube_1000to1000000.fits \
  --roi-lon 60 --roi-lat 60 --disk-cut 10 \
  --coeff-npz plots_fig1/fit_coeffs_per_bin.npz \
  --components mu_gas_counts.fits mu_ics_counts.fits mu_iso_counts.fits mu_ps_counts.fits mu_nfw_counts.fits \
  --labels gas ics iso ps nfw \
  --fb-flat-dnde fermi_data/totani/processed/templates/bubbles_flat_binary_dnde.fits \
  --fbpos fermi_data/totani/processed/templates/mu_fbpos_E4.3GeV_k*_counts.fits \
  --fbneg fermi_data/totani/processed/templates/mu_fbneg_E4.3GeV_k*_counts.fits \
  --outdir check_templates/out

Notes:
- --components are *counts templates cubes* (mu_*.fits) in the same grid as counts.
- If you included loopI and fb_flat in your fit, include them in --components and --labels too.
- fbpos/fbneg checks assume you built them with norm=none; if you used roi-sum, identity check
  won't hold and the script will warn (it will still check non-negativity and make plots).
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# You already have these helpers in your repo
from totani_helpers.totani_io import (
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
    pixel_solid_angle_map,
    load_mask_any_shape,   # expects (nE,ny,nx) output keep-mask
)

EPS_DEFAULT = 1.0


def _read_cube(path, expected_shape=None, allow_glob=True):
    if allow_glob and any(ch in path for ch in ["*", "?", "["]):
        hits = sorted(glob.glob(path))
        if not hits:
            raise FileNotFoundError(f"glob matched nothing: {path}")
        path = hits[0]
    with fits.open(path) as h:
        data = h[0].data
        if data is None and len(h) > 1:
            data = h[1].data
        arr = np.array(data, dtype=np.float64)
    if expected_shape is not None and arr.shape != expected_shape:
        raise RuntimeError(f"{path} shape {arr.shape} != expected {expected_shape}")
    return arr, path


def _energy_from_counts_ebounds(counts_path):
    with fits.open(counts_path) as h:
        eb = h["EBOUNDS"].data
    Emin_mev = np.array(eb["E_MIN"], dtype=float) / 1000.0
    Emax_mev = np.array(eb["E_MAX"], dtype=float) / 1000.0
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    dE_mev = (Emax_mev - Emin_mev)
    return Ectr_mev, dE_mev


def _build_fit_mask(counts_hdr, counts_shape, mask_extsrc_path, roi_lon, roi_lat, disk_cut_deg):
    nE, ny, nx = counts_shape
    wcs = WCS(counts_hdr).celestial
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0

    roi2d = (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)
    disk2d = (np.abs(lat) >= disk_cut_deg) if (disk_cut_deg is not None and disk_cut_deg > 0) else np.ones_like(roi2d, bool)

    # mask_extended_sources.fits is used in your pipeline; helper returns keep-mask
    keep3d = load_mask_any_shape(mask_extsrc_path, counts_shape).astype(bool)
    mask_all = keep3d & roi2d[None, :, :] & disk2d[None, :, :]
    return mask_all, wcs


def _per_bin_stats(counts, model, mask_all, eps=EPS_DEFAULT):
    nE = counts.shape[0]
    sum_ratio = np.full(nE, np.nan)
    chi2dof = np.full(nE, np.nan)
    n_pix = np.zeros(nE, dtype=int)

    for k in range(nE):
        m = mask_all[k]
        if not np.any(m):
            continue
        D = counts[k][m]
        M = model[k][m]
        resid = D - M

        sumD = float(np.sum(D))
        sumM = float(np.sum(M))
        sum_ratio[k] = sumM / sumD if sumD > 0 else np.nan

        chi2 = float(np.sum(resid * resid / (D + eps)))
        dof = max(D.size - 1, 1)
        chi2dof[k] = chi2 / dof
        n_pix[k] = D.size

    return sum_ratio, chi2dof, n_pix


def _recon_identity(model, A, mu_list, mask_all, sample_bins):
    """
    Check that recon = sum_i A[k,i]*mu_i[k] matches model[k] on mask.
    Returns list of dict results per k.
    """
    out = []
    for k in sample_bins:
        m = mask_all[k]
        recon = np.zeros_like(model[k], dtype=np.float64)
        for i, mu in enumerate(mu_list):
            recon += A[k, i] * mu[k]
        diff = recon[m] - model[k][m]
        out.append({
            "k": int(k),
            "max_abs": float(np.max(np.abs(diff))) if diff.size else np.nan,
            "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else np.nan,
        })
    return out


def _plot_series(outdir, x, y, xlabel, ylabel, title, fname, logx=False, logy=False):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o", linewidth=1.2)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_coeffs(outdir, Egev, A, labels):
    plt.figure(figsize=(8, 5))
    for i, lab in enumerate(labels):
        plt.plot(Egev, A[:, i], label=str(lab), linewidth=1.2)
    plt.xscale("log")
    # coefficients can be 0; avoid logy if many zeros
    if np.all(A[A > 0] > 0) and np.count_nonzero(A) > 0:
        plt.yscale("log")
    plt.xlabel("E (GeV)")
    plt.ylabel("NNLS coefficient")
    plt.title("Per-bin NNLS coefficients")
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    path = os.path.join(outdir, "coeffs_vs_energy.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _imshow_masked(outdir, wcs, mask2d, arr2d, title, fname):
    a = arr2d.copy()
    a[~mask2d] = np.nan
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection=wcs)
    im = ax.imshow(a, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Galactic longitude")
    ax.set_ylabel("Galactic latitude")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser()
    repo_dir = os.environ.get("REPO_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    default_counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    default_modelsum = os.path.join(repo_dir, "Totani_paper_check", "fig1", "plots_fig1", "mu_modelsum_counts.fits")
    default_mask = os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits")
    default_outdir = os.path.join(repo_dir, "Totani_paper_check", "check_fb")

    default_coeff_npz = os.path.join(repo_dir, "Totani_paper_check", "fig1", "plots_fig1", "fit_coeffs_per_bin.npz")
    default_components = [
        os.path.join(data_dir, "processed", "templates", "mu_gas_counts.fits"),
        os.path.join(data_dir, "processed", "templates", "mu_ics_counts.fits"),
        os.path.join(data_dir, "processed", "templates", "mu_iso_counts.fits"),
        os.path.join(data_dir, "processed", "templates", "mu_ps_counts.fits"),
        os.path.join(
            data_dir,
            "processed",
            "templates",
            "mu_nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno_counts.fits",
        ),
        os.path.join(data_dir, "processed", "templates", "mu_loopI_counts.fits"),
        os.path.join(data_dir, "processed", "templates", "mu_fb_flat_counts.fits"),
    ]
    default_labels = ["gas", "ics", "iso", "ps", "nfw", "loopI", "fb_flat"]
    default_fb_flat_dnde = os.path.join(data_dir, "processed", "templates", "fb_flat_dnde.fits")
    default_fbpos = os.path.join(data_dir, "processed", "templates", "mu_fbpos*_counts.fits")
    default_fbneg = os.path.join(data_dir, "processed", "templates", "mu_fbneg*_counts.fits")

    ap.add_argument("--counts", default=default_counts)
    ap.add_argument("--modelsum", default=default_modelsum)
    ap.add_argument("--mask", default=default_mask, help="mask_extended_sources.fits (or your keep-mask cube)")

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)

    ap.add_argument("--coeff-npz", default=default_coeff_npz, help="fit_coeffs_per_bin.npz")
    ap.add_argument(
        "--components",
        nargs="*",
        default=default_components,
        help="List of mu_*_counts.fits used in the fit, in order",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=default_labels,
        help="Component labels, same order as --components",
    )

    ap.add_argument("--fb-flat-dnde", default=default_fb_flat_dnde, help="Flat bubbles dnde cube (optional)")
    ap.add_argument("--fbpos", default=default_fbpos, help="mu_fbpos*_counts.fits (optional; glob ok)")
    ap.add_argument("--fbneg", default=default_fbneg, help="mu_fbneg*_counts.fits (optional; glob ok)")

    ap.add_argument("--target-gev", type=float, default=4.3)
    ap.add_argument("--outdir", default=default_outdir)
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------
    # Load counts + header
    # ---------------------------
    with fits.open(args.counts) as h:
        counts = np.array(h[0].data, dtype=np.float64)
        hdr = h[0].header
    nE, ny, nx = counts.shape

    model, model_path = _read_cube(args.modelsum, expected_shape=counts.shape, allow_glob=False)

    # energies from EBOUNDS
    Ectr_mev, dE_mev = _energy_from_counts_ebounds(args.counts)
    Egev = (Ectr_mev / 1000.0)

    # ---------------------------
    # Build mask_all exactly like your fitter mask
    # ---------------------------
    mask_all, wcs = _build_fit_mask(hdr, counts.shape, args.mask, args.roi_lon, args.roi_lat, args.disk_cut)

    # ---------------------------
    # A) Data vs Model sanity
    # ---------------------------
    sum_ratio, chi2dof, n_pix = _per_bin_stats(counts, model, mask_all, eps=args.eps)

    p1 = _plot_series(
        args.outdir, Egev, sum_ratio,
        xlabel="E (GeV)", ylabel="sum(model)/sum(data)",
        title="Mask-summed counts ratio per bin",
        fname="sum_ratio_vs_energy.png",
        logx=True, logy=False
    )
    p2 = _plot_series(
        args.outdir, Egev, chi2dof,
        xlabel="E (GeV)", ylabel="chi2/dof  ( (D-M)^2/(D+eps) )",
        title="Weighted residual chi2/dof per bin",
        fname="chi2dof_vs_energy.png",
        logx=True, logy=False
    )

    # Print worst bins
    finite = np.isfinite(chi2dof)
    if np.any(finite):
        worst = np.argsort(chi2dof[finite])[::-1][:5]
        idxs = np.where(finite)[0][worst]
        print("\n[Worst chi2/dof bins]")
        for k in idxs:
            print(f"  k={k:02d}  E={Egev[k]:.3g} GeV  chi2/dof={chi2dof[k]:.3g}  sumM/sumD={sum_ratio[k]:.4f}  npix={n_pix[k]}")

    print("\n[plots]")
    print(" ", p1)
    print(" ", p2)

    # ---------------------------
    # B) Reconstruction identity (if coeffs + components provided)
    # ---------------------------
    if args.coeff_npz is not None and args.components:
        pack = np.load(args.coeff_npz, allow_pickle=True)
        A = np.array(pack["A"], dtype=np.float64)
        labels_npz = list(pack["labels"]) if "labels" in pack else None

        if A.shape[0] != nE:
            raise RuntimeError(f"A has nE={A.shape[0]} but counts has nE={nE}")

        comp_paths = []
        mu_list = []
        for p in args.components:
            mu, p_used = _read_cube(p, expected_shape=counts.shape, allow_glob=True)
            mu_list.append(mu)
            comp_paths.append(p_used)

        if A.shape[1] != len(mu_list):
            raise RuntimeError(f"A has ncomp={A.shape[1]} but you provided {len(mu_list)} components")

        labels = args.labels if args.labels else (labels_npz if labels_npz is not None else [f"c{i}" for i in range(len(mu_list))])

        def comp_sums(k):
            m = mask_all[k]
            Dsum = float(np.sum(counts[k][m]))
            out = []
            for name, mu in zip(labels, mu_list):
                out.append((str(name), float(np.sum(mu[k][m]))))
            return Dsum, out

        for k in [0, 4, 6, 11]:
            if k < 0 or k >= nE:
                continue
            Dsum, outs = comp_sums(k)
            print(f"\n--- k={k:02d}  E={Egev[k]:.3g} GeV  data sum={Dsum:.4g}")
            for name, s in outs:
                print(f"  {name:7s}  template sum (coeff=1): {s:.4g}   ratio to data: {s/Dsum:.4g}")

        # plot coefficients
        pc = _plot_coeffs(args.outdir, Egev, A, labels)
        print(" ", pc)

        # recon identity on a few bins
        sample_bins = sorted(set([0, nE // 2, nE - 1] + [int(np.argmin(np.abs(Egev - args.target_gev))) ]))
        recon_stats = _recon_identity(model, A, mu_list, mask_all, sample_bins)
        print("\n[Reconstruction identity check: max|recon-model| on mask]")
        for r in recon_stats:
            print(f"  k={r['k']:02d}  max|diff|={r['max_abs']:.6g}  rms={r['rms']:.6g}")

        # also write a small report file
        rep = os.path.join(args.outdir, "recon_identity_report.txt")
        with open(rep, "w") as f:
            f.write("components:\n")
            for lab, p in zip(labels, comp_paths):
                f.write(f"  {lab}: {p}\n")
            f.write("\nreconstruction identity:\n")
            for r in recon_stats:
                f.write(f"  k={r['k']:02d}  max|diff|={r['max_abs']:.6g}  rms={r['rms']:.6g}\n")
        print(" ", rep)
    else:
        print("\n[info] Skipping reconstruction identity (need --coeff-npz and --components).")

    # ---------------------------
    # C) Flat FB checks (optional)
    # ---------------------------
    if args.fb_flat_dnde is not None:
        fbflat, fbflat_path = _read_cube(args.fb_flat_dnde, expected_shape=counts.shape, allow_glob=True)
        k0 = int(np.argmin(np.abs(Egev - args.target_gev)))
        m2d = mask_all[k0]

        vals = fbflat[k0][np.isfinite(fbflat[k0])]
        uniq = np.unique(np.round(vals, 6))
        print("\n[Flat FB check]")
        print("  file:", fbflat_path)
        print("  k0:", k0, "E:", Egev[k0], "GeV")
        print("  min/max:", float(np.nanmin(fbflat[k0])), float(np.nanmax(fbflat[k0])))
        print("  unique-ish (rounded):", uniq[:10], ("..." if uniq.size > 10 else ""))
        print("  nonzero pixels in mask:", int(np.sum((fbflat[k0] > 0) & m2d)))

        _imshow_masked(args.outdir, wcs, m2d, fbflat[k0], f"Flat FB dnde @ {Egev[k0]:.3g} GeV", "flat_fb_k0.png")
        print(" ", os.path.join(args.outdir, "flat_fb_k0.png"))
    else:
        print("\n[info] Skipping flat FB checks (no --fb-flat-dnde).")

    # ---------------------------
    # D/E) FB residual templates checks & plots (optional)
    # ---------------------------
    if args.fbpos and args.fbneg:
        fbpos, fbpos_path = _read_cube(args.fbpos, expected_shape=counts.shape, allow_glob=True)
        fbneg, fbneg_path = _read_cube(args.fbneg, expected_shape=counts.shape, allow_glob=True)

        k0 = int(np.argmin(np.abs(Egev - args.target_gev)))
        m2d = mask_all[k0]

        print("\n[FB residual templates check]")
        print("  fbpos:", fbpos_path)
        print("  fbneg:", fbneg_path)
        print("  k0:", k0, "E:", Egev[k0], "GeV")

        # non-negativity
        min_pos = float(np.nanmin(fbpos[k0][m2d])) if np.any(m2d) else np.nan
        min_neg = float(np.nanmin(fbneg[k0][m2d])) if np.any(m2d) else np.nan
        print("  min(fbpos) on mask:", min_pos)
        print("  min(fbneg) on mask:", min_neg)

        # identity check: fbpos - fbneg == residual (only valid if you built with norm=none)
        residual = counts[k0] - model[k0]
        lhs = (fbpos[k0] - fbneg[k0])
        diff = (lhs - residual)
        maxdiff = float(np.nanmax(np.abs(diff[m2d]))) if np.any(m2d) else np.nan
        rmsdiff = float(np.sqrt(np.nanmean((diff[m2d]) ** 2))) if np.any(m2d) else np.nan
        print("  identity check (expects norm=none):")
        print("    max| (fbpos-fbneg) - (data-model) | on mask =", maxdiff)
        print("    rms  ... =", rmsdiff)
        if maxdiff > 1e-3:
            print("  [warn] Identity mismatch is expected if you used --norm roi-sum when building FB templates.")

        # plots at k0
        _imshow_masked(args.outdir, wcs, m2d, counts[k0],   f"Counts (data) @ {Egev[k0]:.3g} GeV", "k0_counts.png")
        _imshow_masked(args.outdir, wcs, m2d, model[k0],    f"Counts (model) @ {Egev[k0]:.3g} GeV", "k0_model.png")
        _imshow_masked(args.outdir, wcs, m2d, residual,     f"Counts (data-model) @ {Egev[k0]:.3g} GeV", "k0_residual.png")
        _imshow_masked(args.outdir, wcs, m2d, fbpos[k0],    f"FB_pos (counts) @ {Egev[k0]:.3g} GeV", "k0_fbpos.png")
        _imshow_masked(args.outdir, wcs, m2d, fbneg[k0],    f"FB_neg (counts) @ {Egev[k0]:.3g} GeV", "k0_fbneg.png")

        print(" ", os.path.join(args.outdir, "k0_counts.png"))
        print(" ", os.path.join(args.outdir, "k0_model.png"))
        print(" ", os.path.join(args.outdir, "k0_residual.png"))
        print(" ", os.path.join(args.outdir, "k0_fbpos.png"))
        print(" ", os.path.join(args.outdir, "k0_fbneg.png"))
    else:
        print("\n[info] Skipping FB residual template checks (need both --fbpos and --fbneg).")

    # ---------------------------
    # Final summary
    # ---------------------------
    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("check_templates summary\n")
        f.write(f"counts: {args.counts}\n")
        f.write(f"modelsum: {model_path}\n")
        f.write(f"mask: {args.mask}\n")
        f.write(f"roi: |l|<={args.roi_lon}, |b|<={args.roi_lat}\n")
        f.write(f"disk_cut: {args.disk_cut}\n")
        f.write("\nPer-bin stats (finite only):\n")
        fin = np.isfinite(sum_ratio) & np.isfinite(chi2dof)
        if np.any(fin):
            f.write(f"  sumM/sumD median: {np.nanmedian(sum_ratio[fin]):.4f}\n")
            f.write(f"  sumM/sumD min/max: {np.nanmin(sum_ratio[fin]):.4f} / {np.nanmax(sum_ratio[fin]):.4f}\n")
            f.write(f"  chi2/dof median: {np.nanmedian(chi2dof[fin]):.4f}\n")
            f.write(f"  chi2/dof min/max: {np.nanmin(chi2dof[fin]):.4f} / {np.nanmax(chi2dof[fin]):.4f}\n")
        else:
            f.write("  (no finite bins)\n")
    print("\n✓ wrote", summary_path)
    print("✓ Done. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
