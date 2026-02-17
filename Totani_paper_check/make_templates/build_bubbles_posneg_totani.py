#!/usr/bin/env python3
"""
Totani-style structured Fermi Bubbles templates from a single-energy residual map (~4.3 GeV).

Method (Totani):
  1) Do an initial fit including a *flat* FB template.
  2) At E ≈ 4.3 GeV (where FB are clear), form a residual map R(l,b).
  3) Define two spatial templates:
       FB_pos(l,b) = max(R, 0)   (bubble-like positive regions)
       FB_neg(l,b) = max(-R, 0)  (negative regions absorbing model mismatch)
     Use these as energy-independent spatial templates in subsequent fits.
  4) Important: FB_pos is based on positive regions regardless of the flat-template boundary.

This script does steps 2–3 (builds FB_pos and FB_neg) and writes:
  - mu_fbpos*_counts.fits
  - mu_fbneg*_counts.fits
  - fbpos*_dnde.fits, fbpos*_E2dnde.fits
  - fbneg*_dnde.fits, fbneg*_E2dnde.fits

NO unnecessary normalisations:
  - by default, we DO NOT renormalize FB_pos/FB_neg (their absolute scale is whatever the residual gives).
  - optional --norm roi-sum if you *want* to match "sum over ROI pixels = 1" convention.

Residual sources:
  - Either provide --model-mu (3D expected counts cube) and we compute R = counts[k0]-model[k0]
  - Or provide --residual-map (2D FITS) on the CCUBE grid in *counts* units.
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)


def pick_energy_index(Ectr_mev, target_gev):
    target_mev = 1000.0 * float(target_gev)
    return int(np.argmin(np.abs(Ectr_mev - target_mev)))


def read_2d_fits(path):
    with fits.open(path) as h:
        data = h[0].data if h[0].data is not None else h[1].data
        arr = np.array(data, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"--residual-map must be 2D, got shape {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()

    repo_dir = os.environ.get("REPO_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    default_counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    default_expo = os.path.join(data_dir, "processed", "expcube_1000to1000000.fits")
    default_outdir = os.path.join(data_dir, "processed", "templates")
    default_model_mu = os.path.join(repo_dir, "Totani_paper_check", "fig1", "plots_fig1", "mu_modelsum_counts.fits")

    ap.add_argument("--counts", default=default_counts, help="Counts CCUBE (authoritative WCS + EBOUNDS)")
    ap.add_argument("--expo", default=default_expo, help="Exposure cube (expcube)")
    ap.add_argument("--outdir", default=default_outdir)

    # Provide exactly one of these:
    ap.add_argument(
        "--model-mu",
        default=default_model_mu,
        help="Best-fit model expected counts cube (3D, same shape as counts)",
    )
    ap.add_argument("--residual-map", default=None, help="2D residual map at target energy on CCUBE grid (counts units)")

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    ap.add_argument("--target-gev", type=float, default=4.3, help="Energy (GeV) to build residual templates from")

    # Totani’s halo-search stage excludes disk, but the *construction* is just “positive/negative regions at 4.3 GeV”.
    # You can optionally zero the disk here if you want the templates to ignore it.
    ap.add_argument("--mask-disk", type=float, default=0.0, help="If >0, zero |b| < mask_disk degrees")

    ap.add_argument(
        "--norm",
        choices=["none", "roi-sum"],
        default="none",
        help="Template normalization. Default 'none' = no rescaling (Totani-style, avoid unnecessary norms).",
    )

    ap.add_argument(
        "--expo-sampling",
        choices=["center", "edge-mean"],
        default="center",
        help="Exposure per CCUBE bin: at bin center, or mean of exposure at (Emin,Emax).",
    )

    args = ap.parse_args()

    if (args.model_mu is None) == (args.residual_map is None):
        raise RuntimeError("Provide exactly one of: --model-mu OR --residual-map")

    os.makedirs(args.outdir, exist_ok=True)

    # --- Read CCUBE header + EBOUNDS ---
    with fits.open(args.counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        counts_cube = np.array(hc[0].data, dtype=np.float64)
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]

        if "EBOUNDS" not in hc:
            raise RuntimeError("Counts CCUBE missing EBOUNDS extension.")
        eb = hc["EBOUNDS"].data
        eb_hdu = hc["EBOUNDS"].copy()

    if counts_cube.ndim != 3:
        raise RuntimeError(f"Counts cube must be 3D (nE,ny,nx), got {counts_cube.shape}")

    Emin_mev = np.array(eb["E_MIN"], dtype=float) / 1000.0
    Emax_mev = np.array(eb["E_MAX"], dtype=float) / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    nE = int(Ectr_mev.size)

    if counts_cube.shape[0] != nE:
        raise RuntimeError(f"Counts cube nE={counts_cube.shape[0]} but EBOUNDS has nE={nE}")

    # --- Read exposure cube + resample to CCUBE E bins ---
    with fits.open(args.expo) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(he)

    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure spatial shape {expo_raw.shape[1:]} != counts {(ny, nx)}")

    if args.expo_sampling == "center":
        expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)
        expo_comment = "Exposure at CCUBE bin centers (logE interp, clamped)."
    else:
        expo_min = resample_exposure_logE_interp(expo_raw, E_expo_mev, Emin_mev)
        expo_max = resample_exposure_logE_interp(expo_raw, E_expo_mev, Emax_mev)
        expo = 0.5 * (expo_min + expo_max)
        expo_comment = "Exposure = 0.5*(expo(Emin)+expo(Emax)) with logE interp (clamped)."

    # --- lon/lat grid + ROI/disk mask (only zeroing; no renorm) ---
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0

    roi = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    if args.mask_disk and args.mask_disk > 0:
        roi = roi & (np.abs(lat) >= args.mask_disk)

    # --- pick target energy bin (~4.3 GeV) ---
    k0 = pick_energy_index(Ectr_mev, args.target_gev)
    E0_gev = Ectr_mev[k0] / 1000.0
    print(f"[info] Using k0={k0} with Ectr={E0_gev:.3f} GeV (target {args.target_gev:.3f} GeV)")

    # --- build residual map at k0 ---
    if args.residual_map is not None:
        R = read_2d_fits(args.residual_map)
        if R.shape != (ny, nx):
            raise RuntimeError(f"Residual map shape {R.shape} != CCUBE spatial {(ny, nx)}")
        residual_comment = f"Residual map read from {os.path.basename(args.residual_map)}"
    else:
        with fits.open(args.model_mu) as hm:
            model_cube = np.array(hm[0].data, dtype=np.float64)
        if model_cube.shape != counts_cube.shape:
            raise RuntimeError(f"Model cube shape {model_cube.shape} != counts cube {counts_cube.shape}")
        R = counts_cube[k0] - model_cube[k0]
        residual_comment = f"Residual map computed as counts[k0]-model_mu[k0] from {os.path.basename(args.model_mu)}"

    # apply ROI mask (zero outside)
    Rm = np.zeros_like(R)
    Rm[roi] = R[roi]

    # Totani split by sign
    fb_pos_2d = np.clip(Rm, 0.0, np.inf)
    fb_neg_2d = np.clip(-Rm, 0.0, np.inf)

    # optional normalization (OFF by default)
    if args.norm == "roi-sum":
        s_pos = float(np.nansum(fb_pos_2d[roi]))
        s_neg = float(np.nansum(fb_neg_2d[roi]))
        if not np.isfinite(s_pos) or s_pos <= 0:
            raise RuntimeError("FB_pos is zero in ROI; cannot roi-sum normalize.")
        if not np.isfinite(s_neg) or s_neg <= 0:
            raise RuntimeError("FB_neg is zero in ROI; cannot roi-sum normalize.")
        fb_pos_2d /= s_pos
        fb_neg_2d /= s_neg

    # --- build energy-independent counts templates by repeating the 2D maps across energy bins ---
    mu_pos = np.broadcast_to(fb_pos_2d[None, :, :], (nE, ny, nx)).astype(np.float64).copy()
    mu_neg = np.broadcast_to(fb_neg_2d[None, :, :], (nE, ny, nx)).astype(np.float64).copy()

    # --- convert to dnde / E2dnde using your pipeline convention ---
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    denom = expo * omega[None, :, :] * dE_mev[:, None, None]

    def mu_to_dnde(mu):
        dnde = np.full_like(mu, np.nan, dtype=np.float64)
        ok = np.isfinite(denom) & (denom > 0)
        dnde[ok] = mu[ok] / denom[ok]
        e2dnde = dnde * (Ectr_mev[:, None, None] ** 2)
        return dnde, e2dnde

    dnde_pos, e2_pos = mu_to_dnde(mu_pos)
    dnde_neg, e2_neg = mu_to_dnde(mu_neg)

    # --- write outputs ---
    def _write_cube(path, data, bunit, comments):
        phdu = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
        phdu.header["BUNIT"] = bunit
        for c in comments:
            phdu.header["COMMENT"] = c
        fits.HDUList([phdu, eb_hdu]).writeto(path, overwrite=True)

    tag = f"_E{args.target_gev:g}GeV_k{k0}_roiL{args.roi_lon:g}_roiB{args.roi_lat:g}"
    if args.mask_disk and args.mask_disk > 0:
        tag += f"_maskDisk{args.mask_disk:g}"
    tag += f"_norm{args.norm}"
    tag += f"_expo{args.expo_sampling}"

    out_mu_pos   = os.path.join(args.outdir, f"mu_fbpos{tag}_counts.fits")
    out_mu_neg   = os.path.join(args.outdir, f"mu_fbneg{tag}_counts.fits")
    out_dnde_pos = os.path.join(args.outdir, f"fbpos{tag}_dnde.fits")
    out_dnde_neg = os.path.join(args.outdir, f"fbneg{tag}_dnde.fits")
    out_e2_pos   = os.path.join(args.outdir, f"fbpos{tag}_E2dnde.fits")
    out_e2_neg   = os.path.join(args.outdir, f"fbneg{tag}_E2dnde.fits")

    comments = [
        "Totani-style FB templates from single-bin residual split by sign:",
        "FB_pos = max(R,0), FB_neg = max(-R,0), where R is residual at ~4.3 GeV.",
        f"Using k0={k0}, Ectr={E0_gev:.3f} GeV (closest to target {args.target_gev:.3f} GeV).",
        residual_comment,
        f"ROI: |l|<={args.roi_lon:g}, |b|<={args.roi_lat:g}" + (f", with |b|>={args.mask_disk:g} masked" if args.mask_disk and args.mask_disk > 0 else ""),
        f"Normalization: {args.norm} (default none = no rescaling).",
        "Energy-independent spatial map repeated across all energy bins (as in Totani).",
        expo_comment,
        "mu = dnde * exposure * Omega_pix * dE (for the derived dnde products).",
    ]

    _write_cube(out_mu_pos, mu_pos, "counts", ["FB_pos expected counts-template (shape-only)."] + comments)
    _write_cube(out_mu_neg, mu_neg, "counts", ["FB_neg expected counts-template (shape-only)."] + comments)
    _write_cube(out_dnde_pos, dnde_pos, "ph cm-2 s-1 sr-1 MeV-1", comments)
    _write_cube(out_dnde_neg, dnde_neg, "ph cm-2 s-1 sr-1 MeV-1", comments)
    _write_cube(out_e2_pos, e2_pos, "MeV cm-2 s-1 sr-1", ["E^2 dN/dE derived using Ectr^2."] + comments)
    _write_cube(out_e2_neg, e2_neg, "MeV cm-2 s-1 sr-1", ["E^2 dN/dE derived using Ectr^2."] + comments)

    print("✓ wrote", out_mu_pos)
    print("✓ wrote", out_mu_neg)
    print("✓ wrote", out_dnde_pos)
    print("✓ wrote", out_dnde_neg)
    print("✓ wrote", out_e2_pos)
    print("✓ wrote", out_e2_neg)

    # quick sanity prints at the construction energy
    print("[sanity] sum FB_pos_2d (ROI):", float(np.nansum(fb_pos_2d[roi])))
    print("[sanity] sum FB_neg_2d (ROI):", float(np.nansum(fb_neg_2d[roi])))


if __name__ == "__main__":
    main()
