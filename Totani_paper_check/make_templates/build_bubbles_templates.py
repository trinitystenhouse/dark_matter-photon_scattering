#!/usr/bin/env python3
"""build_bubbles_templates.py

Construct Totani-style Fermi Bubbles *flat* positive/negative morphology from data residuals.

Outputs (in --out-dir):
- fb_flat_pos_mask.fits : 2D mask (1 inside positive bubbles, 0 outside)
- fb_flat_neg_mask.fits : 2D mask (1 inside negative bubbles, 0 outside)

Optional (if --save-counts-templates):
- fb_flat_pos_counts.fits : 3D counts template cube (nE,ny,nx)
- fb_flat_neg_counts.fits : 3D counts template cube (nE,ny,nx)

Also saves diagnostics:
- bubbles_smoothed_iterXX.png
- bubbles_mask_evolution.png

Notes:
- Fit is done in the full ROI including the disk, but morphology extraction excludes |b|<--disk-cut-deg.
- Morphology is derived from residual flux combined over chosen energy bins.
- All energies are MeV internally.
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from astropy.io import fits
from astropy.wcs import WCS


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from totani_helpers.bubbles_templates import (
    build_flat_counts_template,
    iterate_bubbles_masks,
)
from totani_helpers.fit_utils import load_mu_templates_from_fits
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    load_mask_any_shape,
)


def _read_vertices_lonlat(path: str) -> np.ndarray:
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        raise RuntimeError(f"Need >=3 vertices in {path}")
    return np.asarray(pts, dtype=float)


def _polygon_mask_from_lonlat_vertices(wcs, ny: int, nx: int, verts_lonlat_deg: np.ndarray) -> np.ndarray:
    lon = np.mod(np.asarray(verts_lonlat_deg[:, 0], float), 360.0)
    lat = np.asarray(verts_lonlat_deg[:, 1], float)
    x, y = wcs.world_to_pixel_values(lon, lat)
    poly = Path(np.vstack([x, y]).T)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    return poly.contains_points(pts).reshape(ny, nx)


def _parse_energy_bins(arg: str, Ectr_mev: np.ndarray) -> List[int]:
    if str(arg).strip().lower() == "auto":
        targets_gev = [1.5, 4.3]
        Egev = np.asarray(Ectr_mev, float) / 1000.0
        ks = [int(np.argmin(np.abs(Egev - t))) for t in targets_gev]
        ks = sorted(set(ks))
        return ks

    parts = [p.strip() for p in str(arg).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--energy-bins-for-bubbles must be 'auto' or 'Emin,Emax' in GeV")

    Emin_gev = float(parts[0])
    Emax_gev = float(parts[1])
    if Emin_gev <= 0 or Emax_gev <= 0 or Emax_gev < Emin_gev:
        raise ValueError("Invalid energy range")

    Egev = np.asarray(Ectr_mev, float) / 1000.0
    ks = [int(k) for k in np.where((Egev >= Emin_gev) & (Egev <= Emax_gev))[0]]
    if len(ks) == 0:
        # fallback: nearest to mid
        mid = 0.5 * (Emin_gev + Emax_gev)
        ks = [int(np.argmin(np.abs(Egev - mid)))]
    return ks


def _save_mask_fits(out_path: str, mask2d: np.ndarray, hdr_counts) -> None:
    h = hdr_counts.copy()
    h["BUNIT"] = "dimensionless"
    fits.PrimaryHDU(np.asarray(mask2d, np.uint8), header=h).writeto(out_path, overwrite=True)


def _plot_smoothed(out_png: str, smoothed: np.ndarray, pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    img = np.asarray(smoothed, float)
    pos = np.asarray(pos, bool)
    neg = np.asarray(neg, bool)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)

    finite = np.isfinite(img)
    lim = float(np.nanpercentile(np.abs(img[finite]), 99.0)) if finite.any() else 1.0
    lim = max(lim, 1e-30)

    im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.contour(pos.astype(float), levels=[0.5], colors=["gold"], linewidths=1.5)
    ax.contour(neg.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_mask_only(out_png: str, pos: np.ndarray, neg: np.ndarray, title: str) -> None:
    pos = np.asarray(pos, bool)
    neg = np.asarray(neg, bool)
    img = np.zeros_like(pos, dtype=float)
    img[pos] = 1.0
    img[neg] = -1.0

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("mask: +1 pos, -1 neg")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_lonlat_overlay(
    out_png: str,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    smoothed: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    roi2d: np.ndarray,
    title: str,
) -> None:
    lon2d = np.asarray(lon2d, float)
    lat2d = np.asarray(lat2d, float)
    img = np.asarray(smoothed, float).copy()
    pos = np.asarray(pos, bool)
    neg = np.asarray(neg, bool)
    roi2d = np.asarray(roi2d, bool)

    img[~roi2d] = np.nan
    finite = np.isfinite(img)
    lim = float(np.nanpercentile(np.abs(img[finite]), 99.0)) if finite.any() else 1.0
    lim = max(lim, 1e-30)

    fig = plt.figure(figsize=(7.8, 5.2))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(lon2d, lat2d, img, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax.contour(lon2d, lat2d, pos.astype(float), levels=[0.5], colors=["gold"], linewidths=1.5)
    ax.contour(lon2d, lat2d, neg.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.5)
    ax.set_xlabel("l [deg]")
    ax.set_ylabel("b [deg]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    repo_dir = os.environ.get("REPO_PATH", os.path.expanduser("~/Documents/PhD/Year 2/DM_Photon_Scattering"))
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

    ap.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(data_dir, "processed", "templates"))

    ap.add_argument("--srcmask", default=os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits"))

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut-deg", type=float, default=10.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    here = os.path.dirname(__file__)
    ap.add_argument("--verts-north", default=os.path.join(here, "bubble_vertices_north.txt"))
    ap.add_argument("--verts-south", default=os.path.join(here, "bubble_vertices_south.txt"))
    ap.add_argument(
        "--restrict-to-vertices",
        action="store_true",
        help="Restrict morphology iteration to the union of the north/south vertices-defined polygons.",
    )
    ap.add_argument(
        "--no-restrict-to-vertices",
        action="store_true",
        help="Disable vertex polygon restriction.",
    )

    ap.add_argument(
        "--energy-bins-for-bubbles",
        default="auto",
        help="Either 'auto' or 'Emin,Emax' in GeV. Used to combine residual flux into morphology image.",
    )

    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--smooth-sigma-deg", type=float, default=2.0)
    ap.add_argument("--pos-thresh", type=float, default=0.0)
    ap.add_argument("--neg-thresh", type=float, default=0.0)
    ap.add_argument("--morph-open-deg", type=float, default=1.0)
    ap.add_argument("--morph-close-deg", type=float, default=2.0)

    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--stop-frac", type=float, default=0.01)

    ap.add_argument("--out-dir", default=os.path.join(data_dir, "processed", "templates"))
    ap.add_argument("--save-counts-templates", action="store_true")
    ap.add_argument("--out-plots", default=os.path.join(here, "plots"))

    ap.add_argument(
        "--components",
        nargs="+",
        default=[
            "gas",
            "ics",
            "iso",
            "ps",
            "loopA",
            "loopB",
            "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
        ],
        help="Base background components to fit (must exist as mu_{name}_counts.fits).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_plots, exist_ok=True)

    counts, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape

    expo_raw, E_expo_mev = read_exposure(args.expo)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != counts.shape:
        raise SystemExit(f"Exposure shape {expo.shape} != counts shape {counts.shape}")

    wcs = WCS(hdr).celestial
    omega = pixel_solid_angle_map(wcs, ny, nx, float(args.binsz))
    lon2d, lat2d = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon2d) <= float(args.roi_lon)) & (np.abs(lat2d) <= float(args.roi_lat))

    src_keep_3d = load_mask_any_shape(args.srcmask, counts.shape) if args.srcmask else np.ones_like(counts, bool)
    src_keep_2d = np.all(np.asarray(src_keep_3d, bool), axis=0)

    # Load base background templates in counts-space
    labels = list(args.components)
    mu_list, _hdrs = load_mu_templates_from_fits(
        template_dir=str(args.templates_dir),
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )

    templates_counts: Dict[str, np.ndarray] = {lab: np.asarray(mu, float) for lab, mu in zip(labels, mu_list)}

    # parse energy bins for bubbles morphology
    k_bins = _parse_energy_bins(args.energy_bins_for_bubbles, Ectr_mev)
    print(f"[bubbles] using k bins {k_bins} (Ectr_GeV={np.asarray(Ectr_mev)[k_bins]/1000.0})")

    restrict_vertices = bool(args.restrict_to_vertices) and (not bool(args.no_restrict_to_vertices))
    if restrict_vertices:
        if (not os.path.exists(str(args.verts_north))) or (not os.path.exists(str(args.verts_south))):
            raise SystemExit(
                "Vertices file not found. Pass --no-restrict-to-vertices or set --verts-north/--verts-south."
            )
        verts_n = _read_vertices_lonlat(str(args.verts_north))
        verts_s = _read_vertices_lonlat(str(args.verts_south))
        boundary2d = _polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_n) | _polygon_mask_from_lonlat_vertices(
            wcs, ny, nx, verts_s
        )
    else:
        boundary2d = None

    # Default thresholds if not specified: use symmetric percentiles of initial residual flux
    # We do this by taking a quick 1-iter run with pos/neg=0, then setting thresholds to percentiles.
    pos_thr = float(args.pos_thresh)
    neg_thr = float(args.neg_thresh)

    if (pos_thr == 0.0) and (neg_thr == 0.0):
        # Use small non-zero defaults; adaptive thresholding will adjust if empty.
        # neg must be strictly negative, otherwise (smoothed < neg_thr) never selects negatives.
        pos_thr = 0.0
        neg_thr = -1e-30
    elif neg_thr == 0.0:
        neg_thr = -1e-30
    elif neg_thr > 0.0:
        neg_thr = -abs(neg_thr)

    pos_mask, neg_mask, hist = iterate_bubbles_masks(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        templates_counts=templates_counts,
        base_component_order=labels,
        roi2d=roi2d,
        srcmask2d=src_keep_2d,
        boundary2d=boundary2d,
        lat_deg_2d=lat2d,
        disk_cut_deg=float(args.disk_cut_deg),
        k_bins_for_bubbles=k_bins,
        n_iter=int(args.n_iter),
        smooth_sigma_deg=float(args.smooth_sigma_deg),
        binsz_deg=float(args.binsz),
        pos_thresh=float(pos_thr),
        neg_thresh=float(neg_thr),
        morph_open_deg=float(args.morph_open_deg),
        morph_close_deg=float(args.morph_close_deg),
        alpha=float(args.alpha),
        stop_frac=float(args.stop_frac),
        ridge=0.0,
    )

    # Save masks
    pos_out = os.path.join(args.out_dir, "fb_pos_mask.fits")
    neg_out = os.path.join(args.out_dir, "fb_neg_mask.fits")
    _save_mask_fits(pos_out, pos_mask.astype(np.uint8), hdr)
    _save_mask_fits(neg_out, neg_mask.astype(np.uint8), hdr)

    flat_mask = (np.asarray(pos_mask, bool) | np.asarray(neg_mask, bool))
    flat_out = os.path.join(args.out_dir, "fb_flat_mask.fits")
    _save_mask_fits(flat_out, flat_mask.astype(np.uint8), hdr)

    print("[out]", pos_out)
    print("[out]", neg_out)
    print("[out]", flat_out)


    mu_pos = build_flat_counts_template(mask2d=pos_mask, expo=expo, omega=omega, dE_mev=dE_mev)
    mu_neg = build_flat_counts_template(mask2d=neg_mask, expo=expo, omega=omega, dE_mev=dE_mev)
    mu_flat = build_flat_counts_template(mask2d=flat_mask, expo=expo, omega=omega, dE_mev=dE_mev)

    fits.PrimaryHDU(mu_pos.astype(np.float32), header=hdr).writeto(
        os.path.join(args.out_dir, "mu_fb_pos_counts.fits"), overwrite=True
    )
    fits.PrimaryHDU(mu_neg.astype(np.float32), header=hdr).writeto(
        os.path.join(args.out_dir, "mu_fb_neg_counts.fits"), overwrite=True
    )
    fits.PrimaryHDU(mu_flat.astype(np.float32), header=hdr).writeto(
        os.path.join(args.out_dir, "mu_fb_flat_counts.fits"), overwrite=True
    )

    # Diagnostics
    frac_pos = [r.frac_change_pos for r in hist]
    frac_neg = [r.frac_change_neg for r in hist]

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(hist)), frac_pos, marker="o", label="pos")
    ax.plot(np.arange(len(hist)), frac_neg, marker="o", label="neg")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fractional pixel change")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_plots, "bubbles_mask_evolution.png"), dpi=200)
    plt.close(fig)

    for i, r in enumerate(hist):
        _plot_smoothed(
            os.path.join(args.out_plots, f"bubbles_smoothed_iter{i:02d}.png"),
            r.smoothed,
            r.pos_mask,
            r.neg_mask,
            title=(
                f"Bubbles smoothed iter={i}  pos_thr={r.pos_thresh_used:.3e}  neg_thr={r.neg_thresh_used:.3e}"
            ),
        )

    # Final easy-to-interpret plots: masks only + lon/lat overlay
    _plot_mask_only(
        os.path.join(args.out_plots, "bubbles_masks_final.png"),
        pos_mask,
        neg_mask,
        title="Fermi Bubbles masks (final): pos=gold / neg=cyan",
    )
    _plot_mask_only(
        os.path.join(args.out_plots, "bubbles_flat_mask_final.png"),
        flat_mask,
        np.zeros_like(flat_mask, dtype=bool),
        title="Fermi Bubbles flat mask (final)",
    )
    if len(hist) > 0:
        rfin = hist[-1]
        _plot_lonlat_overlay(
            os.path.join(args.out_plots, "bubbles_lonlat_overlay_final.png"),
            lon2d,
            lat2d,
            rfin.smoothed,
            pos_mask,
            neg_mask,
            roi2d=roi2d,
            title="Bubbles (final) overlay on smoothed residual image",
        )
        _plot_lonlat_overlay(
            os.path.join(args.out_plots, "bubbles_lonlat_overlay_flat_final.png"),
            lon2d,
            lat2d,
            rfin.smoothed,
            flat_mask,
            np.zeros_like(flat_mask, dtype=bool),
            roi2d=roi2d,
            title="Bubbles (final) flat mask overlay on smoothed residual image",
        )

    # Print summary
    ov = int(np.sum(pos_mask & neg_mask))
    if ov != 0:
        raise SystemExit(f"pos/neg overlap detected after resolve: {ov} pixels")

    print("[summary] pos pixels:", int(np.sum(pos_mask)))
    print("[summary] neg pixels:", int(np.sum(neg_mask)))
    print("[summary] overlap pixels:", ov)


if __name__ == "__main__":
    raise SystemExit(main())
