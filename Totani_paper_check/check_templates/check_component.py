#!/usr/bin/env python3
"""check_component.py

Generic sanity-check + plotting utility for a single template component on the CCUBE grid.

Designed for the Totani paper-check workflow where most components are stored as
mu templates in counts space (mu_*_counts.fits).

It will:
- load WCS + EBOUNDS from --counts
- load a template cube (typically mu counts)
- print basic stats + shape checks
- save standard plots (if --plot-dir)
  - spatial map at one energy bin (LogNorm by default)
  - spectrum: sum over ROI pixels vs energy

This is intentionally lightweight: it does not attempt to validate the physical
model, only I/O consistency and visual diagnostics.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS


def read_ebounds_from_counts_ccube(hdul):
    if "EBOUNDS" not in hdul:
        raise RuntimeError("Counts CCUBE missing EBOUNDS extension")
    eb = hdul["EBOUNDS"].data
    Emin_kev = np.array(eb["E_MIN"], dtype=float)
    Emax_kev = np.array(eb["E_MAX"], dtype=float)

    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    return Emin_mev, Emax_mev, Ectr_mev, dE_mev


def roi_mask_from_wcs(wcs_cel, ny, nx, roi_lon, roi_lat):
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs_cel.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    roi = (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)
    return roi, lon, lat


def finite_stats(x):
    x = np.asarray(x)
    fin = np.isfinite(x)
    if fin.any():
        return {
            "shape": tuple(x.shape),
            "finite": float(fin.mean()),
            "nan": float(np.isnan(x).mean()),
            "neg": float((x < 0).mean()),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
            "mean": float(np.nanmean(x)),
        }
    return {
        "shape": tuple(x.shape),
        "finite": 0.0,
        "nan": 1.0,
        "neg": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "mean": float("nan"),
    }


def print_stats(name, x):
    s = finite_stats(x)
    print(
        f"[{name}] shape={s['shape']} finite={s['finite']:.6f} nan={s['nan']:.6f} neg={s['neg']:.6f} "
        f"min={s['min']:.6e} max={s['max']:.6e} mean={s['mean']:.6e}"
    )


def _safe_lognorm_from_positive(img):
    pos = img[np.isfinite(img) & (img > 0)]
    if pos.size == 0:
        return None
    vmin = float(np.nanpercentile(pos, 1))
    vmax = float(np.nanpercentile(pos, 99.9))
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = float(np.nanmin(pos))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(pos))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0:
        return None
    return LogNorm(vmin=max(vmin, 1e-300), vmax=max(vmax, vmin * 1.0001))


def run_component_check(
    *,
    label,
    template_path,
    counts_path,
    roi_lon=60.0,
    roi_lat=60.0,
    plot_dir=None,
    plot_k=None,
    map_norm="log",
    show=False,
):
    if not os.path.exists(template_path):
        raise SystemExit(f"Template file not found: {template_path}")
    if not os.path.exists(counts_path):
        raise SystemExit(f"Counts file not found: {counts_path}")

    with fits.open(counts_path) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        _, _, Ectr_mev, _ = read_ebounds_from_counts_ccube(hc)

    with fits.open(template_path) as ht:
        cube = np.array(ht[0].data, dtype=float)
        bunit = ht[0].header.get("BUNIT", "")

    nE = int(Ectr_mev.size)
    if cube.shape != (nE, ny, nx):
        raise SystemExit(f"Shape mismatch: template {cube.shape} vs expected {(nE, ny, nx)}")

    print(f"== Component: {label} ==")
    print(f"file: {template_path}")
    print(f"BUNIT: '{bunit}'")
    print_stats("cube", cube)

    roi, lon_deg, lat_deg = roi_mask_from_wcs(wcs, ny, nx, roi_lon, roi_lat)

    if plot_k is None:
        k0 = int(np.clip(nE // 2, 0, nE - 1))
    else:
        k0 = int(np.clip(int(plot_k), 0, nE - 1))

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

        # Spatial map
        img = cube[k0].copy()
        img[~roi] = np.nan

        norm = None
        if map_norm == "log":
            norm = _safe_lognorm_from_positive(img)

        fig = plt.figure(figsize=(6.8, 4.8))
        ax = fig.add_subplot(111)
        im = ax.imshow(img, origin="lower", cmap="magma", norm=norm)
        ax.set_title(f"{label}  k={k0}  Ectr={Ectr_mev[k0]/1000.0:.3g} GeV")
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        fig.colorbar(im, ax=ax, label=bunit if bunit else "arb.")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{label}_map_k{k0:02d}.png"), dpi=200)
        if show:
            plt.show()
        plt.close(fig)

        # Spectrum
        y = np.nansum(cube[:, roi], axis=1)
        fig = plt.figure(figsize=(6.8, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(Ectr_mev / 1000.0, y, lw=2.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Ectr [GeV]")
        ax.set_ylabel(f"sum over ROI ({bunit})" if bunit else "sum over ROI")
        ax.set_title(f"{label} spectrum (ROI sum)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{label}_spectrum_roi_sum.png"), dpi=200)
        if show:
            plt.show()
        plt.close(fig)

        # Latitude profile at chosen k
        lat_bins = np.linspace(-roi_lat, roi_lat, 81)
        lat_ctr = 0.5 * (lat_bins[:-1] + lat_bins[1:])
        prof_b = np.full(len(lat_ctr), np.nan, float)
        for i in range(len(lat_ctr)):
            m = roi & (lat_deg >= lat_bins[i]) & (lat_deg < lat_bins[i + 1])
            if np.any(m):
                prof_b[i] = np.nanmean(cube[k0][m])

        fig = plt.figure(figsize=(6.8, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(lat_ctr, prof_b, lw=2.0)
        ax.set_xlabel("b [deg]")
        ax.set_ylabel(f"mean ({bunit})" if bunit else "mean")
        ax.set_title(f"{label} latitude profile (k={k0})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{label}_profile_lat_k{k0:02d}.png"), dpi=200)
        if show:
            plt.show()
        plt.close(fig)

    return 0


def main():
    repo_dir = os.environ.get("REPO_PATH")
    data_dir = os.path.join(repo_dir, "fermi_data", "totani") if repo_dir else None
    default_counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits") if data_dir else None

    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Short label for plots")
    ap.add_argument("--template", required=True, help="Path to template cube (usually mu_*_counts.fits)")
    ap.add_argument("--counts", default=default_counts, help="Counts CCUBE (for WCS+EBOUNDS)")

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    ap.add_argument("--plot-dir", default=os.path.join(os.path.dirname(__file__), "plots_check_components"))
    ap.add_argument("--plot-k", type=int, default=None)
    ap.add_argument("--map-norm", choices=["linear", "log"], default="log")
    ap.add_argument("--show", action="store_true")

    args = ap.parse_args()

    if args.counts is None or not os.path.exists(str(args.counts)):
        raise SystemExit("Missing --counts (set REPO_PATH or pass --counts explicitly)")

    return run_component_check(
        label=str(args.label),
        template_path=str(args.template),
        counts_path=str(args.counts),
        roi_lon=float(args.roi_lon),
        roi_lat=float(args.roi_lat),
        plot_dir=str(args.plot_dir) if args.plot_dir is not None else None,
        plot_k=args.plot_k,
        map_norm=str(args.map_norm),
        show=bool(args.show),
    )


if __name__ == "__main__":
    raise SystemExit(main())
