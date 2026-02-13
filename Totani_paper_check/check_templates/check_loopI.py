#!/usr/bin/env python3
"""
Sanity-check Loop I template FITS products:
  - loopI_dnde.fits
  - loopI_E2dnde.fits
  - mu_loopI_counts.fits

Checks:
  1) shapes match counts CCUBE (nE, ny, nx)
  2) finite values, non-negative
  3) ROI masking behaviour (outside ROI ~ 0 in dnde)
  4) consistency: E2dnde == dnde * E^2
  5) closure: mu == dnde * expo * omega * dE (if expo provided)
  6) optional comparison to a reference mu template (e.g. Totani baseline): diff stats + correlation

Optional plots (if --plot-dir provided):
  - reconstructed spatial map (lognorm by default)
  - azimuthal profile vs psi from GC
  - latitude profile I(b) averaged over |l|<=plot_lonwin
  - mu mismatch per energy bin
  - if --ref-mu provided: difference map and scatter plot for a chosen k
"""

import argparse
import os
import inspect

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS

from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)


def _finite_stats(x, name):
    x = np.asarray(x)
    fin = np.isfinite(x)
    if fin.any():
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        mean = float(np.nanmean(x))
    else:
        mn = mx = mean = float("nan")
    return {
        "name": name,
        "shape": tuple(x.shape),
        "finite_frac": float(fin.mean()),
        "min": mn,
        "max": mx,
        "mean": mean,
        "neg_frac": float((x < 0).mean()) if x.size else float("nan"),
        "nan_frac": float(np.isnan(x).mean()) if x.size else float("nan"),
    }


def _print_stats(stats):
    print(
        f"[{stats['name']}] shape={stats['shape']} finite={stats['finite_frac']:.6f} "
        f"nan={stats['nan_frac']:.6f} neg={stats['neg_frac']:.6f} "
        f"min={stats['min']:.6e} max={stats['max']:.6e} mean={stats['mean']:.6e}"
    )


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


def _roi_mask_from_wcs(wcs_cel, ny, nx, roi_lon, roi_lat):
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs_cel.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    roi = (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)
    return roi, lon, lat


def _azimuthal_profile(map2d, lon_deg, lat_deg, psi_bins_deg):
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)
    cospsi = np.cos(b) * np.cos(l)
    cospsi = np.clip(cospsi, -1.0, 1.0)
    psi = np.rad2deg(np.arccos(cospsi))

    prof = np.full(len(psi_bins_deg) - 1, np.nan, float)
    for i in range(len(prof)):
        m = (psi >= psi_bins_deg[i]) & (psi < psi_bins_deg[i + 1]) & np.isfinite(map2d)
        if np.any(m):
            prof[i] = np.nanmean(map2d[m])
    return prof


def _corrcoef(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) < 3:
        return np.nan
    aa = a[m] - np.nanmean(a[m])
    bb = b[m] - np.nanmean(b[m])
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom <= 0:
        return np.nan
    return float(np.sum(aa * bb) / denom)

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

def main():
    default_counts = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
    default_expo = os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits")
    default_dnde = os.path.join(DATA_DIR, "processed", "templates", "loopI_dnde.fits")
    default_e2 = os.path.join(DATA_DIR, "processed", "templates", "loopI_E2dnde.fits")
    default_mu = os.path.join(DATA_DIR, "processed", "templates", "mu_loopI_counts.fits")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dnde", default=default_dnde, help="loopI_dnde.fits")
    ap.add_argument("--e2", default=default_e2, help="loopI_E2dnde.fits")
    ap.add_argument("--mu", default=default_mu, help="mu_loopI_counts.fits")
    ap.add_argument("--counts", default=default_counts, help="Counts CCUBE (authoritative WCS + EBOUNDS)")
    ap.add_argument("--expo", default=default_expo, help="Exposure cube (expcube), required for mu closure")
    ap.add_argument("--ref-mu", default=None, help="Optional reference mu template to compare against")

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--eps", type=float, default=1e-6, help="Tolerance for consistency checks (relative)")

    ap.add_argument(
        "--plot-dir",
        default=os.path.join(os.path.dirname(__file__), "plots_check_loopI"),
        help="Directory to save diagnostic plots",
    )
    ap.add_argument("--plot-k", type=int, default=None, help="Energy-bin index for map plots (default: middle bin)")
    ap.add_argument("--plot-lonwin", type=float, default=5.0, help="Longitude half-width (deg) for latitude profile")
    ap.add_argument("--map-norm", choices=["linear", "log"], default="log", help="Map color normalization")
    ap.add_argument("--show", action="store_true", help="Show plots interactively (in addition to saving)")

    args = ap.parse_args()

    for name in ["counts", "dnde", "e2", "mu"]:
        p = getattr(args, name)
        if p is None or not os.path.exists(str(p)):
            raise SystemExit(
                f"Missing required file --{name}. Either set REPO_PATH so defaults resolve, or pass --{name} explicitly."
            )

    if args.expo is not None and (not os.path.exists(str(args.expo))):
        raise SystemExit(f"Exposure file not found: {args.expo}")

    with fits.open(args.counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_ebounds_from_counts_ccube(hc)

    nE = int(Ectr_mev.size)

    with fits.open(args.dnde) as h:
        dnde = np.array(h[0].data, dtype=float)
    with fits.open(args.e2) as h:
        e2 = np.array(h[0].data, dtype=float)
    with fits.open(args.mu) as h:
        mu = np.array(h[0].data, dtype=float)

    print("== Basic stats ==")
    _print_stats(_finite_stats(dnde, "dnde"))
    _print_stats(_finite_stats(e2, "E2dnde"))
    _print_stats(_finite_stats(mu, "mu_counts"))

    assert dnde.shape == (nE, ny, nx), f"dnde shape {dnde.shape} != {(nE, ny, nx)}"
    assert e2.shape == (nE, ny, nx), f"e2 shape {e2.shape} != {(nE, ny, nx)}"
    assert mu.shape == (nE, ny, nx), f"mu shape {mu.shape} != {(nE, ny, nx)}"
    print("✓ Shapes match expected (nE, ny, nx)")

    for name, arr in [("dnde", dnde), ("E2dnde", e2), ("mu", mu)]:
        minv = float(np.nanmin(arr))
        if minv < -1e-12:
            print(f"⚠ {name} has significantly negative values: min={minv:.3e}")
        else:
            print(f"✓ {name} non-negative (within tolerance)")

    roi, lon_deg, lat_deg = _roi_mask_from_wcs(wcs, ny, nx, args.roi_lon, args.roi_lat)
    outside = ~roi
    outside_level = float(np.nanmax(np.abs(dnde[:, outside]))) if np.any(outside) else 0.0
    print(f"[ROI] max |dnde| outside ROI = {outside_level:.3e}")

    pred_e2 = dnde * (Ectr_mev[:, None, None] ** 2)
    denom = np.nanmax(np.abs(e2))
    rel = float(np.nanmax(np.abs(e2 - pred_e2)) / (denom if denom > 0 else 1.0))
    print(f"[CONSIST] max rel |E2dnde - dnde*E^2| = {rel:.3e}")
    if rel > args.eps:
        print("⚠ E2dnde mismatch")
    else:
        print("✓ E2dnde matches dnde * E^2")

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    if args.plot_k is None:
        k0 = int(np.clip(nE // 2, 0, nE - 1))
    else:
        k0 = int(np.clip(args.plot_k, 0, nE - 1))

    # For LoopI builder: loopI_dnde[k] = T / (omega * dE[k]) with T normalised so sum_{ROI} T = 1.
    # Therefore spatial template reconstructs as T = dnde * omega * dE.
    spatial_rec = dnde[k0] * omega * dE_mev[k0]
    spatial_rec[~roi] = 0.0

    spatial_sum = float(np.nansum(spatial_rec))
    print(f"[NORM] reconstructed spatial sum over ROI pixels (k={k0}) = {spatial_sum:.8f} (expected ~1)")

    rel_per_bin = None

    if args.expo is not None:
        with fits.open(args.expo) as he:
            expo_raw = np.array(he[0].data, dtype=np.float64)
            E_expo_mev = read_expcube_energies_mev(he)

        expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)

        print("[DBG] resampler:", resample_exposure_logE_interp.__name__)
        print("[DBG] resampler file:", inspect.getsourcefile(resample_exposure_logE_interp))
        print("[DBG] expo per-bin sum:", np.nansum(expo, axis=(1, 2)))

        pred_mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

        num = np.nansum(np.abs(mu - pred_mu), axis=(1, 2))
        den = np.nansum(np.abs(mu), axis=(1, 2)) + 1e-30
        rel_per_bin = num / den

        print("[CONSIST] rel mismatch per energy bin:")
        for k, r in enumerate(rel_per_bin):
            print(f"  k={k:02d}  Ectr={Ectr_mev[k]:.1f} MeV  rel={r:.4e}")

        denom = np.nanmax(np.abs(mu))
        rel_mu = float(np.nanmax(np.abs(mu - pred_mu)) / (denom if denom > 0 else 1.0))
        print(f"[CONSIST] max rel |mu - dnde*expo*omega*dE| = {rel_mu:.3e}")
        if rel_mu > 1e-5:
            print("⚠ mu mismatch")
        else:
            print("✓ mu matches dnde*expo*omega*dE (within tolerance)")

    mu_ref = None
    if args.ref_mu is not None:
        if not os.path.exists(str(args.ref_mu)):
            print(f"[REF] WARNING: ref-mu not found: {args.ref_mu} (skipping comparison)")
        else:
            with fits.open(args.ref_mu) as h:
                mu_ref = np.array(h[0].data, dtype=float)
            if mu_ref.shape != mu.shape:
                raise RuntimeError(f"ref-mu shape {mu_ref.shape} != mu shape {mu.shape}")

            m = roi[None, :, :]
            corr = _corrcoef(mu[m], mu_ref[m])
            num = float(np.nansum(np.abs(mu[m] - mu_ref[m])))
            den = float(np.nansum(np.abs(mu_ref[m])) + 1e-30)
            l1 = num / den
            maxrel = float(np.nanmax(np.abs(mu - mu_ref)) / (np.nanmax(np.abs(mu_ref)) + 1e-30))
            print(f"[REF] corr(mu, ref_mu) in ROI = {corr:.6f}")
            print(f"[REF] L1 rel |mu-ref|/|ref| in ROI = {l1:.6e}")
            print(f"[REF] max-rel |mu-ref|/max|ref| = {maxrel:.6e}")

    if args.plot_dir is not None:
        os.makedirs(args.plot_dir, exist_ok=True)
        out_base = os.path.splitext(os.path.basename(args.dnde))[0]

        pos = spatial_rec[np.isfinite(spatial_rec) & (spatial_rec > 0)]
        if pos.size:
            vmin = float(np.nanpercentile(pos, 1))
            vmax = float(np.nanpercentile(pos, 99.9))
            if not np.isfinite(vmin) or vmin <= 0:
                vmin = float(np.nanmin(pos))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(pos))
        else:
            vmin, vmax = 1e-30, 1.0

        norm = None
        if args.map_norm == "log":
            norm = LogNorm(vmin=vmin, vmax=vmax)

        fig = plt.figure(figsize=(6.8, 4.8))
        ax = fig.add_subplot(111)
        im = ax.imshow(
            spatial_rec,
            origin="lower",
            cmap="magma",
            norm=norm,
            vmin=(None if norm is not None else 0.0),
        )
        ax.set_title(f"Loop I spatial (reconstructed) k={k0} Ectr={Ectr_mev[k0]/1000.0:.3g} GeV")
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        fig.colorbar(im, ax=ax, label="arb.")
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, f"{out_base}_spatial_k{k0:02d}.png"), dpi=200)
        if args.show:
            plt.show()
        plt.close(fig)

        psi_bins = np.linspace(0.0, min(args.roi_lon, args.roi_lat), 41)
        prof = _azimuthal_profile(spatial_rec, lon_deg, lat_deg, psi_bins)
        psi_ctr = 0.5 * (psi_bins[:-1] + psi_bins[1:])

        fig = plt.figure(figsize=(6.8, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(psi_ctr, prof, lw=2.0)
        ax.set_xlabel(r"$\psi$ from GC  [deg]")
        ax.set_ylabel("mean intensity (arb.)")
        ax.set_title("Azimuthal mean profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, f"{out_base}_profile_psi.png"), dpi=200)
        if args.show:
            plt.show()
        plt.close(fig)

        mlat = roi & (np.abs(lon_deg) <= float(args.plot_lonwin))
        lat_bins = np.linspace(-args.roi_lat, args.roi_lat, 81)
        lat_ctr = 0.5 * (lat_bins[:-1] + lat_bins[1:])
        prof_b = np.full(len(lat_ctr), np.nan, float)
        for i in range(len(lat_ctr)):
            mm = mlat & (lat_deg >= lat_bins[i]) & (lat_deg < lat_bins[i + 1])
            if np.any(mm):
                prof_b[i] = np.nanmean(spatial_rec[mm])

        fig = plt.figure(figsize=(6.8, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(lat_ctr, prof_b, lw=2.0)
        ax.set_xlabel("b  [deg]")
        ax.set_ylabel("mean intensity (arb.)")
        ax.set_title(f"Latitude profile (|l|<={args.plot_lonwin:g} deg)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.plot_dir, f"{out_base}_profile_lat.png"), dpi=200)
        if args.show:
            plt.show()
        plt.close(fig)

        if rel_per_bin is not None:
            fig = plt.figure(figsize=(6.8, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(Ectr_mev / 1000.0, rel_per_bin, lw=2.0)
            ax.set_xscale("log")
            ax.set_xlabel("Ectr  [GeV]")
            ax.set_ylabel("rel L1 mismatch per bin")
            ax.set_title(r"$||\mu-\mu_{pred}||_1 / ||\mu||_1$")
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_mu_mismatch.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)

        if mu_ref is not None:

            diff = mu[k0] - mu_ref[k0]
            diff[~roi] = np.nan

            fig = plt.figure(figsize=(6.8, 4.8))
            ax = fig.add_subplot(111)
            im = ax.imshow(diff, origin="lower", cmap="coolwarm")
            ax.set_title(f"mu - ref_mu (k={k0})")
            ax.set_xlabel("pixel x")
            ax.set_ylabel("pixel y")
            fig.colorbar(im, ax=ax, label="counts")
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_mu_diff_k{k0:02d}.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)

            x = mu_ref[k0][roi].ravel()
            y = mu[k0][roi].ravel()
            good = np.isfinite(x) & np.isfinite(y)
            x = x[good]
            y = y[good]

            fig = plt.figure(figsize=(6.0, 6.0))
            ax = fig.add_subplot(111)
            if x.size:
                ax.plot(x, y, ".", ms=1.0, alpha=0.3)
            ax.set_xlabel("ref mu")
            ax.set_ylabel("mu")
            ax.set_title(f"Scatter (k={k0}) corr={_corrcoef(x, y):.6f}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_mu_scatter_k{k0:02d}.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
