#!/usr/bin/env python3
"""
MODIFIED: adds a per-energy-bin fit and writes the TOTAL best-fit model counts cube.

What it does:
- For each energy bin k:
    Fit counts[k] with NNLS to a set of COUNTS templates (mu_*[k]) on mask_all[k]
    (ROI + disk cut + point/extended-source mask).
- Writes:
    OUTDIR/mu_modelsum_counts.fits   : (nE,ny,nx) total best-fit expected counts cube
    OUTDIR/fit_coeffs_per_bin.npz    : coefficients A[k, i] and labels
    OUTDIR/fit_coeffs_per_bin.txt    : human-readable table

Notes:
- Uses weighted NNLS (same weighting you already use).
- No extra normalisations of templates; assumes mu_* are already "true counts templates".
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u
from totani_helpers.totani_io import (
    load_mask_any_shape,
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    resample_exposure_logE_interp,
    read_expcube_energies_mev,
    smooth_nan_2d,
)
from totani_helpers.fit_utils import build_fit_mask3d
from totani_helpers.mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, pick_coeff
from scipy.ndimage import gaussian_filter

try:
    from skimage.measure import find_contours
except Exception:
    find_contours = None

PLOT_EXT_SOURCES = True
MARKER_RADIUS_DEG = 1.0
MARKER_LW = 1.0


def read_extended_sources(psc_path):
    with fits.open(psc_path) as f:
        psc = f[1].data
        cols = {c.upper() for c in psc.columns.names}

        if "EXTENDED_SOURCE_NAME" in cols:
            ext_name = psc["EXTENDED_SOURCE_NAME"]
            ext_name = np.array([
                x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                for x in ext_name
            ])
            is_ext = np.array([len(s.strip()) > 0 and s.strip().upper() != "NONE" for s in ext_name])
        elif "EXTENDED" in cols:
            is_ext = np.array(psc["EXTENDED"]).astype(bool)
        else:
            raise RuntimeError("No EXTENDED_SOURCE_NAME or EXTENDED column found in PSC file")

        psc_ext = psc[is_ext]

    return SkyCoord(psc_ext["GLON"] * u.deg, psc_ext["GLAT"] * u.deg, frame="galactic")


def _read_cube(path, expected_shape):
    with fits.open(path) as h:
        d = h[0].data.astype(float)
    if d.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {d.shape}, expected {expected_shape}")
    return d


def main():
    ap = argparse.ArgumentParser(description="Reproduce Totani Fig.1 bubble image maps from MCMC coefficients.")

    repo_dir = os.environ.get("REPO_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

    ap.add_argument(
        "--mcmc-dir",
        default=os.path.join(repo_dir, "Totani_paper_check", "mcmc", "mcmc_results_fig2_3"),
        help="Directory containing mcmc_results_kXX.npz files.",
    )
    ap.add_argument(
        "--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"],
        default="f_ml",
        help="Which MCMC summary coefficient to use per bin.",
    )

    ap.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--ext-mask", default=os.path.join(data_dir, "processed", "templates", "mask_extended_sources.fits"))
    ap.add_argument("--psc", default=os.path.join(data_dir, "templates", "gll_psc_v35.fit"))

    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig1"))

    ap.add_argument("--disk-cut", type=float, default=10.0)
    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--cell-deg", type=float, default=10.0)
    ap.add_argument("--target-mev", type=float, nargs="+", default=[1500.0, 4300.0])

    ap.add_argument("--mu-gas", default=os.path.join(data_dir, "processed", "templates", "mu_gas_counts.fits"))
    ap.add_argument("--mu-ics", default=os.path.join(data_dir, "processed", "templates", "mu_ics_counts.fits"))
    ap.add_argument("--mu-iso", default=os.path.join(data_dir, "processed", "templates", "mu_iso_counts.fits"))
    ap.add_argument("--mu-ps", default=os.path.join(data_dir, "processed", "templates", "mu_ps_counts.fits"))
    ap.add_argument(
        "--mu-nfw",
        default=os.path.join(data_dir, "processed", "templates", "mu_nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno_counts.fits"),
    )
    ap.add_argument("--mu-loopI", default=os.path.join(data_dir, "processed", "templates", "mu_loopI_counts.fits"))
    ap.add_argument("--mu-fb-flat", default=os.path.join(data_dir, "processed", "templates", "mu_fb_flat_counts.fits"))
    ap.add_argument("--fb-flat-dnde", default=os.path.join(data_dir, "processed", "templates", "fb_flat_dnde.fits"))
    ap.add_argument("--bubble-mask", default=os.path.join(data_dir, "processed", "templates", "bubbles_flat_binary_mask.fits"))

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    counts, hdr, _Emin, _Emax, Ectr, dE = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    expo_raw, E_expo = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo, Ectr)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")

    ps_mask = load_mask_any_shape(args.ext_mask, counts.shape)
    omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=float(args.binsz))
    lon, lat = lonlat_grids(wcs, ny, nx)

    disk_mask2d = np.abs(lat) >= float(args.disk_cut)
    roi2d = (np.abs(lon) <= float(args.roi_lon)) & (np.abs(lat) <= float(args.roi_lat))
    mask_all = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=ps_mask,
        counts=counts,
        expo=expo,
        extra2d=disk_mask2d,
    )

    mu_gas = _read_cube(args.mu_gas, counts.shape)
    mu_ics = _read_cube(args.mu_ics, counts.shape)
    mu_iso = _read_cube(args.mu_iso, counts.shape)
    mu_ps = _read_cube(args.mu_ps, counts.shape)
    mu_nfw = _read_cube(args.mu_nfw, counts.shape)
    mu_loopi = _read_cube(args.mu_loopI, counts.shape)

    if os.path.exists(str(args.mu_fb_flat)):
        mu_flat = _read_cube(str(args.mu_fb_flat), counts.shape)
    else:
        bubbles_flat_dnde = _read_cube(str(args.fb_flat_dnde), counts.shape)
        mu_flat = (bubbles_flat_dnde > 0).astype(np.float64)

    labels = ["gas", "ics", "iso", "ps", "nfw", "loopI", "fb_flat"]
    templates_counts = {
        "gas": mu_gas,
        "ics": mu_ics,
        "iso": mu_iso,
        "ps": mu_ps,
        "nfw": mu_nfw,
        "loopI": mu_loopi,
        "fb_flat": mu_flat,
    }

    tab = load_mcmc_coeffs_by_label(mcmc_dir=str(args.mcmc_dir), stat=str(args.mcmc_stat), nE=nE)
    coeffs_plot = combine_loopI(coeffs_by_label=dict(tab.coeffs_by_label), out_key="loopI", drop_inputs=False)

    comp_counts = {}
    model_counts = np.zeros_like(counts, dtype=float)
    for lab in labels:
        mu = np.asarray(templates_counts[lab], float)
        a = pick_coeff(coeffs_by_label=coeffs_plot, template_key=str(lab))
        pred = np.asarray(a, float).reshape(-1)[:, None, None] * mu
        comp_counts[lab] = pred
        model_counts += pred

    fb_mask2d = None
    if os.path.exists(str(args.bubble_mask)):
        try:
            fb_mask2d = fits.getdata(str(args.bubble_mask)).astype(bool)
            if fb_mask2d.ndim == 3:
                fb_mask2d = fb_mask2d[0].astype(bool)
            if fb_mask2d.shape != (ny, nx):
                fb_mask2d = None
        except Exception:
            fb_mask2d = None

    ext_coords = None
    if PLOT_EXT_SOURCES and os.path.exists(str(args.psc)):
        try:
            ext_coords = read_extended_sources(str(args.psc))
        except Exception:
            ext_coords = None

    for target_mev in args.target_mev:
        k = int(np.argmin(np.abs(Ectr - float(target_mev))))
        m = mask_all[k]

        bubble_image_counts_k = np.array(comp_counts["fb_flat"][k], dtype=float) + (
            np.array(counts[k], dtype=float) - np.array(model_counts[k], dtype=float)
        )

        denom = np.array(expo[k], dtype=float) * np.array(omega, dtype=float) * float(dE[k])
        bubble_flux_k = np.full((ny, nx), np.nan, dtype=float)
        ok = m & np.isfinite(denom) & (denom > 0)
        bubble_flux_k[ok] = bubble_image_counts_k[ok] / denom[ok]

        sig_pix = float(1.0 / float(args.binsz))
        bubble_flux_smooth_k = smooth_nan_2d(bubble_flux_k, sig_pix)
        bubble_flux_smooth_k[~m] = np.nan

        fig = plt.figure(figsize=(6.0, 4.8))
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
        ax.set_xlabel("l")
        ax.set_ylabel("b")

        if float(target_mev) == float(args.target_mev[0]):
            vmin, vmax = -5e-10, 5e-10
        elif len(args.target_mev) > 1 and float(target_mev) == float(args.target_mev[1]):
            vmin, vmax = -5e-11, 5e-11
        else:
            vmax = np.nanpercentile(np.abs(bubble_flux_smooth_k), 99.5)
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            vmin = -vmax

        im = ax.imshow(bubble_flux_smooth_k, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"Best-fit bubble image (flux), smoothed $\\sigma$=1$^\\circ$\nE={Ectr[k]/1000:.3g} GeV (k={k})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if fb_mask2d is not None:
            try:
                ax.contour(
                    fb_mask2d.astype(float),
                    levels=[0.5],
                    colors="k",
                    linewidths=1.0,
                    alpha=0.8,
                )
            except Exception:
                ax.imshow(
                    np.where(fb_mask2d, 1.0, np.nan),
                    origin="lower",
                    cmap="gray",
                    alpha=0.12,
                )

        out_png = os.path.join(args.outdir, f"fig1_bubble_image_flux_smooth1deg_E{float(target_mev):.0f}MeV_k{k}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        print("✓ wrote", out_png)

    print("✓ Done:", args.outdir)


if __name__ == "__main__":
    main()
