#!/usr/bin/env python3
"""build_fermi_bubbles_templates.py

Construct Fermi bubble templates in the style described in Totani (2025), on the
same grid/units as the other template builders in this repo.

Outputs written to OUTDIR (defaults to fermi_data/processed/templates):
  - bubbles_pos_dnde.fits          [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - bubbles_pos_E2dnde.fits        [MeV cm^-2 s^-1 sr^-1]
  - mu_bubbles_pos_counts.fits     [counts]
  - bubbles_neg_dnde.fits          [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - bubbles_neg_E2dnde.fits        [MeV cm^-2 s^-1 sr^-1]
  - mu_bubbles_neg_counts.fits     [counts]

Method (high-level):
- Choose the analysis energy bin closest to 4.3 GeV.
- Fit that bin (including disk) with existing components + a rough flat bubble mask.
- Build a bubble image as (best-fit flat bubble component + residuals), convert to
  intensity using exposure, smooth with a 1 deg Gaussian, then split into positive
  and negative templates.

Notes:
- The initial flat bubble boundary is necessarily approximate; you can supply an
  explicit mask via --bubble-mask to override the heuristic.
"""

import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    """Ω_pix ≈ Δl Δb cos(b) for CAR (matches your other template builders)."""
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), dtype=float)


def smooth_nan_2d(x2d, sigma_pix):
    m = np.isfinite(x2d)
    if not np.any(m):
        return np.full_like(x2d, np.nan, dtype=float)
    w = m.astype(float)
    xs = gaussian_filter(np.where(m, x2d, 0.0), sigma_pix)
    ws = gaussian_filter(w, sigma_pix)
    y = np.full_like(x2d, np.nan, dtype=float)
    ok = ws > 0
    y[ok] = xs[ok] / ws[ok]
    return y


def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    """Interpolate exposure planes onto target energy centers in log(E)."""
    if expo_raw.shape[0] == len(E_tgt_mev):
        return expo_raw
    if E_expo_mev is None:
        raise RuntimeError("Exposure planes != counts planes and EXPO has no ENERGIES table.")

    order = np.argsort(E_expo_mev)
    E_expo_mev = E_expo_mev[order]
    expo_raw = expo_raw[order]

    logEs = np.log(E_expo_mev)
    logEt = np.log(E_tgt_mev)

    ne, ny, nx = expo_raw.shape
    flat = expo_raw.reshape(ne, ny * nx)

    idx = np.searchsorted(logEs, logEt)
    idx = np.clip(idx, 1, ne - 1)
    i0 = idx - 1
    i1 = idx
    w = (logEt - logEs[i0]) / (logEs[i1] - logEs[i0])

    out = np.empty((len(E_tgt_mev), ny * nx), float)
    for j in range(len(E_tgt_mev)):
        out[j] = (1 - w[j]) * flat[i0[j]] + w[j] * flat[i1[j]]
    return out.reshape(len(E_tgt_mev), ny, nx)


def write_primary_with_bunit(path, data, hdr, bunit, comments):
    hdu = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
    hdu.header["BUNIT"] = bunit
    for c in comments:
        c_ascii = c.encode("ascii", "replace").decode("ascii")
        hdu.header.add_comment(c_ascii)
    hdu.writeto(path, overwrite=True)


def _heuristic_flat_bubble_mask(lon_deg, lat_deg, lon_max=20.0, lat_min=10.0, lat_max=55.0):
    """Simple rectangular-ish boundary for a rough initial flat bubble area."""
    return (
        (np.abs(lon_deg) <= lon_max)
        & (np.abs(lat_deg) >= lat_min)
        & (np.abs(lat_deg) <= lat_max)
    )


def _load_mask_fits(path, hdr_like):
    with fits.open(path) as h:
        data = h[0].data
        hdr = h[0].header

    if data.ndim == 3:
        data = data[0]

    if data.shape != (hdr_like["NAXIS2"], hdr_like["NAXIS1"]):
        raise RuntimeError(
            f"bubble mask shape {data.shape} != counts grid {(hdr_like['NAXIS2'], hdr_like['NAXIS1'])}"
        )

    return np.isfinite(data) & (data != 0)


def fit_bin_weighted_nnls(counts_2d, mu_components, mask_2d, eps=1.0):
    """Weighted NNLS on a single energy bin.

    counts_2d: (ny,nx)
    mu_components: list of (ny,nx) expected counts maps
    mask_2d: boolean

    Returns amplitudes A (len = ncomp)
    """
    m = mask_2d.ravel()
    y = counts_2d.ravel()[m]

    X = np.vstack([mu.ravel()[m] for mu in mu_components]).T

    # Approx Poisson weighting; stabilize with eps to avoid div by zero
    w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
    yw = y * w
    Xw = X * w[:, None]

    A, _ = nnls(Xw, yw)
    return A


def bubble_image_counts_cellwise(
    counts_2d,
    *,
    mu_iem_2d,
    mu_iso_2d,
    mu_ps_2d,
    mu_nfw_2d,
    mu_flat_2d,
    fit_mask2d,
    lon_wrapped_deg,
    lat_deg,
    roi_lon_deg,
    roi_lat_deg,
    cell_deg,
):
    bubble_counts = np.full_like(counts_2d, np.nan, dtype=float)
    l_edges = np.arange(-roi_lon_deg, roi_lon_deg + 1e-6, cell_deg)
    b_edges = np.arange(-roi_lat_deg, roi_lat_deg + 1e-6, cell_deg)

    for l0 in l_edges[:-1]:
        l1 = l0 + cell_deg
        in_l = (lon_wrapped_deg >= l0) & (lon_wrapped_deg < l1)
        for b0 in b_edges[:-1]:
            b1 = b0 + cell_deg
            cell_mask = fit_mask2d & in_l & (lat_deg >= b0) & (lat_deg < b1)
            if not np.any(cell_mask):
                continue

            mu_components = [mu_iem_2d, mu_iso_2d, mu_ps_2d, mu_nfw_2d, mu_flat_2d]
            A = fit_bin_weighted_nnls(counts_2d, mu_components, cell_mask)

            model_all = (
                A[0] * mu_iem_2d[cell_mask]
                + A[1] * mu_iso_2d[cell_mask]
                + A[2] * mu_ps_2d[cell_mask]
                + A[3] * mu_nfw_2d[cell_mask]
                + A[4] * mu_flat_2d[cell_mask]
            )
            residual = counts_2d[cell_mask] - model_all
            bubble_counts[cell_mask] = (A[4] * mu_flat_2d[cell_mask]) + residual

    return bubble_counts


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--expcube",
        default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--mu-iem",
        default=os.path.join(DATA_DIR, "processed", "templates", "mu_iem_counts.fits"),
    )
    ap.add_argument(
        "--mu-iso",
        default=os.path.join(DATA_DIR, "processed", "templates", "mu_iso_counts.fits"),
    )
    ap.add_argument(
        "--mu-ps",
        default=os.path.join(DATA_DIR, "processed", "templates", "mu_ps_counts.fits"),
    )
    ap.add_argument(
        "--mu-nfw",
        default=os.path.join(DATA_DIR, "processed", "templates", "mu_nfw_counts.fits"),
    )

    ap.add_argument(
        "--outdir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    ap.add_argument(
        "--cell-deg",
        type=float,
        default=10.0,
        help="Cell size for Totani-style cell-by-cell fit (deg)",
    )

    ap.add_argument("--ref-energy-gev", type=float, default=4.3)
    ap.add_argument("--smooth-sigma-deg", type=float, default=1.0)

    ap.add_argument("--bubble-mask", default=None, help="Optional FITS mask for flat bubble boundary on counts grid")
    ap.add_argument("--flat-lon-max", type=float, default=20.0)
    ap.add_argument("--flat-lat-min", type=float, default=10.0)
    ap.add_argument("--flat-lat-max", type=float, default=55.0)

    ap.add_argument(
        "--norm-disk-cut",
        type=float,
        default=10.0,
        help="Normalize positive/negative templates over |b|>=this cut inside ROI",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load counts grid + energies ---
    with fits.open(args.counts) as h:
        hdr = h[0].header
        counts = h[0].data.astype(float)
        eb = h["EBOUNDS"].data

    wcs = WCS(hdr).celestial
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])

    Emin_kev = eb["E_MIN"].astype(float)
    Emax_kev = eb["E_MAX"].astype(float)
    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    Ectr_gev = Ectr_mev / 1000.0

    nE = len(Ectr_mev)

    # --- exposure (cm^2 s), resample if needed ---
    with fits.open(args.expcube) as h:
        expo_raw = h[0].data.astype(float)
        E_expo_mev = None
        if "ENERGIES" in h:
            col0 = h["ENERGIES"].columns.names[0]
            E_expo_mev = np.array(h["ENERGIES"].data[col0], dtype=float)

    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != (nE, ny, nx):
        raise RuntimeError(f"Exposure shape {expo.shape} not compatible with {(nE, ny, nx)}")

    # --- lon/lat, omega ---
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)

    # Wrap lon to [-180,180) for consistent ROI/cell definitions
    lon_w = ((lon + 180.0) % 360.0) - 180.0

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # --- ROI masks ---
    roi2d = (np.abs(lon_w) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)

    # --- load base component mu templates ---
    def _read_mu(path):
        with fits.open(path) as h:
            d = h[0].data.astype(float)
        if d.shape != (nE, ny, nx):
            raise RuntimeError(f"{path} has shape {d.shape}, expected {(nE, ny, nx)}")
        return d

    mu_iem = _read_mu(args.mu_iem)
    mu_iso = _read_mu(args.mu_iso)
    mu_ps = _read_mu(args.mu_ps)
    mu_nfw = _read_mu(args.mu_nfw)

    # --- choose reference bin closest to 4.3 GeV ---
    k_ref = int(np.argmin(np.abs(Ectr_gev - args.ref_energy_gev)))

    # --- build flat bubble spatial mask ---
    if args.bubble_mask is not None:
        flat_mask = _load_mask_fits(args.bubble_mask, hdr_like=hdr)
    else:
        flat_mask = _heuristic_flat_bubble_mask(
            lon_w,
            lat,
            lon_max=args.flat_lon_max,
            lat_min=args.flat_lat_min,
            lat_max=args.flat_lat_max,
        )

    flat_mask &= roi2d

    # Flat FB template (isotropic within the flat boundary)
    T_flat_out = np.zeros((ny, nx), float)
    T_flat_out[flat_mask] = 1.0

    # Normalise flat template over |b|>=norm_disk_cut inside ROI (Totani convention)
    # and build mu_flat for the reference-bin fit using the same normalized template
    # that will be written out for Fig. 2/3.
    norm_mask = roi2d & (np.abs(lat) >= args.norm_disk_cut)
    T_flat_out[~roi2d] = 0.0
    s_flat = float(np.nansum(T_flat_out[norm_mask]))
    if not np.isfinite(s_flat) or s_flat <= 0:
        raise RuntimeError("Flat bubble template is zero after ROI/disk-cut processing")
    T_flat_out /= s_flat
    mu_flat_ref = T_flat_out * expo[k_ref]

    # --- fit the reference energy bin including disk ---
    fit_mask2d = roi2d

    bubble_plus_residual_counts = bubble_image_counts_cellwise(
        counts[k_ref],
        mu_iem_2d=mu_iem[k_ref],
        mu_iso_2d=mu_iso[k_ref],
        mu_ps_2d=mu_ps[k_ref],
        mu_nfw_2d=mu_nfw[k_ref],
        mu_flat_2d=mu_flat_ref,
        fit_mask2d=fit_mask2d,
        lon_wrapped_deg=lon_w,
        lat_deg=lat,
        roi_lon_deg=args.roi_lon,
        roi_lat_deg=args.roi_lat,
        cell_deg=args.cell_deg,
    )

    # Convert to intensity-like map at reference energy
    denom = expo[k_ref] * omega * dE_mev[k_ref]
    bubble_plus_residual_dnde = np.full_like(bubble_plus_residual_counts, np.nan, dtype=float)
    good = (denom > 0) & np.isfinite(denom) & np.isfinite(bubble_plus_residual_counts)
    bubble_plus_residual_dnde[good] = bubble_plus_residual_counts[good] / denom[good]

    # Smooth with sigma=1 deg
    sigma_pix = args.smooth_sigma_deg / args.binsz
    bubble_plus_residual_dnde_sm = smooth_nan_2d(bubble_plus_residual_dnde, sigma_pix=sigma_pix)

    # Split to positive/negative templates
    T_pos = np.where(bubble_plus_residual_dnde_sm > 0, bubble_plus_residual_dnde_sm, 0.0)
    T_neg = np.where(bubble_plus_residual_dnde_sm < 0, -bubble_plus_residual_dnde_sm, 0.0)

    # Apply ROI cut, and normalise templates over |b|>=norm_disk_cut inside ROI
    T_pos[~roi2d] = 0.0
    T_neg[~roi2d] = 0.0

    s_pos = float(np.nansum(T_pos[norm_mask]))
    s_neg = float(np.nansum(T_neg[norm_mask]))

    if not np.isfinite(s_pos) or s_pos <= 0:
        raise RuntimeError("Positive bubble template is zero after processing")
    if not np.isfinite(s_neg) or s_neg <= 0:
        raise RuntimeError("Negative bubble template is zero after processing")
    T_pos /= s_pos
    T_neg /= s_neg

    # Build cubes with your conventions
    dnde_pos = np.empty((nE, ny, nx), float)
    dnde_neg = np.empty((nE, ny, nx), float)
    dnde_flat = np.empty((nE, ny, nx), float)
    for k in range(nE):
        dnde_pos[k] = T_pos / (omega * dE_mev[k])
        dnde_neg[k] = T_neg / (omega * dE_mev[k])
        dnde_flat[k] = T_flat_out / (omega * dE_mev[k])

    E2dnde_pos = dnde_pos * (Ectr_mev[:, None, None] ** 2)
    E2dnde_neg = dnde_neg * (Ectr_mev[:, None, None] ** 2)
    E2dnde_flat = dnde_flat * (Ectr_mev[:, None, None] ** 2)

    mu_pos = dnde_pos * expo * omega[None, :, :] * dE_mev[:, None, None]
    mu_neg = dnde_neg * expo * omega[None, :, :] * dE_mev[:, None, None]
    mu_flat = dnde_flat * expo * omega[None, :, :] * dE_mev[:, None, None]

    # Write
    out_pos_mu = os.path.join(args.outdir, "mu_bubbles_pos_counts.fits")
    out_pos_dnde = os.path.join(args.outdir, "bubbles_pos_dnde.fits")
    out_pos_e2 = os.path.join(args.outdir, "bubbles_pos_E2dnde.fits")

    out_neg_mu = os.path.join(args.outdir, "mu_bubbles_neg_counts.fits")
    out_neg_dnde = os.path.join(args.outdir, "bubbles_neg_dnde.fits")
    out_neg_e2 = os.path.join(args.outdir, "bubbles_neg_E2dnde.fits")

    out_flat_mu = os.path.join(args.outdir, "mu_bubbles_flat_counts.fits")
    out_flat_dnde = os.path.join(args.outdir, "bubbles_flat_dnde.fits")
    out_flat_e2 = os.path.join(args.outdir, "bubbles_flat_E2dnde.fits")

    write_primary_with_bunit(
        out_pos_mu,
        mu_pos,
        hdr,
        "counts",
        comments=[
            "Fermi bubbles positive residual template; expected counts per bin per pixel",
            f"Derived from bin closest to {args.ref_energy_gev} GeV, smoothed {args.smooth_sigma_deg} deg",
        ],
    )
    write_primary_with_bunit(
        out_pos_dnde,
        dnde_pos,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        comments=[
            "Fermi bubbles positive residual template; dN/dE per MeV",
            f"Derived from bin closest to {args.ref_energy_gev} GeV, smoothed {args.smooth_sigma_deg} deg",
        ],
    )
    write_primary_with_bunit(
        out_pos_e2,
        E2dnde_pos,
        hdr,
        "MeV cm-2 s-1 sr-1",
        comments=[
            "Fermi bubbles positive residual template; E^2 dN/dE",
        ],
    )

    write_primary_with_bunit(
        out_neg_mu,
        mu_neg,
        hdr,
        "counts",
        comments=[
            "Fermi bubbles negative residual template; expected counts per bin per pixel",
            f"Derived from bin closest to {args.ref_energy_gev} GeV, smoothed {args.smooth_sigma_deg} deg",
        ],
    )
    write_primary_with_bunit(
        out_neg_dnde,
        dnde_neg,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        comments=[
            "Fermi bubbles negative residual template; dN/dE per MeV",
            f"Derived from bin closest to {args.ref_energy_gev} GeV, smoothed {args.smooth_sigma_deg} deg",
        ],
    )
    write_primary_with_bunit(
        out_neg_e2,
        E2dnde_neg,
        hdr,
        "MeV cm-2 s-1 sr-1",
        comments=[
            "Fermi bubbles negative residual template; E^2 dN/dE",
        ],
    )

    write_primary_with_bunit(
        out_flat_mu,
        mu_flat,
        hdr,
        "counts",
        comments=[
            "Fermi bubbles flat template (FB flat); expected counts per bin per pixel",
            f"Flat boundary from {'--bubble-mask' if args.bubble_mask is not None else 'heuristic'}",
        ],
    )
    write_primary_with_bunit(
        out_flat_dnde,
        dnde_flat,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        comments=[
            "Fermi bubbles flat template (FB flat); dN/dE per MeV",
        ],
    )
    write_primary_with_bunit(
        out_flat_e2,
        E2dnde_flat,
        hdr,
        "MeV cm-2 s-1 sr-1",
        comments=[
            "Fermi bubbles flat template (FB flat); E^2 dN/dE",
        ],
    )

    # Small debug text
    with open(os.path.join(args.outdir, "debug_bubbles_fit_refbin.txt"), "w") as f:
        f.write("# Reference bin\n")
        f.write(f"k_ref {k_ref}\n")
        f.write(f"Ectr_GeV {Ectr_gev[k_ref]:.6g}\n")
        f.write("# Fit method: cell-by-cell weighted NNLS\n")
        f.write(f"cell_deg {args.cell_deg}\n")
        f.write(f"roi_lon_deg {args.roi_lon}\n")
        f.write(f"roi_lat_deg {args.roi_lat}\n")
        f.write(f"smooth_sigma_deg {args.smooth_sigma_deg}\n")
        f.write("# Note: no single global amplitude vector is produced in cell-by-cell mode.\n")

    print("✓ Wrote:")
    print(" ", out_pos_mu)
    print(" ", out_pos_dnde)
    print(" ", out_pos_e2)
    print(" ", out_neg_mu)
    print(" ", out_neg_dnde)
    print(" ", out_neg_e2)
    print(" ", out_flat_mu)
    print(" ", out_flat_dnde)
    print(" ", out_flat_e2)


if __name__ == "__main__":
    main()
