#!/usr/bin/env python3
"""
Build an NFW (rho^p) template on the counts CCUBE grid using the *explicit LOS integral*:

  T_p(l,b) ∝ ∫_0^{smax} ρ[r(l,b,s)]^p ds

with
  r(l,b,s) = sqrt(R0^2 + s^2 - 2 R0 s cos b cos l)

This matches the description in Totani-style analyses: pick NFW params, pick p (1,2,2.5),
project along the line of sight, and rasterize into the CCUBE pixel grid.

Outputs (same conventions as your PS template):
  - nfw*_dnde.fits      : dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - nfw*_E2dnde.fits    : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - mu_nfw*_counts.fits : expected counts per bin per pixel [counts]

Notes / conventions:
- "pheno" mode: mu is just the *shape* (energy-independent) repeated over E bins.
- Exposure sampling/resampling stays exactly as you had (logE interpolation, center or edge-mean).
- Spatial normalisation options:
    (A) Totani-like: normalize by *Galactic pole* value so T_p(b=±90°)=1.
    (B) Your fitter-like: normalize so sum over ROI pixels = 1.
  Default below: do pole-normalize FIRST, then (optionally) ROI-sum normalize.

NFW parameters default to Via Lactea II-like values used by Totani:
  rs = 21 kpc, rvir = 402 kpc, R0 = 8 kpc
  rhos given in Msun/kpc^3 (units irrelevant for shape-only templates)
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
    write_cube,
)
# ----------------------------
# NFW + LOS projection helpers
# ----------------------------

def rho_nfw(r_kpc, rs_kpc=21.0, rhos=8.1e6, gamma=1.0):
    """
    Generalised NFW density in arbitrary units (default rhos in Msun/kpc^3):
      rho(r) = rhos * x^{-gamma} * (1+x)^{gamma-3}, x=r/rs
    For standard NFW, gamma=1.
    """
    x = np.maximum(r_kpc / rs_kpc, 1e-12)
    return rhos * (x ** (-gamma)) * ((1.0 + x) ** (gamma - 3.0))


def los_r_kpc(lon_deg, lat_deg, s_kpc, R0_kpc=8.0):
    """
    Galactocentric radius for LOS distance s (kpc), for given lon/lat in degrees.
    Uses lon in [-180,180] convention (cos handles either).
    """
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)
    cospsi = np.cos(b) * np.cos(l)
    # r^2 = R0^2 + s^2 - 2 R0 s cospsi
    r2 = (R0_kpc * R0_kpc) + (s_kpc * s_kpc) - (2.0 * R0_kpc * s_kpc * cospsi)
    return np.sqrt(np.maximum(r2, 0.0))


def make_u_grid(n_s=1024):
    """
    Unit integration coordinate u in [0,1] for per-pixel mapping to s in [s0, smax(l,b)].
    """
    u = np.linspace(0.0, 1.0, int(n_s), dtype=np.float64)
    return u


def make_los_grid_kpc(*, rvir_kpc=402.0, R0_kpc=8.0, n_s=4096, smax_kpc=None):
    """Linear LOS grid for 1D integrations.

    Returns:
      s_mid: (n_s,) midpoints in kpc
      ds: (n_s,) bin widths in kpc
    """
    if smax_kpc is None:
        smax_kpc = float(R0_kpc + rvir_kpc)
    edges = np.linspace(0.0, float(smax_kpc), int(n_s) + 1, dtype=np.float64)
    ds = np.diff(edges)
    s_mid = 0.5 * (edges[:-1] + edges[1:])
    return s_mid, ds

def smax_to_rvir_kpc(lon_deg, lat_deg, R0_kpc=8.0, rvir_kpc=402.0):
    """
    Far intersection distance (forward LOS) where r(s)=rvir.
    Solve: s^2 - 2 R0 cospsi s + (R0^2 - rvir^2) = 0
    """
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)
    cospsi = np.cos(b) * np.cos(l)

    disc = (R0_kpc * cospsi)**2 - (R0_kpc**2 - rvir_kpc**2)
    disc = np.maximum(disc, 0.0)
    s_far = R0_kpc * cospsi + np.sqrt(disc)
    return np.maximum(s_far, 0.0)

def los_integral_rhopow_map(
    lon2d,
    lat2d,
    rho_power=2.0,
    gamma=1.0,
    rs_kpc=21.0,
    rhos=8.1e6,
    R0_kpc=8.0,
    rvir_kpc=402.0,
    n_s=1024,
    s0_kpc=1e-3,          # start > 0 for log spacing
    s_spacing="log",      # "log" or "linear"
    chunk=8,              # stripe height
):
    """
    J_p(l,b) = ∫_0^{smax(l,b)} rho(r(l,b,s))^p ds, truncated at rvir via smax(l,b).
    Vectorized over s for each stripe for speed.
    """
    ny, nx = lon2d.shape
    J = np.zeros((ny, nx), dtype=np.float64)

    u = make_u_grid(n_s=n_s)

    for y0 in range(0, ny, chunk):
        y1 = min(ny, y0 + chunk)
        lon = lon2d[y0:y1, :].astype(np.float64)
        lat = lat2d[y0:y1, :].astype(np.float64)

        # per-pixel smax to the halo boundary
        smax = smax_to_rvir_kpc(lon, lat, R0_kpc=R0_kpc, rvir_kpc=rvir_kpc)

        # Pixels with essentially zero path length
        good = smax > s0_kpc
        if not np.any(good):
            continue

        # Build s-grid per pixel: s has shape (n_s, NyStripe, Nx)
        if s_spacing == "log":
            ratio = np.where(good, smax / s0_kpc, 1.0)
            s = s0_kpc * (ratio[None, :, :] ** u[:, None, None])
        elif s_spacing == "linear":
            s = s0_kpc + (smax[None, :, :] - s0_kpc) * u[:, None, None]
        else:
            raise ValueError("s_spacing must be 'log' or 'linear'")

        # r(s)^2 = R0^2 + s^2 - 2 R0 s cospsi
        lrad = np.deg2rad(lon)
        brad = np.deg2rad(lat)
        cospsi = np.cos(brad) * np.cos(lrad)

        r2 = (R0_kpc**2 + s**2 - 2.0 * R0_kpc * s * cospsi[None, :, :])
        r = np.sqrt(np.maximum(r2, 0.0))

        rho = rho_nfw(r, rs_kpc=rs_kpc, rhos=rhos, gamma=gamma)
        integrand = rho ** rho_power

        # Trapezoid integrate along s axis (axis=0)
        _trapezoid = getattr(np, "trapezoid", np.trapz)
        Jstripe = _trapezoid(integrand, s, axis=0)

        # If smax <= s0, Jstripe is junk; enforce mask
        Jstripe[~good] = 0.0
        J[y0:y1, :] = Jstripe

    return J

def pole_normalization_value(
    rho_power=2.0,
    gamma=1.0,
    rs_kpc=21.0,
    rhos=8.1e6,
    R0_kpc=8.0,
    rvir_kpc=402.0,
    n_s=4096,
    smax_kpc=None,
):
    """
    Compute J_p at the Galactic pole (b=+90 deg). There cos b = 0, so r = sqrt(R0^2 + s^2).
    This matches the "normalize to GP" convention used to quote pole LOS integrals.
    """
    s_mid, ds = make_los_grid_kpc(rvir_kpc=rvir_kpc, R0_kpc=R0_kpc, n_s=n_s, smax_kpc=smax_kpc)
    r = np.sqrt((R0_kpc * R0_kpc) + (s_mid * s_mid))
    rho = rho_nfw(r, rs_kpc=rs_kpc, rhos=rhos, gamma=gamma)
    Jpole = np.sum((rho ** rho_power) * ds)
    return float(Jpole)


# ----------------------------
# Main script
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["pheno"], default="pheno")  # keep as you had; "physical" removed here

    ap.add_argument("--counts", default=None, help="Counts CCUBE (authoritative WCS + EBOUNDS)")
    ap.add_argument("--expo", default=None, help="Exposure cube (expcube)")
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    # NFW / LOS parameters
    ap.add_argument("--rho-power", type=float, default=2.5, help="p in ∫rho^p ds (e.g. 1,2,2.5)")
    ap.add_argument("--gamma", type=float, default=1.0, help="inner slope gamma for gNFW (gamma=1 is NFW)")
    ap.add_argument("--rs-kpc", type=float, default=21.0)
    ap.add_argument("--rhos", type=float, default=8.1e6, help="density scale (units irrelevant in pheno mode)")
    ap.add_argument("--R0-kpc", type=float, default=8.0)
    ap.add_argument("--rvir-kpc", type=float, default=402.0)
    ap.add_argument("--smax-kpc", type=float, default=None, help="LOS upper limit; default is R0+rvir")

    ap.add_argument("--n-s", type=int, default=2048, help="LOS grid resolution")
    ap.add_argument("--chunk", type=int, default=8, help="stripe height for memory control")

    # Normalisation choices
    ap.add_argument(
        "--norm",
        choices=["pole", "roi-sum", "pole+roi-sum"],
        default="pole",
        help="How to normalize the spatial template.",
    )

    # Exposure sampling
    ap.add_argument(
        "--expo-sampling",
        choices=["center", "edge-mean"],
        default="center",
        help="How to evaluate exposure per CCUBE bin: at bin center, or mean of exposure at (Emin,Emax).",
    )

    args = ap.parse_args()

    repo_dir = os.environ["REPO_PATH"]
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

    counts = args.counts or os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    expo_path = args.expo or os.path.join(data_dir, "processed", "expcube_1000to1000000.fits")
    outdir = args.outdir or os.path.join(data_dir, "processed", "templates")
    os.makedirs(outdir, exist_ok=True)

    suffix = (
        f"_NFW_g{args.gamma:g}_rho{args.rho_power:g}"
        f"_rs{args.rs_kpc:g}_R0{args.R0_kpc:g}_rvir{args.rvir_kpc:g}"
        f"_ns{args.n_s:d}"
    )
    suffix += f"_norm{args.norm}"
    if args.expo_sampling != "center":
        suffix += f"_expo{args.expo_sampling}"
    suffix += "_pheno"

    out_dnde = os.path.join(outdir, f"nfw{suffix}_dnde.fits")
    out_e2dnde = os.path.join(outdir, f"nfw{suffix}_E2dnde.fits")
    out_mu = os.path.join(outdir, f"mu_nfw{suffix}_counts.fits")

    # --- Read CCUBE (authoritative WCS + EBOUNDS) ---
    counts_cube, hdr, _Emin, _Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(counts)
    nE, ny, nx = counts_cube.shape
    wcs = WCS(hdr).celestial

    # --- Read exposure cube + energies, resample to CCUBE binning ---
    expo_raw, E_expo_mev = read_exposure(expo_path)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure spatial shape {expo_raw.shape[1:]} != counts {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError(f"Resampled exposure has {expo.shape[0]} planes, expected {nE}")
    expo_comment = "Exposure resampled to CCUBE E bins using logE interpolation."

    # --- lon/lat grid ---
    lon, lat = lonlat_grids(wcs, ny, nx)

    # --- Restrict to pixels with actual data coverage (template footprint) ---
    data_ok3d = np.isfinite(counts_cube) & np.isfinite(expo) & (expo > 0)
    data_ok2d = np.any(data_ok3d, axis=0)

    # --- Compute LOS-projected NFW template J_p(l,b) ---
    J = los_integral_rhopow_map(
        lon2d=lon,
        lat2d=lat,
        rho_power=args.rho_power,
        gamma=args.gamma,
        rs_kpc=args.rs_kpc,
        rhos=args.rhos,
        R0_kpc=args.R0_kpc,
        rvir_kpc=args.rvir_kpc,
        n_s=args.n_s,
    )

    # ROI mask (keep exactly as your pipeline)
    roi = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    roi &= data_ok2d

    # Optional: pole normalisation (interpretability)
    if args.norm in ("pole", "pole+roi-sum"):
        Jpole = pole_normalization_value(
            rho_power=args.rho_power,
            gamma=args.gamma,
            rs_kpc=args.rs_kpc,
            rhos=args.rhos,
            R0_kpc=args.R0_kpc,
            rvir_kpc=args.rvir_kpc,
            n_s=max(4096, args.n_s),
            smax_kpc=args.smax_kpc,
        )
        if not np.isfinite(Jpole) or Jpole <= 0:
            raise RuntimeError("Pole normalization integral is non-positive.")
        J = J / Jpole

    # For fitting stability: normalise spatial template to mean=1 within ROI
    # vals = J[roi]
    # vals = vals[np.isfinite(vals) & (vals > 0)]
    # if vals.size == 0:
    #     raise RuntimeError("NFW template is zero in ROI after masking.")
    # J = J / np.mean(vals)

    nfw_spatial = J.astype(np.float64)

    # Solid angle map
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    # Pole pixel is not exactly b=90 in your ROI grid, so sample a small cap
    pole_cap = roi & (lat > 59.0)  # or > 55, depending on ROI
    print("mean J in pole cap:", np.mean(J[pole_cap]))


    # --- COUNTS TEMPLATE (this is what your mu_list expects) ---
    I0 = 1e-7
    # denom here is the factor that converts flux -> counts
    conv = expo * omega[None, :, :] * dE_mev[:, None, None]   # shape (nE,ny,nx)
    mu_nfw = nfw_spatial[None, :, :] * conv  * I0                 # shape (nE,ny,nx)

    nfw_dnde = np.full_like(mu_nfw, np.nan, dtype=np.float64)
    ok = np.isfinite(conv) & (conv > 0)
    nfw_dnde[ok] = mu_nfw[ok] / conv[ok]        # equals nfw_spatial broadcast, by construction
    nfw_E2dnde = nfw_dnde * (Ectr_mev[:, None, None] ** 2)


    # -------------------------
    # Write outputs (include EBOUNDS so checkers don’t need counts file)
    # -------------------------

    write_cube(out_dnde, nfw_dnde, hdr, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_e2dnde, nfw_E2dnde, hdr, bunit="MeV cm-2 s-1 sr-1")
    write_cube(out_mu, mu_nfw, hdr, bunit="counts")

    print("✓ wrote", out_dnde)
    print("✓ wrote", out_e2dnde)
    print("✓ wrote", out_mu)
    print("mu_nfw total (shape units):", float(np.nansum(mu_nfw)))
    print("mu_nfw per-bin:", np.nansum(mu_nfw, axis=(1, 2)))


if __name__ == "__main__":
    main()
