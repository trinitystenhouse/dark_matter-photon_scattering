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
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
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


def make_los_grid_kpc(rvir_kpc=402.0, R0_kpc=8.0, n_s=2048, smax_kpc=None):
    """
    Build a 1D LOS distance grid s in kpc and weights ds for integration.
    We integrate from s=0 to s=smax. If smax not provided, use R0 + rvir.

    Uses a mixed grid: dense at small s (log), then linear tail.
    This reduces error near the GC without insane n_s.
    """
    if smax_kpc is None:
        smax_kpc = R0_kpc + rvir_kpc  # safely beyond far edge for most directions

    # fraction for log portion
    f_log = 0.6
    n_log = max(16, int(f_log * n_s))
    n_lin = max(16, n_s - n_log)

    s_min = 1e-6  # kpc
    s_break = min(5.0, 0.2 * smax_kpc)  # put breakpoint at a few kpc-ish

    s_log = np.geomspace(s_min, s_break, n_log, endpoint=False)
    s_lin = np.linspace(s_break, smax_kpc, n_lin)

    s = np.concatenate([s_log, s_lin])
    ds = np.diff(s)
    # for simple Riemann: use midpoints
    s_mid = 0.5 * (s[:-1] + s[1:])
    return s_mid, ds


def los_integral_rhopow_map(
    lon2d,
    lat2d,
    rho_power=2.0,
    gamma=1.0,
    rs_kpc=21.0,
    rhos=8.1e6,
    R0_kpc=8.0,
    rvir_kpc=402.0,
    n_s=2048,
    smax_kpc=None,
    chunk=8,
):
    """
    Compute J_p(l,b) = ∫ rho(r(l,b,s))^p ds on a 2D lon/lat grid.

    chunk: process in lat-stripes to limit memory
    """
    ny, nx = lon2d.shape
    J = np.zeros((ny, nx), dtype=np.float64)

    s_mid, ds = make_los_grid_kpc(rvir_kpc=rvir_kpc, R0_kpc=R0_kpc, n_s=n_s, smax_kpc=smax_kpc)

    # stripe over y to avoid allocating huge (ny,nx,ns)
    for y0 in range(0, ny, chunk):
        y1 = min(ny, y0 + chunk)
        lon = lon2d[y0:y1, :]
        lat = lat2d[y0:y1, :]

        acc = np.zeros((y1 - y0, nx), dtype=np.float64)

        # integrate with simple midpoint rule
        for si, dsi in zip(s_mid, ds):
            r = los_r_kpc(lon, lat, si, R0_kpc=R0_kpc)
            rho = rho_nfw(r, rs_kpc=rs_kpc, rhos=rhos, gamma=gamma)
            acc += (rho ** rho_power) * dsi

        J[y0:y1, :] = acc

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
        default="pole+roi-sum",
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

    # --- Read CCUBE header + EBOUNDS ---
    with fits.open(counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]

        if "EBOUNDS" not in hc:
            raise RuntimeError("Counts CCUBE missing EBOUNDS extension.")
        eb = hc["EBOUNDS"].data
        eb_hdu = hc["EBOUNDS"].copy()

    Emin_kev = np.array(eb["E_MIN"], dtype=float)
    Emax_kev = np.array(eb["E_MAX"], dtype=float)

    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    nE = int(Ectr_mev.size)

    # --- Read exposure cube + resample to CCUBE E bins ---
    with fits.open(expo_path) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(he)

    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure spatial shape {expo_raw.shape[1:]} != counts {(ny, nx)}")

    if args.expo_sampling == "center":
        expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)
        expo_comment = "Exposure evaluated at CCUBE bin centers Ectr, interpolated linearly in logE (clamped)."
    else:
        expo_min = resample_exposure_logE_interp(expo_raw, E_expo_mev, Emin_mev)
        expo_max = resample_exposure_logE_interp(expo_raw, E_expo_mev, Emax_mev)
        expo = 0.5 * (expo_min + expo_max)
        expo_comment = "Exposure = 0.5*(expo(Emin)+expo(Emax)), each interpolated linearly in logE (clamped)."

    if expo.shape[0] != nE:
        raise RuntimeError(f"Resampled exposure has {expo.shape[0]} planes, expected {nE}")

    # --- lon/lat grid ---
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0

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
        smax_kpc=args.smax_kpc,
        chunk=args.chunk,
    )

    # ROI mask (keep exactly as your pipeline)
    roi = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    J[~roi] = 0.0

    # --- Normalization ---
    if args.norm in ("pole", "pole+roi-sum"):
        Jpole = pole_normalization_value(
            rho_power=args.rho_power,
            gamma=args.gamma,
            rs_kpc=args.rs_kpc,
            rhos=args.rhos,
            R0_kpc=args.R0_kpc,
            rvir_kpc=args.rvir_kpc,
            n_s=max(4096, args.n_s),  # make pole norm stable
            smax_kpc=args.smax_kpc,
        )
        if not np.isfinite(Jpole) or Jpole <= 0:
            raise RuntimeError("Pole normalization integral is non-positive.")
        J = J / Jpole

    if args.norm in ("roi-sum", "pole+roi-sum"):
        s_roi = float(np.nansum(J[roi]))
        if not np.isfinite(s_roi) or s_roi <= 0:
            raise RuntimeError("ROI sum normalization failed (template is zero in ROI).")
        J = J / s_roi

    nfw_spatial = J.astype(np.float64)

    # --- solid angle map (match fitter conventions) ---
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # --- PHENO mode: energy-independent shape replicated across bins ---
    mu_nfw = np.broadcast_to(nfw_spatial[None, :, :], (nE, ny, nx)).astype(np.float64).copy()

    denom = expo * omega[None, :, :] * dE_mev[:, None, None]
    nfw_dnde = np.full_like(mu_nfw, np.nan, dtype=np.float64)
    ok = np.isfinite(denom) & (denom > 0)
    nfw_dnde[ok] = mu_nfw[ok] / denom[ok]
    nfw_E2dnde = nfw_dnde * (Ectr_mev[:, None, None] ** 2)

    # -------------------------
    # Write outputs (include EBOUNDS so checkers don’t need counts file)
    # -------------------------
    def _write_cube(path, data, bunit, comments):
        phdu = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
        phdu.header["BUNIT"] = bunit
        for c in comments:
            phdu.header["COMMENT"] = c
        hdul = fits.HDUList([phdu, eb_hdu])
        hdul.writeto(path, overwrite=True)

    comments_main = [
        "LOS-projected NFW template: J_p(l,b) = int rho(r(l,b,s))^p ds",
        f"Profile: gNFW with gamma={args.gamma:g}, rs={args.rs_kpc:g} kpc, R0={args.R0_kpc:g} kpc, rvir={args.rvir_kpc:g} kpc",
        f"rho_power p = {args.rho_power:g}",
        f"LOS grid: n_s={args.n_s:d}, smax={args.smax_kpc if args.smax_kpc is not None else (args.R0_kpc + args.rvir_kpc):g} kpc",
        f"Normalization: {args.norm}",
        "ROI cut applied: |l|<=roi_lon and |b|<=roi_lat; outside ROI set to 0",
        expo_comment,
    ]

    _write_cube(
        out_dnde,
        nfw_dnde,
        "ph cm-2 s-1 sr-1 MeV-1",
        comments_main,
    )

    _write_cube(
        out_e2dnde,
        nfw_E2dnde,
        "MeV cm-2 s-1 sr-1",
        [
            "E^2 dN/dE version of NFW template",
            "(Derived from dnde using Ectr^2, where Ectr from CCUBE EBOUNDS geometric mean.)",
            expo_comment,
        ],
    )

    _write_cube(
        out_mu,
        mu_nfw,
        "counts",
        [
            "Expected counts = dN/dE * exposure * Omega_pix * dE",
            expo_comment,
        ],
    )

    print("✓ wrote", out_dnde)
    print("✓ wrote", out_e2dnde)
    print("✓ wrote", out_mu)
    print("mu_nfw total (shape units):", float(np.nansum(mu_nfw)))
    print("mu_nfw per-bin:", np.nansum(mu_nfw, axis=(1, 2)))


if __name__ == "__main__":
    main()
