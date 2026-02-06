#!/usr/bin/env python3
"""build_loopI_template.py

Build a Loop I geometric template on the counts CCUBE grid, following the same
output conventions as the other template builders in this repo.

Because the exact shell parameters used by Totani (2025) are taken from LAT team
papers and are currently not encoded in this repository, this script makes the
geometry configurable. Once you decide on the shell parameters (centers/radii),
you can re-run and regenerate the template.

Outputs written to OUTDIR (defaults to fermi_data/processed/templates):
  - loopI_dnde.fits         : dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - loopI_E2dnde.fits       : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - mu_loopI_counts.fits    : expected counts per bin per pixel [counts]

The spatial template is energy-independent (only its normalization per bin is
fit), consistent with Totani's treatment of energy-independent morphologies.
"""

import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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


def shell_chord_length(lon_deg, lat_deg, l0_deg, b0_deg, dist_pc, r_in_pc, r_out_pc):
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)
    l0 = np.deg2rad(l0_deg)
    b0 = np.deg2rad(b0_deg)

    cosang = np.sin(b) * np.sin(b0) + np.cos(b) * np.cos(b0) * np.cos(l - l0)
    cosang = np.clip(cosang, -1.0, 1.0)
    theta = np.arccos(cosang)

    D = float(dist_pc)
    if D <= 0:
        raise RuntimeError("dist_pc must be > 0")

    p = D * np.sin(theta)

    Rin = float(r_in_pc)
    Rout = float(r_out_pc)
    if Rout <= 0 or Rin < 0 or Rout <= Rin:
        raise RuntimeError("Require r_out_pc > r_in_pc >= 0")

    Lout = np.zeros_like(p, dtype=float)
    ok_out = p < Rout
    Lout[ok_out] = 2.0 * np.sqrt(np.maximum(Rout * Rout - p[ok_out] * p[ok_out], 0.0))

    Lin = np.zeros_like(p, dtype=float)
    ok_in = p < Rin
    Lin[ok_in] = 2.0 * np.sqrt(np.maximum(Rin * Rin - p[ok_in] * p[ok_in], 0.0))

    return np.maximum(Lout - Lin, 0.0)


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
        "--outdir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    ap.add_argument("--shell1-l", type=float, default=341.0)
    ap.add_argument("--shell1-b", type=float, default=3.0)
    ap.add_argument("--shell1-dist-pc", type=float, default=78.0)
    ap.add_argument("--shell1-rin-pc", type=float, default=62.0)
    ap.add_argument("--shell1-rout-pc", type=float, default=81.0)

    ap.add_argument("--shell2-l", type=float, default=332.0)
    ap.add_argument("--shell2-b", type=float, default=37.0)
    ap.add_argument("--shell2-dist-pc", type=float, default=95.0)
    ap.add_argument("--shell2-rin-pc", type=float, default=58.0)
    ap.add_argument("--shell2-rout-pc", type=float, default=82.0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load counts grid + energies ---
    with fits.open(args.counts) as h:
        hdr = h[0].header
        eb = h["EBOUNDS"].data

    wcs = WCS(hdr).celestial
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])

    Emin_kev = eb["E_MIN"].astype(float)
    Emax_kev = eb["E_MAX"].astype(float)
    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)

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
    lon = ((lon + 180.0) % 360.0) - 180.0
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # --- ROI mask ---
    roi2d = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)

    shell1 = shell_chord_length(
        lon,
        lat,
        args.shell1_l,
        args.shell1_b,
        args.shell1_dist_pc,
        args.shell1_rin_pc,
        args.shell1_rout_pc,
    )
    shell2 = shell_chord_length(
        lon,
        lat,
        args.shell2_l,
        args.shell2_b,
        args.shell2_dist_pc,
        args.shell2_rin_pc,
        args.shell2_rout_pc,
    )

    T = shell1 + shell2

    T[~roi2d] = 0.0

    s = float(np.nansum(T))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Loop I template is zero in ROI; check shell parameters")

    T /= s

    # --- build cubes with your conventions ---
    loopI_dnde = np.empty((nE, ny, nx), float)
    for k in range(nE):
        loopI_dnde[k] = T / (omega * dE_mev[k])

    loopI_E2dnde = loopI_dnde * (Ectr_mev[:, None, None] ** 2)

    mu_loopI = loopI_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    # --- write ---
    out_dnde = os.path.join(args.outdir, "loopI_dnde.fits")
    out_e2 = os.path.join(args.outdir, "loopI_E2dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_loopI_counts.fits")

    write_primary_with_bunit(
        out_dnde,
        loopI_dnde,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        comments=[
            "Loop I geometric template (two-shell rings) on counts grid",
            "Energy-independent morphology; per-bin intensity scales as 1/(Omega*dE)",
            f"shell1 center(l,b)=({args.shell1_l},{args.shell1_b}) deg, D={args.shell1_dist_pc} pc, r=[{args.shell1_rin_pc},{args.shell1_rout_pc}] pc",
            f"shell2 center(l,b)=({args.shell2_l},{args.shell2_b}) deg, D={args.shell2_dist_pc} pc, r=[{args.shell2_rin_pc},{args.shell2_rout_pc}] pc",
        ],
    )
    write_primary_with_bunit(
        out_e2,
        loopI_E2dnde,
        hdr,
        "MeV cm-2 s-1 sr-1",
        comments=[
            "Loop I geometric template: E^2 dN/dE",
        ],
    )
    write_primary_with_bunit(
        out_mu,
        mu_loopI,
        hdr,
        "counts",
        comments=[
            "Loop I geometric template: expected counts per bin per pixel",
        ],
    )

    print("✓ Wrote:")
    print(" ", out_mu)
    print(" ", out_dnde)
    print(" ", out_e2)


if __name__ == "__main__":
    main()
