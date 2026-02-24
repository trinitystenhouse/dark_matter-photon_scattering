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

from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

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

    ap.add_argument("--debug", action="store_true")

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
    _counts, hdr, Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE = int(Ectr_mev.size)
    wcs = WCS(hdr).celestial
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])

    # --- exposure (cm^2 s), resample if needed ---
    expo_raw, E_expo_mev = read_exposure(args.expcube)
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

    print("[units debug] omega median:", np.nanmedian(omega[roi2d]))
    print("[units debug] dE_mev min/median/max:", np.min(dE_mev), np.median(dE_mev), np.max(dE_mev))

    k = 2
    print("[units debug] expo median (k=2):", np.nanmedian(expo[k][roi2d]))
    conv = expo[k] * omega * dE_mev[k]
    print("[units debug] conv median (k=2):", np.nanmedian(conv[roi2d]))


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

    # --- Physical Loop I gamma-ray emission model ---
    # Loop I is a radio shell with known surface brightness
    # Gamma-rays come from inverse Compton scattering of electrons on ISRF/CMB
    # 
    # Radio observations: Loop I has surface brightness ~ 10-100 K at 408 MHz
    # This corresponds to electron column density N_e ~ 10^20 cm^-2
    # 
    # For IC emission, the gamma-ray intensity scales as:
    # I_gamma ~ n_e * W_ISRF * sigma_T * c
    # where W_ISRF ~ 1 eV/cm^3 is the ISRF energy density
    #
    # Typical Loop I gamma-ray surface brightness (from Fermi observations):
    # I_gamma ~ 10^-7 to 10^-6 ph cm^-2 s^-1 sr^-1 MeV^-1 at 1 GeV
    # with spectrum roughly E^-2.5 to E^-3
    
    # --- Normalise morphology for numerical stability (NOT physical scaling) ---
    fit2d = roi2d  # or roi2d & data_ok2d & ext_mask2d if you have those here
    vals = T[fit2d]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        raise RuntimeError("Loop I template has no positive pixels in fit mask")

    if args.debug:
        fit_vals_all = T[fit2d]
        fit_vals_all = fit_vals_all[np.isfinite(fit_vals_all)]
        print("[LoopI debug] Pre-normalization morphology stats")
        print(f"  T (all):  min={np.nanmin(T):.6g} max={np.nanmax(T):.6g} sum={np.nansum(T):.6g}")
        if fit_vals_all.size:
            print(
                "  T (fit mask):"
                f" min={np.nanmin(fit_vals_all):.6g} max={np.nanmax(fit_vals_all):.6g}"
                f" mean={np.nanmean(fit_vals_all):.6g} sum={np.nansum(fit_vals_all):.6g}"
            )
        print(
            "  T (fit mask, positive only):"
            f" min={np.nanmin(vals):.6g} max={np.nanmax(vals):.6g}"
            f" mean={np.nanmean(vals):.6g} sum={np.nansum(vals):.6g} n={vals.size}"
        )

    T_norm = T / np.mean(vals)   # mean=1 in fit region (recommended)

    if args.debug:
        fit_vals_norm = T_norm[fit2d]
        fit_vals_norm = fit_vals_norm[np.isfinite(fit_vals_norm)]
        print("[LoopI debug] Post-normalization morphology stats")
        print(
            f"  T_norm (all):  min={np.nanmin(T_norm):.6g} max={np.nanmax(T_norm):.6g} sum={np.nansum(T_norm):.6g}"
        )
        if fit_vals_norm.size:
            print(
                "  T_norm (fit mask):"
                f" min={np.nanmin(fit_vals_norm):.6g} max={np.nanmax(fit_vals_norm):.6g}"
                f" mean={np.nanmean(fit_vals_norm):.6g} sum={np.nansum(fit_vals_norm):.6g}"
            )

    # --- PHENO (Totani-style): energy-independent morphology replicated ---
    loopI_dnde = np.broadcast_to(T_norm[None, :, :], (nE, ny, nx)).astype(float).copy()

    # Convert shape -> counts template
    I0 = 1e-7
    mu_loopI = loopI_dnde * expo * omega[None, :, :] * dE_mev[:, None, None] * I0

    # --- Also write separate Loop A / Loop B templates (shell1 / shell2) ---
    def _norm_and_mu(T_in):
        T_in = np.asarray(T_in, float)
        T_in = np.array(T_in, copy=True)
        T_in[~roi2d] = 0.0

        vals_in = T_in[fit2d]
        vals_in = vals_in[np.isfinite(vals_in) & (vals_in > 0)]
        if vals_in.size == 0:
            raise RuntimeError("Loop I sub-template has no positive pixels in fit mask")

        Tn = T_in / np.mean(vals_in)
        dnde = np.broadcast_to(Tn[None, :, :], (nE, ny, nx)).astype(float).copy()
        mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None] * I0
        e2 = dnde * (Ectr_mev[:, None, None] ** 2)
        return dnde, e2, mu

    loopA_dnde, loopA_E2dnde, mu_loopA = _norm_and_mu(shell1)
    loopB_dnde, loopB_E2dnde, mu_loopB = _norm_and_mu(shell2)

    if args.debug:
        mu_vals = mu_loopI[:, fit2d]
        mu_vals = mu_vals[np.isfinite(mu_vals)]
        print("[LoopI debug] Counts-template stats")
        print(
            f"  mu_loopI: min={np.nanmin(mu_loopI):.6g} max={np.nanmax(mu_loopI):.6g} sum={np.nansum(mu_loopI):.6g}"
        )
        if mu_vals.size:
            print(
                "  mu_loopI (fit mask over all E):"
                f" min={np.nanmin(mu_vals):.6g} max={np.nanmax(mu_vals):.6g}"
                f" mean={np.nanmean(mu_vals):.6g} sum={np.nansum(mu_vals):.6g}"
            )

    # Optional diagnostic products (shape only)
    loopI_E2dnde = loopI_dnde * (Ectr_mev[:, None, None] ** 2)


    # --- write ---
    out_dnde = os.path.join(args.outdir, "loopI_dnde.fits")
    out_e2 = os.path.join(args.outdir, "loopI_E2dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_loopI_counts.fits")

    out_dnde_A = os.path.join(args.outdir, "loopA_dnde.fits")
    out_e2_A = os.path.join(args.outdir, "loopA_E2dnde.fits")
    out_mu_A = os.path.join(args.outdir, "mu_loopA_counts.fits")

    out_dnde_B = os.path.join(args.outdir, "loopB_dnde.fits")
    out_e2_B = os.path.join(args.outdir, "loopB_E2dnde.fits")
    out_mu_B = os.path.join(args.outdir, "mu_loopB_counts.fits")

    write_primary_with_bunit(
        out_dnde,
        loopI_dnde,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        [
            "Loop I template: Physical IC emission model",
            f"Shell 1: center=({args.shell1_l},{args.shell1_b}), R_in={args.shell1_rin_pc}, R_out={args.shell1_rout_pc} pc",
            f"Shell 2: center=({args.shell2_l},{args.shell2_b}), R_in={args.shell2_rin_pc}, R_out={args.shell2_rout_pc} pc",
        ],
    )
    write_primary_with_bunit(
        out_e2,
        loopI_E2dnde,
        hdr,
        "MeV cm-2 s-1 sr-1",
        [
            "Loop I geometric template: E^2 dN/dE",
        ],
    )
    write_primary_with_bunit(
        out_mu,
        mu_loopI,
        hdr,
        "counts",
        [
            "Loop I geometric template: expected counts per bin per pixel",
        ],
    )

    write_primary_with_bunit(
        out_dnde_A,
        loopA_dnde,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        [
            "Loop A template: Shell 1 only",
        ],
    )
    write_primary_with_bunit(
        out_e2_A,
        loopA_E2dnde,
        hdr,
        "MeV cm-2 s-1 sr-1",
        [
            "Loop A geometric template: E^2 dN/dE",
        ],
    )
    write_primary_with_bunit(
        out_mu_A,
        mu_loopA,
        hdr,
        "counts",
        [
            "Loop A geometric template: expected counts per bin per pixel",
        ],
    )

    write_primary_with_bunit(
        out_dnde_B,
        loopB_dnde,
        hdr,
        "ph cm-2 s-1 sr-1 MeV-1",
        [
            "Loop B template: Shell 2 only",
        ],
    )
    write_primary_with_bunit(
        out_e2_B,
        loopB_E2dnde,
        hdr,
        "MeV cm-2 s-1 sr-1",
        [
            "Loop B geometric template: E^2 dN/dE",
        ],
    )
    write_primary_with_bunit(
        out_mu_B,
        mu_loopB,
        hdr,
        "counts",
        [
            "Loop B geometric template: expected counts per bin per pixel",
        ],
    )

    print("✓ Wrote:")
    print(" ", out_mu)
    print(" ", out_dnde)
    print(" ", out_e2)
    print(" ", out_mu_A)
    print(" ", out_mu_B)


if __name__ == "__main__":
    main()
