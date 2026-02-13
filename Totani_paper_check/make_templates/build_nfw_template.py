#!/usr/bin/env python3
"""
Build an NFW (rho^p) template on the counts CCUBE grid.

Outputs (same conventions as your PS template):
  - nfw*_dnde.fits      : dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - nfw*_E2dnde.fits    : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - mu_nfw*_counts.fits : expected counts per bin per pixel [counts]

Key fix vs your previous version:
  ✅ Exposure resampling is now deterministic and explicit: linear interpolation in log(E),
     clamped to the exposure energy range, evaluated at CCUBE bin CENTERS by default.
  ✅ (Optional) You can switch to edge-mean exposure per bin (mean of expo(Emin), expo(Emax))
     which sometimes matches older pipelines.

Notes:
- Spatial template is normalised so sum over ROI pixels = 1 (pixel-sum, as you had).
- Spectrum is unit-normalised over bins (sum_k Phi_bin = 1 ph/cm^2/s).
  Fit coefficient then corresponds to integrated flux over the energy range.
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from make_nfw_rho25 import make_nfw_template, make_nfw_rho25_template
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=None, help="Counts CCUBE (authoritative WCS + EBOUNDS)")
    ap.add_argument("--expo", default=None, help="Exposure cube (expcube)")
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)

    ap.add_argument("--halo-gamma", type=float, default=1.25)
    ap.add_argument("--rho-power", type=float, default=2.5)
    ap.add_argument("--n-s", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=8)

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

    suffix = f"_rho{args.rho_power:g}_g{args.halo_gamma:g}"
    if args.expo_sampling != "center":
        suffix += f"_expo{args.expo_sampling}"

    out_dnde = os.path.join(outdir, f"nfw{suffix}_dnde.fits")
    out_e2dnde = os.path.join(outdir, f"nfw{suffix}_E2dnde.fits")
    out_mu = os.path.join(outdir, f"mu_nfw{suffix}_counts.fits")

    # --- Read CCUBE header + EBOUNDS (authoritative energy binning) ---
    with fits.open(counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]

        if "EBOUNDS" not in hc:
            raise RuntimeError("Counts CCUBE missing EBOUNDS extension.")
        eb = hc["EBOUNDS"].data
        eb_hdu = hc["EBOUNDS"].copy()  # we'll copy into outputs for convenience

    Emin_kev = np.array(eb["E_MIN"], dtype=float)
    Emax_kev = np.array(eb["E_MAX"], dtype=float)

    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    nE = int(Ectr_mev.size)

    # --- Read exposure cube + energies, resample to CCUBE binning ---
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

    # DEBUG: save the resampled exposure used to build mu
    out_expo_used = os.path.join(outdir, f"expo_used{suffix}.npy")
    np.save(out_expo_used, expo.astype(np.float32))
    print("✓ wrote", out_expo_used, "(resampled exposure used in mu)")
    print("[DBG] expo used per-bin sum:", np.nansum(expo, axis=(1,2)))


    # --- lon/lat grid ---
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0

    # --- spatial template ---
    if (args.rho_power == 2.5) and (args.halo_gamma == 1.25):
        nfw_spatial = make_nfw_rho25_template(lon, lat).astype(float)
    else:
        nfw_spatial = make_nfw_template(
            lon,
            lat,
            gamma=args.halo_gamma,
            rho_power=args.rho_power,
            n_s=args.n_s,
            chunk=args.chunk,
        ).astype(float)

    roi = (np.abs(lon) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    nfw_spatial[~roi] = 0.0

    norm = np.nansum(nfw_spatial)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("NFW template is zero everywhere in ROI")
    nfw_spatial /= norm

    # --- solid angle map (must match your fitter conventions) ---
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # --- spectrum: flat Phi_bin, unit normalised over bins ---
    Phi_bin = np.ones(nE, float)
    Phi_bin /= Phi_bin.sum()  # sum_k Phi_bin = 1

    # --- build cubes ---
    nfw_dnde = np.empty((nE, ny, nx), float)
    for k in range(nE):
        nfw_dnde[k] = (Phi_bin[k] * nfw_spatial) / (omega * dE_mev[k])

    nfw_E2dnde = nfw_dnde * (Ectr_mev[:, None, None] ** 2)
    mu_nfw = nfw_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

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

    _write_cube(
        out_dnde,
        nfw_dnde,
        "ph cm-2 s-1 sr-1 MeV-1",
        [
            f"gNFW (gamma={args.halo_gamma:g}) rho^{args.rho_power:g} spatial template; sum ROI pixels = 1",
            "Spectrum: Phi_bin flat, renormalised so sum_k Phi_bin = 1 ph/cm^2/s",
            expo_comment,
        ],
    )

    _write_cube(
        out_e2dnde,
        nfw_E2dnde,
        "MeV cm-2 s-1 sr-1",
        [
            "E^2 dN/dE version of NFW template",
            f"(Derived from dnde using Ectr^2, where Ectr from CCUBE EBOUNDS geometric mean.)",
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
    print("sum Phi_bin:", float(Phi_bin.sum()))
    print("mu_nfw total counts:", float(np.nansum(mu_nfw)))
    print("mu_nfw per-bin:", np.nansum(mu_nfw, axis=(1, 2)))


if __name__ == "__main__":
    main()
