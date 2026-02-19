#!/usr/bin/env python3

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_mapcube_primary,
    read_exposure,
    reproject_cube_to_target,
    resample_exposure_logE,
    write_cube,
)

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

def interp_energy_planes(cube, E_src, E_tgt):
    if E_src is None:
        raise RuntimeError("No energy axis available to interpolate.")

    return resample_exposure_logE(cube, E_src, E_tgt)

def main():
    import argparse

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
        "--ics",
        default=os.path.join(DATA_DIR, "templates", "ics_anisotropic_mapcube_54_0f770002.gz"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    _, hdr_cnt, Emin, Emax, Ectr, dE = read_counts_and_ebounds(args.counts)
    nE = int(Ectr.size)

    w_tgt = WCS(hdr_cnt).celestial
    ny, nx = hdr_cnt["NAXIS2"], hdr_cnt["NAXIS1"]

    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)

    expo, E_exp = read_exposure(args.expcube)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")
    if expo.shape[0] != nE:
        if E_exp is None:
            raise RuntimeError("Exposure has different #planes and no energy axis.")
        expo = interp_energy_planes(expo, E_exp, Ectr)

    ics_cube, w_src, E_ics, _hdr_ics = read_mapcube_primary(args.ics)
    if E_ics is None:
        if ics_cube.shape[0] != nE:
            raise RuntimeError("ICS cube has no energies and does not match counts energy axis.")
        ics_E = ics_cube
    else:
        if ics_cube.shape[0] != len(E_ics):
            raise RuntimeError("ICS cube energy axis length mismatch.")
        ics_E = interp_energy_planes(ics_cube, E_ics, Ectr) if ics_cube.shape[0] != nE else ics_cube

    ics_on_grid = reproject_cube_to_target(src_cube=ics_E, w_src=w_src, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx)

    # Interpret mapcube values as differential intensity dN/dE
    ics_dnde = ics_on_grid

    # Expected counts per bin:
    # mu = (dN/dE) * exposure * Ω_pix * ΔE
    mu_ics = ics_dnde * expo * omega[None, :, :] * dE[:, None, None]

    out_dnde = os.path.join(args.outdir, "ics_dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_ics_counts.fits")

    write_cube(out_dnde, ics_dnde, hdr_cnt, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_mu, mu_ics, hdr_cnt, bunit="counts")

    print("✓ Wrote", out_dnde)
    print("✓ Wrote", out_mu)
    print("  mu_ics total:", float(np.nansum(mu_ics)))


if __name__ == "__main__":
    main()
