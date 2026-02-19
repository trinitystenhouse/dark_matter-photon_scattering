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


def _mapcube_to_counts_energy_grid(cube, E_cube, E_tgt):
    if E_cube is None:
        if cube.shape[0] != len(E_tgt):
            raise RuntimeError("Mapcube has no energies and does not match counts energy axis.")
        return cube

    if cube.shape[0] != len(E_cube):
        raise RuntimeError("Mapcube energy axis length mismatch.")

    return interp_energy_planes(cube, E_cube, E_tgt) if cube.shape[0] != len(E_tgt) else cube


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
        "--pion",
        default=os.path.join(DATA_DIR, "templates", "pion_decay_mapcube_54_0f770002.gz"),
    )
    ap.add_argument(
        "--bremss",
        default=os.path.join(DATA_DIR, "templates", "bremss_mapcube_54_0f770002.gz"),
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

    pion_cube, w_pion, E_pion, _hdr_pion = read_mapcube_primary(args.pion)
    brem_cube, w_brem, E_brem, _hdr_brem = read_mapcube_primary(args.bremss)

    pion_E = _mapcube_to_counts_energy_grid(pion_cube, E_pion, Ectr)
    brem_E = _mapcube_to_counts_energy_grid(brem_cube, E_brem, Ectr)

    pion_on_grid = reproject_cube_to_target(src_cube=pion_E, w_src=w_pion, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx)
    brem_on_grid = reproject_cube_to_target(src_cube=brem_E, w_src=w_brem, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx)

    gas_dnde = pion_on_grid + brem_on_grid

    mu_gas = gas_dnde * expo * omega[None, :, :] * dE[:, None, None]

    out_dnde = os.path.join(args.outdir, "gas_dnde.fits")
    out_mu = os.path.join(args.outdir, "mu_gas_counts.fits")

    write_cube(out_dnde, gas_dnde, hdr_cnt, bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(out_mu, mu_gas, hdr_cnt, bunit="counts")

    print("✓ Wrote", out_dnde)
    print("✓ Wrote", out_mu)
    print("  mu_gas total:", float(np.nansum(mu_gas)))


if __name__ == "__main__":
    main()
