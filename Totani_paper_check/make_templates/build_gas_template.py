#!/usr/bin/env python3

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import pixel_solid_angle_map, read_exposure, resample_exposure_logE

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def read_counts_bins(counts_ccube_path):
    with fits.open(counts_ccube_path) as h:
        hdr = h[0].header
        eb = h["EBOUNDS"].data
        emin = np.array(eb["E_MIN"], float)  # keV (for this dataset)
        emax = np.array(eb["E_MAX"], float)  # keV (for this dataset)
    ectr = np.sqrt(emin * emax)
    return hdr, emin, emax, ectr


def interp_energy_planes(cube, E_src, E_tgt):
    if E_src is None:
        raise RuntimeError("No energy axis available to interpolate.")

    return resample_exposure_logE(cube, E_src, E_tgt)


def reproject_plane_to_target(src_plane, w_src, w_tgt, ny_tgt, nx_tgt):
    yy, xx = np.mgrid[0:ny_tgt, 0:nx_tgt]
    lon, lat = w_tgt.pixel_to_world_values(xx, yy)
    xs, ys = w_src.world_to_pixel_values(lon, lat)

    ny, nx = src_plane.shape
    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)

    interp = RegularGridInterpolator(
        (y, x),
        src_plane,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.vstack([ys.ravel(), xs.ravel()]).T
    return interp(pts).reshape(ny_tgt, nx_tgt)


def reproject_cube_to_target(src_cube, w_src, w_tgt, ny_tgt, nx_tgt):
    out = np.empty((src_cube.shape[0], ny_tgt, nx_tgt), float)
    for k in range(src_cube.shape[0]):
        out[k] = reproject_plane_to_target(src_cube[k], w_src, w_tgt, ny_tgt, nx_tgt)
    return out


def read_mapcube(path):
    with fits.open(path) as h:
        data = h[0].data.astype(float)
        hdr = h[0].header
        w = WCS(hdr).celestial

        E = None
        if "ENERGIES" in h:
            tab = h["ENERGIES"].data
            col = tab.columns.names[0]
            E = np.array(tab[col], float)
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E = np.sqrt(Emin * Emax)

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D mapcube in primary HDU; got shape {data.shape}")

    if E is not None:
        if data.shape[0] == len(E):
            cube = data
        elif data.shape[-1] == len(E):
            cube = np.moveaxis(data, -1, 0)
        else:
            cube = data
    else:
        cube = data

    return cube, w, E


def write_cube(path, data, hdr_like, bunit=None):
    hdu = fits.PrimaryHDU(data.astype("f4"), header=hdr_like)
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)


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

    hdr_cnt, Emin_raw, Emax_raw, Ectr_raw = read_counts_bins(args.counts)

    if float(np.nanmedian(Emax_raw)) > 1e6:
        Emin = Emin_raw / 1000.0
        Emax = Emax_raw / 1000.0
        Ectr = Ectr_raw / 1000.0
    else:
        Emin = Emin_raw
        Emax = Emax_raw
        Ectr = Ectr_raw

    nE = len(Ectr)
    dE = (Emax - Emin)

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

    pion_cube, w_pion, E_pion = read_mapcube(args.pion)
    brem_cube, w_brem, E_brem = read_mapcube(args.bremss)

    pion_E = _mapcube_to_counts_energy_grid(pion_cube, E_pion, Ectr)
    brem_E = _mapcube_to_counts_energy_grid(brem_cube, E_brem, Ectr)

    pion_on_grid = reproject_cube_to_target(pion_E, w_pion, w_tgt, ny, nx)
    brem_on_grid = reproject_cube_to_target(brem_E, w_brem, w_tgt, ny, nx)

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
