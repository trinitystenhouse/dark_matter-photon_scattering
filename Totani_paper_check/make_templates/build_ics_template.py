#!/usr/bin/env python3

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def read_counts_bins(counts_ccube_path):
    with fits.open(counts_ccube_path) as h:
        hdr = h[0].header
        eb = h["EBOUNDS"].data
        emin = np.array(eb["E_MIN"], float)  # keV (for this dataset)
        emax = np.array(eb["E_MAX"], float)  # keV (for this dataset)
    ectr = np.sqrt(emin * emax)  # keV or MeV depending on file
    return hdr, emin, emax, ectr


def read_exposure(expcube_path):
    with fits.open(expcube_path) as h:
        expo = h[0].data.astype(float)
        E = None
        if "ENERGIES" in h:
            tab = h["ENERGIES"].data
            col = tab.columns.names[0]
            E = np.array(tab[col], float)  # MeV (as written by gtexpcube2)
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E = np.sqrt(Emin * Emax)
    return expo, E


def interp_energy_planes(cube, E_src, E_tgt):
    if E_src is None:
        raise RuntimeError("No energy axis available to interpolate.")

    if cube.shape[0] == len(E_tgt):
        return cube

    o = np.argsort(E_src)
    E_src = E_src[o]
    cube = cube[o].astype(float)

    logEs = np.log(E_src)
    logEt = np.log(E_tgt)

    ne, ny, nx = cube.shape
    flat = cube.reshape(ne, ny * nx)

    idx = np.searchsorted(logEs, logEt)
    idx = np.clip(idx, 1, ne - 1)
    i0 = idx - 1
    i1 = idx
    w = (logEt - logEs[i0]) / (logEs[i1] - logEs[i0])

    out = np.empty((len(E_tgt), ny * nx), float)
    for j in range(len(E_tgt)):
        out[j] = (1.0 - w[j]) * flat[i0[j]] + w[j] * flat[i1[j]]
    return out.reshape(len(E_tgt), ny, nx)


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


def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), dtype=float)


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

    # Try to ensure (nE, ny, nx)
    if E is not None:
        if data.shape[0] == len(E):
            cube = data
        elif data.shape[-1] == len(E):
            cube = np.moveaxis(data, -1, 0)
        else:
            cube = data
    else:
        cube = data

    return cube, w, E, hdr


def write_cube(path, data, hdr_like, bunit=None):
    hdu = fits.PrimaryHDU(data.astype("f4"), header=hdr_like)
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)


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

    hdr_cnt, Emin_raw, Emax_raw, Ectr_raw = read_counts_bins(args.counts)

    # Auto-detect keV vs MeV in counts EBOUNDS
    if float(np.nanmedian(Emax_raw)) > 1e6:
        Emin = Emin_raw / 1000.0
        Emax = Emax_raw / 1000.0
        Ectr = Ectr_raw / 1000.0
    else:
        Emin = Emin_raw
        Emax = Emax_raw
        Ectr = Ectr_raw

    nE = len(Ectr)
    dE = (Emax - Emin)  # MeV

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

    ics_cube, w_src, E_ics, _ = read_mapcube(args.ics)
    if E_ics is None:
        if ics_cube.shape[0] != nE:
            raise RuntimeError("ICS cube has no energies and does not match counts energy axis.")
        ics_E = ics_cube
    else:
        if ics_cube.shape[0] != len(E_ics):
            raise RuntimeError("ICS cube energy axis length mismatch.")
        ics_E = interp_energy_planes(ics_cube, E_ics, Ectr) if ics_cube.shape[0] != nE else ics_cube

    ics_on_grid = reproject_cube_to_target(ics_E, w_src, w_tgt, ny, nx)

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
