#!/usr/bin/env python3
"""
Build IEM + isotropic templates on the SAME grid/bins as counts CCUBE.

Outputs:
  - iem_dnde.fits        : dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - iso_dnde.fits        : dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - iem_E2dnde.fits      : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - iso_E2dnde.fits      : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
  - mu_iem_counts.fits   : expected counts per bin per pixel
  - mu_iso_counts.fits   : expected counts per bin per pixel
"""

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

DEG2RAD = np.pi / 180.0
REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

# ---------- helpers ----------
def read_counts_bins(counts_ccube_path):
    with fits.open(counts_ccube_path) as h:
        hdr = h[0].header
        eb  = h["EBOUNDS"].data
        emin = np.array(eb["E_MIN"], float)  # MeV
        emax = np.array(eb["E_MAX"], float)  # MeV
    ectr = np.sqrt(emin * emax)              # MeV
    return hdr, emin, emax, ectr

def read_exposure(expcube_path):
    with fits.open(expcube_path) as h:
        expo = h[0].data.astype(float)
        E = None
        if "ENERGIES" in h:
            tab = h["ENERGIES"].data
            col = tab.columns.names[0]
            E = np.array(tab[col], float)   # MeV (this is how gtexpcube2 writes it)
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E = np.sqrt(Emin * Emax)
    return expo, E

def interp_energy_planes(cube, E_src, E_tgt):
    """Interpolate (NE,ny,nx) cube from E_src->E_tgt in log(E). Energies in MeV."""
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

def read_iem(iem_path):
    """Return (cube, WCS_celestial, energies_MeV, header)."""
    with fits.open(iem_path) as h:
        cube = h[0].data.astype(float)
        hdr  = h[0].header
        w    = WCS(hdr).celestial

        E = None
        if "ENERGIES" in h:
            tab = h["ENERGIES"].data
            col = tab.columns.names[0]
            E = np.array(tab[col], float)  # MeV
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E = np.sqrt(Emin * Emax)
        else:
            raise RuntimeError("IEM has no ENERGIES/EBOUNDS extension.")
    return cube, w, E, hdr

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

def read_isotropic_txt(path):
    """
    Reads iso_* txt files: typically columns are E(MeV), ..., I (ph cm^-2 s^-1 sr^-1 MeV^-1)
    You previously used 3rd column; keep that.
    """
    E, I = [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                E.append(float(parts[0]))
                I.append(float(parts[2]))
            except ValueError:
                continue
    E = np.array(E, float)
    I = np.array(I, float)
    o = np.argsort(E)
    return E[o], I[o]

def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    """Match your PS template: Ω_pix ≈ Δl Δb cos(b) for CAR."""
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), float)

def write_cube(path, data, hdr_like, bunit=None):
    hdu = fits.PrimaryHDU(data.astype("f4"), header=hdr_like)
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)

# ---------- main ----------
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
        "--iem",
        default=os.path.join(DATA_DIR, "templates", "gll_iem_v07.fits"),
    )
    ap.add_argument(
        "--iso",
        default=os.path.join(DATA_DIR, "templates", "isotropic.txt"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Target grid + energies from counts cube (MeV)
    hdr_cnt, Emin, Emax, Ectr = read_counts_bins(args.counts)
    nE = len(Ectr)
    Ectr = Ectr / 1000.0  # Convert to MeV
    w_tgt = WCS(hdr_cnt).celestial
    ny, nx = hdr_cnt["NAXIS2"], hdr_cnt["NAXIS1"]
    dE = (Emax - Emin) /1000 # MeV
    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)  # sr

    # Exposure (cm^2 s), interpolate in energy if needed
    expo, E_exp = read_exposure(args.expcube)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")
    if expo.shape[0] != nE:
        if E_exp is None:
            raise RuntimeError("Exposure has different #planes and no energy axis.")
        expo = interp_energy_planes(expo, E_exp, Ectr)

    # IEM: read, interpolate in energy onto Ectr (MeV), then reproject to counts grid
    iem_cube, w_src, E_iem, hdr_iem = read_iem(args.iem)
    if iem_cube.shape[0] != len(E_iem):
        raise RuntimeError("IEM cube energy axis length mismatch.")

    iem_E = interp_energy_planes(iem_cube, E_iem, Ectr) if iem_cube.shape[0] != nE else iem_cube
    iem_on_grid = reproject_cube_to_target(iem_E, w_src, w_tgt, ny, nx)

    # Ensure IEM is in per-MeV units. If not, convert.
    # If BUNIT says it's per sr per cm2 per s but NOT per MeV, treat as per-energy-bin and divide by dE.
    bunit = str(hdr_iem.get("BUNIT", "")).upper()
    iem_is_per_mev = ("MEV" in bunit and ("-1" in bunit or "MEV-1" in bunit))
    if not iem_is_per_mev:
        # safest fallback: assume it is per-bin intensity and convert to per-MeV
        iem_on_grid = iem_on_grid / dE[:, None, None]

    # Isotropic: spectrum -> intensity at Ectr (MeV) using log-log interpolation
    E_iso, I_iso = read_isotropic_txt(args.iso)
    I_ctr = np.interp(np.log(Ectr), np.log(E_iso), I_iso)  # per MeV
    iso_on_grid = I_ctr[:, None, None] * np.ones((nE, ny, nx), float)

    # Now we have dN/dE cubes in the SAME units as your PS dnde template:
    # [ph cm^-2 s^-1 sr^-1 MeV^-1]
    iem_dnde = iem_on_grid
    iso_dnde = iso_on_grid

    # Optional Totani plotting quantity
    iem_E2dnde = iem_dnde * (Ectr[:, None, None] ** 2)   # [MeV cm^-2 s^-1 sr^-1]
    iso_E2dnde = iso_dnde * (Ectr[:, None, None] ** 2)

    # Expected counts per bin:
    # mu = (dN/dE) * exposure * Ω_pix * ΔE
    mu_iem = iem_dnde * expo * omega[None, :, :] * dE[:, None, None]
    mu_iso = iso_dnde * expo * omega[None, :, :] * dE[:, None, None]

    # Write
    write_cube(os.path.join(args.outdir, "iem_dnde.fits"), iem_dnde, hdr_cnt,
               bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(os.path.join(args.outdir, "iso_dnde.fits"), iso_dnde, hdr_cnt,
               bunit="ph cm-2 s-1 sr-1 MeV-1")
    write_cube(os.path.join(args.outdir, "iem_E2dnde.fits"), iem_E2dnde, hdr_cnt,
               bunit="MeV cm-2 s-1 sr-1")
    write_cube(os.path.join(args.outdir, "iso_E2dnde.fits"), iso_E2dnde, hdr_cnt,
               bunit="MeV cm-2 s-1 sr-1")
    write_cube(os.path.join(args.outdir, "mu_iem_counts.fits"), mu_iem, hdr_cnt,
               bunit="counts")
    write_cube(os.path.join(args.outdir, "mu_iso_counts.fits"), mu_iso, hdr_cnt,
               bunit="counts")

    # Debug energies (MeV)
    with open(os.path.join(args.outdir, "debug_energy.txt"), "w") as f:
        f.write("# k Emin(MeV) Emax(MeV) Ectr(MeV) dE(MeV)\n")
        for k in range(nE):
            f.write(f"{k:02d} {Emin[k]:.6f} {Emax[k]:.6f} {Ectr[k]:.6f} {dE[k]:.6f}\n")

    print("✓ Wrote templates to", args.outdir)
    print("  mu_iem total:", float(np.nansum(mu_iem)))
    print("  mu_iso total:", float(np.nansum(mu_iso)))

if __name__ == "__main__":
    main()
