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
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    reproject_cube_to_target,
    resample_exposure_logE,
    write_cube,
)

DEG2RAD = np.pi / 180.0
REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

def interp_energy_planes(cube, E_src, E_tgt):
    """Interpolate (NE,ny,nx) cube from E_src->E_tgt in log(E). Energies in MeV."""
    if E_src is None:
        raise RuntimeError("No energy axis available to interpolate.")
    return resample_exposure_logE(cube, E_src, E_tgt)

def read_iem(iem_path):
    """Return (cube, WCS_celestial, energies_MeV, header)."""
    with fits.open(iem_path) as h:
        cube = h[0].data.astype(float)
        hdr  = h[0].header
        w    = WCS(hdr).celestial

        print("[E] IEM BUNIT:", str(hdr.get("BUNIT", "")))

        E = None
        if "ENERGIES" in h:
            tab = h["ENERGIES"].data
            col = tab.columns.names[0]
            E = np.array(tab[col], float)  # MeV
            print("[E] IEM ENERGIES (as stored):")
            print("[E]   N:", int(E.size))
            print("[E]   min/max:", float(np.nanmin(E)), float(np.nanmax(E)))
            print("[E]   first/last:", float(E[0]), float(E[-1]))
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E = np.sqrt(Emin * Emax)
            print("[E] IEM EBOUNDS (centers, as stored):")
            print("[E]   N:", int(E.size))
            print("[E]   min/max:", float(np.nanmin(E)), float(np.nanmax(E)))
            print("[E]   first/last:", float(E[0]), float(E[-1]))
        else:
            raise RuntimeError("IEM has no ENERGIES/EBOUNDS extension.")
    return cube, w, E, hdr

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
    ap.add_argument(
        "--iem-per-bin",
        action="store_true",
        help="Treat IEM cube values as per energy-bin intensity (divide by dE to get per-MeV). Use ONLY if you are sure.",
    )
    ap.add_argument(
        "--iem-assume-per-mev",
        action="store_true",
        help="Assume IEM cube is already per-MeV differential intensity even if BUNIT is blank (recommended).",
    )

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Target grid + energies from counts cube
    _, hdr_cnt, Emin, Emax, Ectr, _dE_unused = read_counts_and_ebounds(args.counts)
    nE = int(Ectr.size)

    w_tgt = WCS(hdr_cnt).celestial
    ny, nx = hdr_cnt["NAXIS2"], hdr_cnt["NAXIS1"]

    dE = (Emax - Emin)  # MeV

    print("[E] counts EBOUNDS -> using MeV internally")
    print("[E]   Ectr_MeV:", np.round(Ectr, 6))
    print("[E]   Ectr_GeV:", np.round(Ectr / 1000.0, 6))
    print("[E]   dE_MeV  :", np.round(dE, 6))

    omega = pixel_solid_angle_map(w_tgt, ny, nx, args.binsz)  # sr

    # Exposure (cm^2 s), interpolate in energy if needed
    expo, E_exp = read_exposure(args.expcube)
    print("[E] expcube planes (raw):", expo.shape)
    if expo.shape[1:] != (ny, nx):
        raise RuntimeError("Exposure spatial shape does not match counts grid.")
    if expo.shape[0] != nE:
        if E_exp is None:
            raise RuntimeError("Exposure has different #planes and no energy axis.")
        expo = interp_energy_planes(expo, E_exp, Ectr)

    print("[E] expcube planes (on counts E grid):", expo.shape)

    # IEM: read, interpolate in energy onto Ectr (MeV), then reproject to counts grid
    iem_cube, w_src, E_iem, hdr_iem = read_iem(args.iem)
    if iem_cube.shape[0] != len(E_iem):
        raise RuntimeError("IEM cube energy axis length mismatch.")

    iem_E = interp_energy_planes(iem_cube, E_iem, Ectr) if iem_cube.shape[0] != nE else iem_cube
    iem_on_grid = reproject_cube_to_target(src_cube=iem_E, w_src=w_src, w_tgt=w_tgt, ny_tgt=ny, nx_tgt=nx, wrap_lon=True)

    # Ensure IEM is in per-MeV units. If not, convert.
    # If BUNIT says it's per sr per cm2 per s but NOT per MeV, treat as per-energy-bin and divide by dE.
    bunit = str(hdr_iem.get("BUNIT", "")).upper()
    print("[E] IEM BUNIT (raw):", repr(hdr_iem.get("BUNIT", "")))

    # If BUNIT explicitly indicates per-MeV, do nothing.
    # If BUNIT is blank/unknown, DO NOT divide by dE unless user explicitly requests --iem-per-bin.
    iem_explicit_per_mev = ("MEV" in bunit) and ("-1" in bunit or "MEV-1" in bunit or "MEV^-1" in bunit)

    if args.iem_per_bin:
        print("[E] IEM: forcing per-bin -> dividing by dE to get per-MeV.")
        iem_on_grid = iem_on_grid / dE[:, None, None]
    else:
        # Recommended: assume per-MeV when BUNIT is blank/unknown
        if (not iem_explicit_per_mev) and (not args.iem_assume_per_mev):
            print("[W] IEM BUNIT does not confirm per-MeV and you did not pass --iem-assume-per-mev.")
            print("[W] Leaving IEM unchanged (NOT dividing by dE). If you know it's per-bin, rerun with --iem-per-bin.")
        else:
            print("[E] IEM: treating as per-MeV (no dE division).")

    # Isotropic: spectrum -> intensity at Ectr (MeV) using log-log interpolation
    E_iso, I_iso = read_isotropic_txt(args.iso)
    # log-log interpolation (power-law-ish spectra)
    m = (E_iso > 0) & (I_iso > 0)
    I_ctr = np.exp(np.interp(np.log(Ectr), np.log(E_iso[m]), np.log(I_iso[m])))
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

    k = 2  # ~3.8 GeV in your binning
    conv_k = expo[k] * omega * dE[k]

    print("Ectr(MeV) =", Ectr[k])
    print("I_ctr =", I_ctr[k], " [from iso file]")
    print("conv median =", np.nanmedian(conv_k))
    print("mu_iso median expected =", np.nanmedian(I_ctr[k] * conv_k))
    print("mu_iso sum expected =", np.nansum(I_ctr[k] * conv_k))

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
