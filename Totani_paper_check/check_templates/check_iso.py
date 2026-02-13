#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS

from check_component import run_component_check
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)


def _read_ebounds_from_counts_ccube(hdul):
    if "EBOUNDS" not in hdul:
        raise RuntimeError("Counts CCUBE missing EBOUNDS extension")
    eb = hdul["EBOUNDS"].data
    Emin_kev = np.array(eb["E_MIN"], dtype=float)
    Emax_kev = np.array(eb["E_MAX"], dtype=float)
    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    return Emin_mev, Emax_mev, Ectr_mev, dE_mev


def main():
    repo_dir = os.environ.get("REPO_PATH")
    if repo_dir is None:
        raise SystemExit("REPO_PATH not set")

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    template = os.path.join(data_dir, "processed", "templates", "mu_iso_counts.fits")
    expo_path = os.path.join(data_dir, "processed", "expcube_1000to1000000.fits")
    plot_dir = os.path.join(os.path.dirname(__file__), "plots_check_iso")

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="Energy-bin index for isotropic intensity reconstruction")
    args = ap.parse_args()

    # Standard component plots
    run_component_check(
        label="ISO",
        template_path=template,
        counts_path=counts,
        plot_dir=plot_dir,
    )

    # Isotropic-specific check: reconstruct intensity for one bin
    with fits.open(counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        _, _, Ectr_mev, dE_mev = _read_ebounds_from_counts_ccube(hc)

    nE = int(Ectr_mev.size)
    if args.k is None:
        k0 = int(np.clip(nE // 2, 0, nE - 1))
    else:
        k0 = int(np.clip(args.k, 0, nE - 1))

    with fits.open(template) as ht:
        mu_iso = np.array(ht[0].data, dtype=float)

    with fits.open(expo_path) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(he)

    expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)
    omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=0.125)

    denom = expo[k0] * omega * dE_mev[k0]
    m = np.isfinite(mu_iso[k0]) & np.isfinite(denom) & (denom > 0)

    I_rec = np.full((ny, nx), np.nan, dtype=float)
    I_rec[m] = mu_iso[k0][m] / denom[m]

    vals = I_rec[m]
    med = float(np.nanmedian(vals)) if vals.size else float("nan")
    p16 = float(np.nanpercentile(vals, 16)) if vals.size else float("nan")
    p84 = float(np.nanpercentile(vals, 84)) if vals.size else float("nan")
    rel_scatter = float(np.nanstd(vals) / (med if med != 0 else 1.0)) if vals.size else float("nan")

    print("[ISO] intensity reconstruction")
    print(f"[ISO] k={k0:02d}  Ectr={Ectr_mev[k0]/1000.0:.6g} GeV")
    print(f"[ISO] I_rec median={med:.6e}  (16–84%: {p16:.6e}–{p84:.6e})  rel_std/median={rel_scatter:.4e}")

    # Save map
    os.makedirs(plot_dir, exist_ok=True)
    pos = I_rec[np.isfinite(I_rec) & (I_rec > 0)]
    norm = None
    if pos.size:
        vmin = float(np.nanpercentile(pos, 1))
        vmax = float(np.nanpercentile(pos, 99.9))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax > vmin:
            norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(I_rec, origin="lower", cmap="magma", norm=norm)
    ax.set_title(f"ISO I_rec  k={k0}  Ectr={Ectr_mev[k0]/1000.0:.3g} GeV")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.colorbar(im, ax=ax, label="I_rec")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"ISO_Irec_k{k0:02d}.png"), dpi=200)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
