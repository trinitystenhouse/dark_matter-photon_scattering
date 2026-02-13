#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
import re

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


def _print_region_stats(tag, Irec, m):
    vals = Irec[m]
    if vals.size == 0:
        print(f"[FB:{tag}] no pixels in mask")
        return
    med = float(np.median(vals))
    p16, p84 = np.percentile(vals, [16, 84])
    rel_std = float(np.std(vals) / med) if med != 0 else float("inf")
    print(f"[FB:{tag}] I_rec median {med:.6e}  16-84 {p16:.6e} {p84:.6e}  rel_std/median {rel_std:.4e}")


def _save_Irec_map(out_png, Irec, mask, title):
    img = np.array(Irec, dtype=float, copy=True)
    img[~mask] = np.nan
    pos = img[np.isfinite(img) & (img > 0)]
    norm = None
    if pos.size:
        vmin = float(np.nanpercentile(pos, 1))
        vmax = float(np.nanpercentile(pos, 99.9))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax > vmin:
            norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, origin="lower", cmap="magma", norm=norm)
    ax.set_title(title)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.colorbar(im, ax=ax, label="I_rec")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")
def main():
    counts = os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits")
    templates_dir = os.path.join(DATA_DIR, "processed", "templates")

    # Prefer the new binary-flat product, then fall back to older ones.
    candidates = [
        os.path.join(templates_dir, "mu_bubbles_flat_binary_counts.fits"),
        os.path.join(templates_dir, "mu_bubbles_vertices_sca_full_counts.fits"),
        os.path.join(templates_dir, "mu_bubbles_flat_counts.fits"),
    ]
    template = None
    for p in candidates:
        if os.path.exists(p):
            template = p
            break
    if template is None:
        raise SystemExit("No bubbles flat template found in processed/templates")

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="Energy-bin index for I_rec reconstruction")
    args = ap.parse_args()

    plot_dir = os.path.join(os.path.dirname(__file__), "plots_check_bubbles_flat")

    run_component_check(
        label="FB_FLAT",
        template_path=template,
        counts_path=counts,
        plot_dir=plot_dir,
    )

    # --- FB-specific intensity reconstruction, split North/South ---
    expo_path = os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits")

    # Prefer a mask derived from the mu filename: mu_<prefix>_counts.fits -> <prefix>_mask.fits
    mask_path = None
    m = re.match(r"^mu_(?P<prefix>.+)_counts\\.fits$", os.path.basename(template))
    if m is not None:
        prefix = m.group("prefix")
        cand = os.path.join(templates_dir, f"{prefix}_mask.fits")
        if os.path.exists(cand):
            mask_path = cand

    # Fall back to legacy mask naming based on template flavour
    if mask_path is None:
        if os.path.basename(template).startswith("mu_bubbles_vertices_sca"):
            mask_path = os.path.join(templates_dir, "bubbles_vertices_sca_full_mask.fits")
        else:
            mask_path = os.path.join(templates_dir, "bubbles_vertices_flat_full_mask.fits")

    if not os.path.exists(mask_path):
        raise SystemExit(f"Bubbles mask file not found: {mask_path}")

    with fits.open(counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        _, _, Ectr_mev, dE_mev = _read_ebounds_from_counts_ccube(hc)

    nE = int(Ectr_mev.size)
    k = int(np.clip(int(args.k), 0, nE - 1))

    with fits.open(template) as ht:
        mu_fb = np.array(ht[0].data, dtype=float)

    with fits.open(expo_path) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo_mev = read_expcube_energies_mev(he)
    expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)

    omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=0.125)

    fb_mask_2d = fits.getdata(mask_path).astype(bool)
    if fb_mask_2d.ndim == 3:
        fb_mask_2d = fb_mask_2d[0].astype(bool)
    if fb_mask_2d.shape != (ny, nx):
        raise SystemExit(f"Bubbles mask shape {fb_mask_2d.shape} != {(ny, nx)}")

    yy, xx = np.mgrid[:ny, :nx]
    lon_deg, lat_deg = wcs.pixel_to_world_values(xx, yy)
    lat_deg = np.asarray(lat_deg, dtype=float)

    den = expo[k] * omega * dE_mev[k]
    Irec = np.full((ny, nx), np.nan, dtype=float)
    ok = np.isfinite(mu_fb[k]) & np.isfinite(den) & (den > 0)
    Irec[ok] = mu_fb[k][ok] / den[ok]

    mN = fb_mask_2d & (lat_deg > 0) & np.isfinite(Irec) & (den > 0)
    mS = fb_mask_2d & (lat_deg < 0) & np.isfinite(Irec) & (den > 0)

    print("[FB] intensity reconstruction")
    print(f"[FB] k={k:02d}  Ectr={Ectr_mev[k]/1000.0:.6g} GeV")
    _print_region_stats("N", Irec, mN)
    _print_region_stats("S", Irec, mS)

    os.makedirs(plot_dir, exist_ok=True)
    _save_Irec_map(
        os.path.join(plot_dir, f"FB_Irec_N_k{k:02d}.png"),
        Irec,
        mN,
        f"FB I_rec North  k={k}  Ectr={Ectr_mev[k]/1000.0:.3g} GeV",
    )
    _save_Irec_map(
        os.path.join(plot_dir, f"FB_Irec_S_k{k:02d}.png"),
        Irec,
        mS,
        f"FB I_rec South  k={k}  Ectr={Ectr_mev[k]/1000.0:.3g} GeV",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
