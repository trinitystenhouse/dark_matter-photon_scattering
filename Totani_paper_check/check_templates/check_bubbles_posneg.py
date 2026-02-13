#!/usr/bin/env python3

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from astropy.io import fits
from astropy.wcs import WCS

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


def _stats(tag, x, m=None):
    z = np.asarray(x, dtype=float)
    if m is not None:
        z = z[m]
    z = z[np.isfinite(z)]
    if z.size == 0:
        print(f"[{tag}] EMPTY")
        return
    med = float(np.median(z))
    p16, p84 = np.percentile(z, [16, 84])
    mn = float(np.min(z))
    mx = float(np.max(z))
    rel = float(np.std(z) / med) if med != 0 else float("inf")
    print(f"[{tag}] min={mn:.6e} max={mx:.6e} median={med:.6e} 16-84={p16:.6e},{p84:.6e} rel_std/median={rel:.4e}")


def _save_map(out_png, img, title, *, cmap="magma", norm=None):
    fig = plt.figure(figsize=(6.8, 4.8))
    ax = fig.add_subplot(111)
    im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    repo_dir = os.environ.get("REPO_PATH")
    if repo_dir is None:
        raise SystemExit("REPO_PATH not set")

    data_dir = os.path.join(repo_dir, "fermi_data", "totani")
    templates_dir = os.path.join(data_dir, "processed", "templates")

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expo", default=os.path.join(data_dir, "processed", "expcube_1000to1000000.fits"))

    ap.add_argument("--pos-dnde", default=os.path.join(templates_dir, "bubbles_pos_dnde.fits"))
    ap.add_argument("--neg-dnde", default=os.path.join(templates_dir, "bubbles_neg_dnde.fits"))
    ap.add_argument("--pos-mu", default=os.path.join(templates_dir, "mu_bubbles_pos_counts.fits"))
    ap.add_argument("--neg-mu", default=os.path.join(templates_dir, "mu_bubbles_neg_counts.fits"))

    ap.add_argument("--ref-gev", type=float, default=4.3)
    ap.add_argument("--k", type=int, default=None, help="Override energy-bin index")

    ap.add_argument(
        "--plot-dir",
        default=os.path.join(os.path.dirname(__file__), "plots_check_bubbles_posneg"),
    )
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    args = ap.parse_args()

    for p in (args.counts, args.expo, args.pos_dnde, args.neg_dnde, args.pos_mu, args.neg_mu):
        if not os.path.exists(str(p)):
            raise SystemExit(f"File not found: {p}")

    with fits.open(args.counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        _, _, Ectr_mev, dE_mev = _read_ebounds_from_counts_ccube(hc)

    nE = int(Ectr_mev.size)
    Ectr_gev = Ectr_mev / 1000.0

    if args.k is None:
        k0 = int(np.argmin(np.abs(Ectr_gev - float(args.ref_gev))))
    else:
        k0 = int(np.clip(int(args.k), 0, nE - 1))

    with fits.open(args.expo) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo = read_expcube_energies_mev(he)
    expo = resample_exposure_logE_interp(expo_raw, E_expo, Ectr_mev)

    omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=float(args.binsz))

    pos_dnde = np.array(fits.getdata(args.pos_dnde), dtype=float)
    neg_dnde = np.array(fits.getdata(args.neg_dnde), dtype=float)
    pos_mu = np.array(fits.getdata(args.pos_mu), dtype=float)
    neg_mu = np.array(fits.getdata(args.neg_mu), dtype=float)

    for name, cube in (
        ("pos_dnde", pos_dnde),
        ("neg_dnde", neg_dnde),
        ("pos_mu", pos_mu),
        ("neg_mu", neg_mu),
    ):
        if cube.shape != (nE, ny, nx):
            raise SystemExit(f"{name} shape {cube.shape} != {(nE, ny, nx)}")

    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    roi2d = (np.abs(lon) <= float(args.roi_lon)) & (np.abs(lat) <= float(args.roi_lat))

    denom = expo[k0] * omega * dE_mev[k0]
    good = roi2d & np.isfinite(denom) & (denom > 0)

    Irec_pos = np.full((ny, nx), np.nan, dtype=float)
    Irec_neg = np.full((ny, nx), np.nan, dtype=float)
    Irec_pos[good] = pos_mu[k0][good] / denom[good]
    Irec_neg[good] = neg_mu[k0][good] / denom[good]

    # masks where templates are active
    mpos = good & np.isfinite(pos_dnde[k0]) & (pos_dnde[k0] > 0)
    mneg = good & np.isfinite(neg_dnde[k0]) & (neg_dnde[k0] > 0)

    print("== FB_POS/FB_NEG check ==")
    print(f"k={k0:02d}  Ectr={Ectr_gev[k0]:.6g} GeV")

    # Closure-ish diagnostics: Irec should equal dnde (up to float)
    rel_pos = np.nanmax(np.abs(Irec_pos[mpos] - pos_dnde[k0][mpos]) / np.maximum(pos_dnde[k0][mpos], 1e-300)) if np.any(mpos) else np.nan
    rel_neg = np.nanmax(np.abs(Irec_neg[mneg] - neg_dnde[k0][mneg]) / np.maximum(neg_dnde[k0][mneg], 1e-300)) if np.any(mneg) else np.nan
    print(f"[CONSIST] max rel |Irec_pos - pos_dnde| = {float(rel_pos):.3e}")
    print(f"[CONSIST] max rel |Irec_neg - neg_dnde| = {float(rel_neg):.3e}")

    _stats("pos_dnde (active)", pos_dnde[k0], mpos)
    _stats("neg_dnde (active)", neg_dnde[k0], mneg)

    pos_img = pos_dnde[k0].copy()
    neg_img = neg_dnde[k0].copy()
    comb_img = pos_dnde[k0] - neg_dnde[k0]

    pos_img[~good] = np.nan
    neg_img[~good] = np.nan
    comb_img[~good] = np.nan

    os.makedirs(args.plot_dir, exist_ok=True)

    # LogNorm for pos/neg
    for tag, img in [("pos", pos_img), ("neg", neg_img)]:
        z = img[np.isfinite(img) & (img > 0)]
        norm = None
        if z.size:
            vmin = float(np.nanpercentile(z, 1))
            vmax = float(np.nanpercentile(z, 99.9))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax > vmin:
                norm = LogNorm(vmin=vmin, vmax=vmax)
        _save_map(
            os.path.join(args.plot_dir, f"FB_{tag}_dnde_k{k0:02d}.png"),
            img,
            f"FB_{tag.upper()} dnde  k={k0}  Ectr={Ectr_gev[k0]:.3g} GeV",
            cmap="magma",
            norm=norm,
        )

    # Diverging norm for combined
    lim = float(np.nanpercentile(np.abs(comb_img[np.isfinite(comb_img)]), 99.0)) if np.any(np.isfinite(comb_img)) else 1.0
    lim = max(lim, 1e-30)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    _save_map(
        os.path.join(args.plot_dir, f"FB_pos_minus_neg_dnde_k{k0:02d}.png"),
        comb_img,
        f"FB_POS - FB_NEG (dnde)  k={k0}  Ectr={Ectr_gev[k0]:.3g} GeV",
        cmap="RdBu",
        norm=norm,
    )

    # Also plot reconstructed bubbles image flux from mu (pos-neg)
    comb_Irec = Irec_pos - Irec_neg
    comb_Irec[~good] = np.nan
    _save_map(
        os.path.join(args.plot_dir, f"FB_pos_minus_neg_Irec_k{k0:02d}.png"),
        comb_Irec,
        f"FB_POS - FB_NEG (I_rec)  k={k0}  Ectr={Ectr_gev[k0]:.3g} GeV",
        cmap="RdBu",
        norm=norm,
    )

    print("✓ wrote plots to:")
    print(" ", str(args.plot_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
