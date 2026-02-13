#!/usr/bin/env python3

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import nnls

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")


def fit_bin_weighted_nnls(counts_2d, mu_components, mask_2d, eps=1.0):
    m = mask_2d.ravel()
    y = counts_2d.ravel()[m]
    X = np.vstack([mu.ravel()[m] for mu in mu_components]).T

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]
    if y.size == 0:
        return np.zeros(X.shape[1], dtype=float)

    w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
    yw = y * w
    Xw = X * w[:, None]

    A, _ = nnls(Xw, yw)
    return A


def load_mask_any_shape(mask_path, counts_shape):
    m = fits.getdata(mask_path).astype(bool)
    nE, ny, nx = counts_shape
    if m.shape == (nE, ny, nx):
        return m
    if m.shape == (ny, nx):
        return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
    raise RuntimeError(f"Mask shape {m.shape} not compatible with counts shape {(nE, ny, nx)}")


def _read_cube(path, expected_shape):
    with fits.open(path) as h:
        d = h[0].data.astype(float)
    if d.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {d.shape}, expected {expected_shape}")
    return d


def _write_primary_with_bunit(path, data, hdr_in, bunit):
    hdr = hdr_in.copy()
    hdr["BUNIT"] = str(bunit)
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(path, overwrite=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expcube", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))
    ap.add_argument("--templates-dir", default=os.path.join(DATA_DIR, "processed", "templates"))

    ap.add_argument("--outdir-data", default=os.path.join(DATA_DIR, "processed", "templates"))

    ap.add_argument("--ext-mask", required=False, help="extended-source mask FITS True=keep")
    ap.add_argument("--ps-mask", required=False, help="optional point-source mask FITS True=keep")
    ap.add_argument("--bubble-mask", required=False, help="optional FITS mask (ny,nx) selecting bubble region")

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--include-nfw", action="store_true")
    ap.add_argument("--include-loopi", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.outdir_data, exist_ok=True)

    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    expo_raw, E_expo = read_exposure(args.expcube)
    expo = resample_exposure_logE(expo_raw, E_expo, Ectr_mev)
    if expo.shape != counts.shape:
        raise RuntimeError("Exposure shape mismatch after resampling")

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    lon_w, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon_w) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    disk2d = np.abs(lat) >= args.disk_cut

    templates_dir = args.templates_dir

    mu_gas = _read_cube(os.path.join(templates_dir, "mu_gas_counts.fits"), counts.shape)
    mu_ics = _read_cube(os.path.join(templates_dir, "mu_ics_counts.fits"), counts.shape)
    mu_iso = _read_cube(os.path.join(templates_dir, "mu_iso_counts.fits"), counts.shape)
    mu_ps = _read_cube(os.path.join(templates_dir, "mu_ps_counts.fits"), counts.shape)

    mu_nfw = None
    if args.include_nfw:
        mu_nfw = _read_cube(os.path.join(templates_dir, "mu_nfw_counts.fits"), counts.shape)

    mu_loopi = None
    if args.include_loopi:
        loop_path = None
        for name in ("mu_loopI_counts.fits", "mu_loopi_counts.fits"):
            p = os.path.join(templates_dir, name)
            if os.path.exists(p):
                loop_path = p
                break
        if loop_path is None:
            raise RuntimeError("Requested Loop I but no mu_loopI_counts.fits found")
        mu_loopi = _read_cube(loop_path, counts.shape)

    if args.ext_mask:
        ext_keep3d = load_mask_any_shape(args.ext_mask, counts.shape)
    else:
        default_ext = os.path.join(templates_dir, "mask_extended_sources.fits")
        ext_keep3d = load_mask_any_shape(default_ext, counts.shape) if os.path.exists(default_ext) else np.ones_like(counts, bool)

    if args.ps_mask:
        ps_keep3d = load_mask_any_shape(args.ps_mask, counts.shape)
    else:
        ps_keep3d = np.ones_like(counts, bool)

    bubble2d = None
    if args.bubble_mask:
        bm = fits.getdata(args.bubble_mask).astype(bool)
        if bm.shape != (ny, nx):
            raise RuntimeError("bubble-mask must be (ny,nx)")
        bubble2d = bm

    resid_dnde = np.full_like(counts, np.nan, dtype=float)

    for k in range(nE):
        mask2d = roi2d & disk2d & ext_keep3d[k] & ps_keep3d[k]
        if bubble2d is not None:
            mask2d = mask2d & bubble2d

        denom = expo[k] * omega * dE_mev[k]
        good = mask2d & np.isfinite(denom) & (denom > 0) & np.isfinite(counts[k])

        comps = [mu_gas[k], mu_ics[k], mu_iso[k], mu_ps[k]]
        if mu_nfw is not None:
            comps.append(mu_nfw[k])
        if mu_loopi is not None:
            comps.append(mu_loopi[k])

        A = fit_bin_weighted_nnls(counts[k], comps, good, eps=1.0)

        model = np.zeros((ny, nx), dtype=float)
        for a, mu in zip(A, comps):
            model += a * mu

        resid_counts = counts[k] - model

        out = np.full((ny, nx), np.nan, dtype=float)
        out[good] = resid_counts[good] / denom[good]
        resid_dnde[k] = out

    pos_dnde = np.where(np.isfinite(resid_dnde) & (resid_dnde > 0), resid_dnde, 0.0)
    neg_dnde = np.where(np.isfinite(resid_dnde) & (resid_dnde < 0), -resid_dnde, 0.0)

    pos_E2 = pos_dnde * (Ectr_mev[:, None, None] ** 2)
    neg_E2 = neg_dnde * (Ectr_mev[:, None, None] ** 2)

    pos_mu = pos_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
    neg_mu = neg_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    print(pos_mu)
    print(neg_mu)

    out_pos_dnde = os.path.join(args.outdir_data, "bubbles_pos_dnde.fits")
    out_neg_dnde = os.path.join(args.outdir_data, "bubbles_neg_dnde.fits")
    out_pos_E2 = os.path.join(args.outdir_data, "bubbles_pos_E2dnde.fits")
    out_neg_E2 = os.path.join(args.outdir_data, "bubbles_neg_E2dnde.fits")
    out_pos_mu = os.path.join(args.outdir_data, "mu_bubbles_pos_counts.fits")
    out_neg_mu = os.path.join(args.outdir_data, "mu_bubbles_neg_counts.fits")

    _write_primary_with_bunit(out_pos_dnde, pos_dnde, hdr, "ph cm-2 s-1 sr-1 MeV-1")
    _write_primary_with_bunit(out_neg_dnde, neg_dnde, hdr, "ph cm-2 s-1 sr-1 MeV-1")
    _write_primary_with_bunit(out_pos_E2, pos_E2, hdr, "MeV cm-2 s-1 sr-1")
    _write_primary_with_bunit(out_neg_E2, neg_E2, hdr, "MeV cm-2 s-1 sr-1")
    _write_primary_with_bunit(out_pos_mu, pos_mu, hdr, "counts")
    _write_primary_with_bunit(out_neg_mu, neg_mu, hdr, "counts")


if __name__ == "__main__":
    main()
