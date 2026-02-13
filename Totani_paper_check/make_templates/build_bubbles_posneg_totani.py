#!/usr/bin/env python3

import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import nnls

from totani_helpers.totani_io import (
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
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

    ap.add_argument(
        "--ref-gev",
        type=float,
        default=4.3,
        help="Reference energy (GeV) used to construct FB_POS/FB_NEG from the bubbles image",
    )

    ap.add_argument("--include-nfw", action="store_true")
    ap.add_argument("--include-loopi", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.outdir_data, exist_ok=True)

    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    with fits.open(args.expcube) as he:
        expo_raw = np.array(he[0].data, dtype=np.float64)
        E_expo = read_expcube_energies_mev(he)
    expo = resample_exposure_logE_interp(expo_raw, E_expo, Ectr_mev)
    if expo.shape != counts.shape:
        raise RuntimeError("Exposure shape mismatch after resampling")

    omega = pixel_solid_angle_map(wcs, ny, nx, binsz_deg=float(args.binsz))
    lon_w, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon_w) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)
    disk2d = np.abs(lat) >= args.disk_cut

    templates_dir = args.templates_dir

    mu_gas = _read_cube(os.path.join(templates_dir, "mu_gas_counts.fits"), counts.shape)
    mu_ics = _read_cube(os.path.join(templates_dir, "mu_ics_counts.fits"), counts.shape)
    mu_iso = _read_cube(os.path.join(templates_dir, "mu_iso_counts.fits"), counts.shape)
    mu_ps = _read_cube(os.path.join(templates_dir, "mu_ps_counts.fits"), counts.shape)

    # Flat bubbles template used in the baseline fit to build the bubbles image.
    mu_flat = None
    for name in (
        "mu_bubbles_flat_binary_counts.fits",
        "mu_bubbles_vertices_sca_full_counts.fits",
        "mu_bubbles_flat_counts.fits",
    ):
        p = os.path.join(templates_dir, name)
        if os.path.exists(p):
            mu_flat = _read_cube(p, counts.shape)
            break
    if mu_flat is None:
        raise RuntimeError("No flat bubbles mu template found (need mu_bubbles_flat_binary_counts.fits or legacy)")

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

    # ---- Totani-style construction from a single reference energy bin ----
    Ectr_gev = Ectr_mev / 1000.0
    k_ref = int(np.argmin(np.abs(Ectr_gev - float(args.ref_gev))))
    print(f"[FB_POSNEG] using reference bin k_ref={k_ref} (Ectr={Ectr_gev[k_ref]:.6g} GeV)")

    mask2d = roi2d & disk2d & ext_keep3d[k_ref] & ps_keep3d[k_ref]
    if bubble2d is not None:
        mask2d = mask2d & bubble2d

    denom = expo[k_ref] * omega * dE_mev[k_ref]
    good = mask2d & np.isfinite(denom) & (denom > 0) & np.isfinite(counts[k_ref])

    # Baseline components including FLAT bubbles.
    comps = [mu_gas[k_ref], mu_ics[k_ref], mu_iso[k_ref], mu_ps[k_ref]]
    if mu_nfw is not None:
        comps.append(mu_nfw[k_ref])
    if mu_loopi is not None:
        comps.append(mu_loopi[k_ref])
    comps.append(mu_flat[k_ref])
    i_flat = len(comps) - 1

    A = fit_bin_weighted_nnls(counts[k_ref], comps, good, eps=1.0)

    model = np.zeros((ny, nx), dtype=float)
    for a, mu in zip(A, comps):
        model += a * mu

    resid_counts = counts[k_ref] - model

    # Bubbles image counts = (best-fit flat bubbles counts) + residual counts
    bubbles_img_counts = (A[i_flat] * mu_flat[k_ref]) + resid_counts

    bubbles_img_flux = np.full((ny, nx), np.nan, dtype=float)
    bubbles_img_flux[good] = bubbles_img_counts[good] / denom[good]

    pos_flux = np.where(np.isfinite(bubbles_img_flux) & (bubbles_img_flux > 0), bubbles_img_flux, 0.0)
    neg_flux = np.where(np.isfinite(bubbles_img_flux) & (bubbles_img_flux < 0), -bubbles_img_flux, 0.0)

    # Energy-independent spatial templates (normalised) derived from the reference map.
    spatial_pos = np.zeros((ny, nx), dtype=float)
    spatial_neg = np.zeros((ny, nx), dtype=float)
    spatial_pos[good] = pos_flux[good]
    spatial_neg[good] = neg_flux[good]

    s_pos = float(np.nansum(spatial_pos[good]))
    s_neg = float(np.nansum(spatial_neg[good]))
    if not np.isfinite(s_pos) or s_pos <= 0:
        raise RuntimeError("FB_POS normalisation failed (no positive pixels in bubbles image)")
    if not np.isfinite(s_neg) or s_neg <= 0:
        raise RuntimeError("FB_NEG normalisation failed (no negative pixels in bubbles image)")
    spatial_pos /= s_pos
    spatial_neg /= s_neg

    # Build dnde/mu products for all energies using standard form.
    pos_dnde = np.empty((nE, ny, nx), dtype=float)
    neg_dnde = np.empty((nE, ny, nx), dtype=float)
    for k in range(nE):
        pos_dnde[k] = spatial_pos / (omega * dE_mev[k])
        neg_dnde[k] = spatial_neg / (omega * dE_mev[k])

    pos_E2 = pos_dnde * (Ectr_mev[:, None, None] ** 2)
    neg_E2 = neg_dnde * (Ectr_mev[:, None, None] ** 2)

    pos_mu = pos_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
    neg_mu = neg_dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

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
