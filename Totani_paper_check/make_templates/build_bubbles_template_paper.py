#!/usr/bin/env python3
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.path import Path
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import nnls

def smooth_nan_2d(img, *, sigma_pix):
    x = np.asarray(img, dtype=float)
    good = np.isfinite(x)
    if not np.any(good):
        return np.full_like(x, np.nan, dtype=float)

    x0 = np.zeros_like(x, dtype=float)
    x0[good] = x[good]
    w = np.zeros_like(x, dtype=float)
    w[good] = 1.0

    xs = gaussian_filter(x0, float(sigma_pix), mode="constant", cval=0.0)
    ws = gaussian_filter(w, float(sigma_pix), mode="constant", cval=0.0)

    out = np.full_like(x, np.nan, dtype=float)
    m = ws > 0
    out[m] = xs[m] / ws[m]
    return out



def fit_bin_weighted_nnls(
    counts_2d,
    mu_components,
    mask_2d,
    eps=1.0,
    *,
    normalize_components=False,
    weights_mode="poisson",
):
    """Weighted NNLS on a single energy bin."""
    m = mask_2d.ravel()
    y = counts_2d.ravel()[m]

    X = np.vstack([mu.ravel()[m] for mu in mu_components]).T

    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]
    if y.size == 0:
        return np.zeros(X.shape[1], dtype=float)

    # Optional: normalize each component column to comparable scale on this mask.
    # This prevents small columns (e.g. isotropic) from being ignored due to conditioning.
    col_scale = np.ones(X.shape[1], dtype=float)
    if normalize_components:
        for i in range(X.shape[1]):
            s = float(np.nansum(np.abs(X[:, i])))
            if np.isfinite(s) and s > 0:
                col_scale[i] = s
        X = X / col_scale[None, :]

    # Weights
    if weights_mode == "uniform":
        w = np.ones_like(y, dtype=float)
    elif weights_mode == "poisson":
        # Poisson-ish weights; stabilize
        w = 1.0 / np.sqrt(np.maximum(y, 0.0) + eps)
    else:
        raise ValueError(f"Unknown weights_mode='{weights_mode}'")
    yw = y * w
    Xw = X * w[:, None]

    A, _ = nnls(Xw, yw)

    if normalize_components:
        A = A / col_scale
    return A


def plot_subtraction_diagnostics(
    *,
    out_png,
    wcs,
    roi2d,
    lon_w,
    lat,
    title_prefix,
    counts_k,
    expo_k,
    omega,
    dE_mev_k,
    components_scaled,   # dict: name -> 2D map in counts space
    model_counts,
    resid_counts,
    binsz_deg,
    resid_plot_smooth_deg=1.0,
):
    denom = expo_k * omega * dE_mev_k  # cm^2 s sr MeV
    resid_flux = np.full_like(resid_counts, np.nan, dtype=float)
    good = roi2d & np.isfinite(denom) & (denom > 0)
    resid_flux[good] = resid_counts[good] / denom[good]

    # Mask outside ROI
    def _mask(x):
        y = np.array(x, dtype=float, copy=True)
        y[~roi2d] = np.nan
        return y

    # Robust percentile helpers (ignore NaNs)
    def _finite_roi(x):
        z = x[roi2d]
        z = z[np.isfinite(z)]
        return z

    def _log_limits(x, p_lo=1.0, p_hi=99.5, floor=1e-6):
        z = _finite_roi(x)
        if z.size == 0:
            return (floor, 1.0)
        # LogNorm requires positive values; clip at floor
        zpos = z[z > 0]
        if zpos.size == 0:
            return (floor, 1.0)
        vmin = max(np.percentile(zpos, p_lo), floor)
        vmax = max(np.percentile(zpos, p_hi), vmin * 10.0)
        return (vmin, vmax)

    def _sym_limits(x, p=99.5, floor=1e-6):
        z = _finite_roi(x)
        if z.size == 0:
            return floor
        lim = np.percentile(np.abs(z), p)
        return max(lim, floor)

    # Build panels with a "kind" tag to control scaling
    panels = [("Data (counts)", _mask(counts_k), "counts")]
    for name, m in components_scaled.items():
        panels.append((f"{name} (scaled, counts)", _mask(m), "counts"))

    sigma_pix = float(resid_plot_smooth_deg) / float(binsz_deg)
    resid_counts_sm = smooth_nan_2d(_mask(resid_counts), sigma_pix=sigma_pix)
    resid_flux_sm = smooth_nan_2d(_mask(resid_flux), sigma_pix=sigma_pix)

    panels += [
        ("Total model to subtract (counts)", _mask(model_counts), "counts"),
        (f"Residual (counts), σ={float(resid_plot_smooth_deg):.1f}°", resid_counts_sm, "resid"),
        (f"Residual flux = resid/(expo*Ω*ΔE), σ={float(resid_plot_smooth_deg):.1f}°", resid_flux_sm, "resid"),
    ]

    n = len(panels)
    ncol = 2
    nrow = (n + ncol - 1) // ncol

    plt.figure(figsize=(12, 3.4 * nrow))
    for i, (ttl, img, kind) in enumerate(panels, 1):
        ax = plt.subplot(nrow, ncol, i, projection=wcs)

        if kind == "counts":
            # Use LogNorm so plane/bright pixels don't black-out everything else
            vmin, vmax = _log_limits(img, p_lo=1.0, p_hi=99.5, floor=1e-3)
            # Replace non-positive with NaN so LogNorm behaves
            img_show = np.array(img, copy=True)
            img_show[~np.isfinite(img_show)] = np.nan
            img_show[img_show <= 0] = np.nan
            im = ax.imshow(img_show, origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax))
            cb_label = "counts (log scale)"
        else:
            # Residuals: diverging, centered on 0, symmetric percentiles
            lim = _sym_limits(img, p=99.5, floor=1e-12)
            im = ax.imshow(img, origin="lower",
                           norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
            cb_label = "residual (signed)"

        ax.set_title(ttl)
        ax.set_xlabel("l")
        ax.set_ylabel("b")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cb_label)

    plt.suptitle(title_prefix, y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_subtraction_diagnostics_masked(
    *,
    out_png,
    wcs,
    good_mask2d,
    title_prefix,
    counts_k,
    expo_k,
    omega,
    dE_mev_k,
    components_scaled,
    model_counts,
    resid_counts,
    binsz_deg,
    resid_plot_smooth_deg=1.0,
):
    denom = expo_k * omega * dE_mev_k  # cm^2 s sr MeV
    resid_flux = np.full_like(resid_counts, np.nan, dtype=float)
    good = good_mask2d & np.isfinite(denom) & (denom > 0)
    resid_flux[good] = resid_counts[good] / denom[good]

    def _mask(x):
        y = np.array(x, dtype=float, copy=True)
        y[~good_mask2d] = np.nan
        return y

    def _finite_mask(x):
        z = x[good_mask2d]
        z = z[np.isfinite(z)]
        return z

    def _log_limits(x, p_lo=1.0, p_hi=99.5, floor=1e-6):
        z = _finite_mask(x)
        if z.size == 0:
            return (floor, 1.0)
        zpos = z[z > 0]
        if zpos.size == 0:
            return (floor, 1.0)
        vmin = max(np.percentile(zpos, p_lo), floor)
        vmax = max(np.percentile(zpos, p_hi), vmin * 10.0)
        return (vmin, vmax)

    def _sym_limits(x, p=99.5, floor=1e-12):
        z = _finite_mask(x)
        if z.size == 0:
            return floor
        lim = np.percentile(np.abs(z), p)
        return max(float(lim), floor)

    panels = [("Data (counts)", _mask(counts_k), "counts")]
    for name, m in components_scaled.items():
        panels.append((f"{name} (scaled, counts)", _mask(m), "counts"))

    sigma_pix = float(resid_plot_smooth_deg) / float(binsz_deg)
    resid_counts_sm = smooth_nan_2d(_mask(resid_counts), sigma_pix=sigma_pix)
    resid_flux_sm = smooth_nan_2d(_mask(resid_flux), sigma_pix=sigma_pix)

    panels += [
        ("Total model to subtract (counts)", _mask(model_counts), "counts"),
        (f"Residual (counts), σ={float(resid_plot_smooth_deg):.1f}°", resid_counts_sm, "resid"),
        (f"Residual flux = resid/(expo*Ω*ΔE), σ={float(resid_plot_smooth_deg):.1f}°", resid_flux_sm, "resid"),
    ]

    n = len(panels)
    ncol = 2
    nrow = (n + ncol - 1) // ncol

    plt.figure(figsize=(12, 3.4 * nrow))
    for i, (ttl, img, kind) in enumerate(panels, 1):
        ax = plt.subplot(nrow, ncol, i, projection=wcs)
        if kind == "counts":
            vmin, vmax = _log_limits(img, p_lo=1.0, p_hi=99.5, floor=1e-3)
            img_show = np.array(img, copy=True)
            img_show[~np.isfinite(img_show)] = np.nan
            img_show[img_show <= 0] = np.nan
            im = ax.imshow(img_show, origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax))
            cb_label = "counts (log scale)"
        else:
            lim = _sym_limits(img, p=99.5, floor=1e-12)
            im = ax.imshow(img, origin="lower", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
            cb_label = "residual (signed)"

        ax.set_title(ttl)
        ax.set_xlabel("l")
        ax.set_ylabel("b")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cb_label)

    plt.suptitle(title_prefix, y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _fit_two_powerlaws_weighted(y, sigma, f1, f2):
    """
    Solve y(E) ≈ H*f1(E) + S*f2(E) per pixel with weights 1/sigma^2.

    y: (nE, nPix)
    sigma: (nE, nPix)
    f1,f2: (nE,)
    """
    w = 1.0 / np.maximum(sigma, 1e-30) ** 2

    a11 = np.sum(w * (f1[:, None] ** 2), axis=0)
    a12 = np.sum(w * (f1[:, None] * f2[:, None]), axis=0)
    a22 = np.sum(w * (f2[:, None] ** 2), axis=0)
    b1 = np.sum(w * (f1[:, None] * y), axis=0)
    b2 = np.sum(w * (f2[:, None] * y), axis=0)

    det = a11 * a22 - a12 * a12
    good = det > 0

    H = np.full(det.shape, np.nan, dtype=float)
    S = np.full(det.shape, np.nan, dtype=float)
    sigma_H = np.full(det.shape, np.nan, dtype=float)

    H[good] = (b1[good] * a22[good] - b2[good] * a12[good]) / det[good]
    S[good] = (a11[good] * b2[good] - a12[good] * b1[good]) / det[good]

    # Var(H) = a22/det for (X^T W X)^-1
    sigma_H[good] = np.sqrt(a22[good] / det[good])
    return H, S, sigma_H


def _read_vertices_lonlat(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            pts.append((lon, lat))
    if len(pts) < 3:
        raise RuntimeError(f"Need >=3 vertices in {path}")
    return np.asarray(pts, dtype=float)


def _polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_lonlat_deg):
    lon = np.mod(verts_lonlat_deg[:, 0], 360.0)
    lat = verts_lonlat_deg[:, 1]
    x, y = wcs.world_to_pixel_values(lon, lat)
    poly = Path(np.vstack([x, y]).T)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    return poly.contains_points(pts).reshape(ny, nx)


def _write_primary_with_bunit(path, data, hdr_in, bunit):
    hdr = hdr_in.copy()
    hdr["BUNIT"] = str(bunit)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _write_template_products(*, outdir, prefix, wcs, roi2d, mask2d, expo, omega, dE_mev, Ectr_mev):
    nE, ny, nx = expo.shape
    if mask2d.shape != (ny, nx):
        raise RuntimeError("mask2d shape mismatch")

    m = np.asarray(mask2d, dtype=bool) & np.asarray(roi2d, dtype=bool)
    n_in = int(np.sum(m))
    if n_in <= 0:
        raise RuntimeError(f"Empty template mask within ROI for {prefix} (nmask_in_roi=0)")

    T = np.zeros((ny, nx), dtype=float)
    T[m] = 1.0
    T[~roi2d] = 0.0

    s = float(np.nansum(T[m]))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Template normalization failed (sum<=0) for {prefix} (sum={s}, nmask_in_roi={n_in})")
    T = T / s

    dnde = np.empty((nE, ny, nx), dtype=float)
    for k in range(nE):
        dnde[k] = T / (omega * dE_mev[k])

    e2dnde = dnde * (Ectr_mev[:, None, None] ** 2)
    mu_counts = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    hdr2 = wcs.to_header()
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_mask.fits"), mask2d.astype(np.int16), hdr2, "dimensionless")
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_template.fits"), T.astype(np.float32), hdr2, "dimensionless")
    _write_primary_with_bunit(os.path.join(outdir, f"mu_{prefix}_counts.fits"), mu_counts.astype(np.float32), hdr2, "counts")
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_dnde.fits"), dnde.astype(np.float32), hdr2, "ph cm-2 s-1 sr-1 MeV-1")
    _write_primary_with_bunit(os.path.join(outdir, f"{prefix}_E2dnde.fits"), e2dnde.astype(np.float32), hdr2, "MeV cm-2 s-1 sr-1")


def _read_counts_and_ebounds(counts_path):
    with fits.open(counts_path) as h:
        counts = h[0].data.astype(float)
        hdr = h[0].header
        eb = h["EBOUNDS"].data

    Emin = eb["E_MIN"].astype(float) / 1000.0
    Emax = eb["E_MAX"].astype(float) / 1000.0
    Ectr = np.sqrt(Emin * Emax)
    dE = (Emax - Emin)
    return counts, hdr, Emin, Emax, Ectr, dE


def _read_exposure(expo_path):
    with fits.open(expo_path) as h:
        expo = h[0].data.astype(float)
        E_expo = None
        if "ENERGIES" in h:
            col0 = h["ENERGIES"].columns.names[0]
            E_expo = np.array(h["ENERGIES"].data[col0], dtype=float)
    return expo, E_expo


def _resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    if expo_raw.shape[0] == len(E_tgt_mev):
        return expo_raw
    if E_expo_mev is None:
        raise RuntimeError("Exposure planes != counts planes and EXPO has no ENERGIES table.")

    order = np.argsort(E_expo_mev)
    E_expo_mev = E_expo_mev[order]
    expo_raw = expo_raw[order]

    logEs = np.log(E_expo_mev)
    logEt = np.log(E_tgt_mev)

    ne, ny, nx = expo_raw.shape
    flat = expo_raw.reshape(ne, ny * nx)

    idx = np.searchsorted(logEs, logEt)
    idx = np.clip(idx, 1, ne - 1)
    i0 = idx - 1
    i1 = idx
    w = (logEt - logEs[i0]) / (logEs[i1] - logEs[i0])

    out = np.empty((len(E_tgt_mev), ny * nx), float)
    for j in range(len(E_tgt_mev)):
        out[j] = (1 - w[j]) * flat[i0[j]] + w[j] * flat[i1[j]]
    return out.reshape(len(E_tgt_mev), ny, nx)


def _pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), float)


def _lonlat_grids(wcs, ny, nx):
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def _roi_box_mask(lon, lat, roi_lon, roi_lat):
    return (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)


def _load_mask_any_shape(mask_path, counts_shape):
    m = fits.getdata(mask_path).astype(bool)
    nE, ny, nx = counts_shape
    if m.shape == (nE, ny, nx):
        return m
    if m.shape == (ny, nx):
        return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
    raise RuntimeError(f"Mask shape {m.shape} not compatible with counts shape {(nE, ny, nx)}")


def _stage_load_inputs(args):
    repo_dir = os.environ["REPO_PATH"]
    data_dir = os.path.join(repo_dir, "fermi_data", "totani")

    counts_path = args.counts or os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits")
    expo_path = args.expo or os.path.join(data_dir, "processed", "expcube_1000to1000000.fits")
    templates_dir = args.templates_dir or os.path.join(data_dir, "processed", "templates")

    os.makedirs(args.outdir, exist_ok=True)

    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = _read_counts_and_ebounds(counts_path)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    expo_raw, E_expo = _read_exposure(expo_path)
    expo = _resample_exposure_logE(expo_raw, E_expo, Ectr_mev)
    if expo.shape != (nE, ny, nx):
        raise RuntimeError("Exposure shape mismatch after resampling")

    omega = _pixel_solid_angle_map(wcs, ny, nx, args.binsz)
    lon, lat = _lonlat_grids(wcs, ny, nx)
    roi2d = _roi_box_mask(lon, lat, args.roi_lon, args.roi_lat)

    return {
        "data_dir": data_dir,
        "templates_dir": templates_dir,
        "counts": counts,
        "expo": expo,
        "hdr": hdr,
        "wcs": wcs,
        "Ectr_mev": Ectr_mev,
        "dE_mev": dE_mev,
        "omega": omega,
        "lon": lon,
        "lat": lat,
        "roi2d": roi2d,
    }


def _stage_apply_keep_masks(*, args, counts, expo, roi2d, templates_dir):
    if args.ext_mask:
        ext_keep3d = _load_mask_any_shape(args.ext_mask, counts.shape)
    else:
        default_ext = os.path.join(templates_dir, "mask_extended_sources.fits")
        ext_keep3d = _load_mask_any_shape(default_ext, counts.shape) if os.path.exists(default_ext) else np.ones_like(counts, bool)

    if args.ps_mask:
        ps_keep3d = _load_mask_any_shape(args.ps_mask, counts.shape)
    else:
        ps_keep3d = np.ones_like(counts, bool)

    mask_all = (ext_keep3d & ps_keep3d) & roi2d[None, :, :]
    counts_m = np.array(counts, copy=True)
    expo_m = np.array(expo, copy=True)
    counts_m[~mask_all] = np.nan
    expo_m[~mask_all] = np.nan
    return counts_m, expo_m, mask_all


def _stage_load_mu_templates(*, templates_dir, mask_all, counts_shape):
    with fits.open(os.path.join(templates_dir, "mu_gas_counts.fits")) as h:
        mu_gas = h[0].data.astype(float)
    with fits.open(os.path.join(templates_dir, "mu_ps_counts.fits")) as h:
        mu_ps = h[0].data.astype(float)

    if mu_gas.shape != counts_shape or mu_ps.shape != counts_shape:
        raise RuntimeError("mu_gas/mu_ps shape mismatch")

    mu_gas_m = np.array(mu_gas, copy=True)
    mu_ps_m = np.array(mu_ps, copy=True)
    mu_gas_m[~mask_all] = np.nan
    mu_ps_m[~mask_all] = np.nan
    return mu_gas_m, mu_ps_m


def _stage_optional_user_polygon(*, args, wcs, ny, nx, connected_mask):
    if (args.user_mask_north_verts is None) and (args.user_mask_south_verts is None):
        return connected_mask

    user_mask = np.zeros((ny, nx), dtype=bool)
    if args.user_mask_north_verts is not None:
        vN = _read_vertices_lonlat(args.user_mask_north_verts)
        user_mask |= _polygon_mask_from_lonlat_vertices(wcs, ny, nx, vN)
    if args.user_mask_south_verts is not None:
        vS = _read_vertices_lonlat(args.user_mask_south_verts)
        user_mask |= _polygon_mask_from_lonlat_vertices(wcs, ny, nx, vS)
    return user_mask


def _stage_write_products(*, args, wcs, roi2d, lat, connected_mask, expo, omega, dE_mev, Ectr_mev):
    ny, nx = roi2d.shape
    full_mask = connected_mask & roi2d
    hilat = full_mask & (np.abs(lat) >= args.disk_cut)
    lolat = full_mask & (np.abs(lat) < args.disk_cut)

    def _maybe_write(mask2d, suffix):
        n = int(np.sum(mask2d & roi2d))
        if n <= 0:
            print(f"[products] Skip {suffix}: empty mask within ROI")
            return
        _write_template_products(
            outdir=args.outdir,
            prefix=f"{args.products_prefix}_{suffix}",
            wcs=wcs,
            roi2d=roi2d,
            mask2d=mask2d,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
        )

    _maybe_write(full_mask, "full")
    _maybe_write(hilat, "hilat")
    _maybe_write(lolat, "lolat")
    return full_mask


def _stage_write_debug_maps(*, args, wcs, sig_sm, Hmap, Smap):
    h2 = wcs.to_header()
    fits.writeto(os.path.join(args.outdir, "debug_sca_hard_sig_smoothed.fits"), sig_sm.astype(np.float32), header=h2, overwrite=True)
    fits.writeto(os.path.join(args.outdir, "debug_sca_hard_map.fits"), Hmap.astype(np.float32), header=h2, overwrite=True)
    fits.writeto(os.path.join(args.outdir, "debug_sca_soft_map.fits"), Smap.astype(np.float32), header=h2, overwrite=True)


def _stage_plot_sca_summary(*, args, wcs, roi2d, sig_sm, Hmap, Smap, full_mask):
    def _imshow(ax, img, ttl, lim):
        im = ax.imshow(img, origin="lower", cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
        ax.set_title(ttl)
        ax.set_xlabel("l")
        ax.set_ylabel("b")
        return im

    plt.figure(figsize=(12.0, 8.0))
    ax1 = plt.subplot(2, 2, 1, projection=wcs)
    limH = float(np.nanpercentile(np.abs(Hmap[roi2d & np.isfinite(Hmap)]), 99.0)) if np.any(roi2d & np.isfinite(Hmap)) else 1.0
    im1 = _imshow(ax1, np.where(roi2d, Hmap, np.nan), "Hard map H", max(limH, 1e-12))
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2, projection=wcs)
    limS = float(np.nanpercentile(np.abs(Smap[roi2d & np.isfinite(Smap)]), 99.0)) if np.any(roi2d & np.isfinite(Smap)) else 1.0
    im2 = _imshow(ax2, np.where(roi2d, Smap, np.nan), "Soft map S", max(limS, 1e-12))
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 2, 3, projection=wcs)
    im3 = ax3.imshow(np.where(roi2d, sig_sm, np.nan), origin="lower", cmap="magma")
    ax3.set_title("Hard significance (smoothed)")
    ax3.set_xlabel("l")
    ax3.set_ylabel("b")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4, projection=wcs)
    limH2 = float(np.nanpercentile(np.abs(Hmap[roi2d & np.isfinite(Hmap)]), 99.0)) if np.any(roi2d & np.isfinite(Hmap)) else 1.0
    im4 = _imshow(ax4, np.where(roi2d, Hmap, np.nan), "Hard map H + bubble outline", max(limH2, 1e-12))
    ax4.contour(
        np.where(roi2d, full_mask.astype(float), np.nan),
        levels=[0.5],
        colors=["white"],
        linewidths=1.5,
        origin="lower",
    )
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_xlabel("l")
    ax4.set_ylabel("b")

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sca_summary.png"), dpi=200)
    plt.close()


def derive_bubbles_mask_sca(
    *,
    counts,
    expo,
    omega,
    dE_mev,
    Ectr_mev,
    lon_w,
    lat,
    mu_gas,
    mu_ics,
    mu_iso,
    mu_ps,
    roi2d,
    binsz_deg,
    e_min_gev=1.0,
    e_max_gev=10.0,
    hard_index=1.9,
    soft_index=2.4,
    sig_smooth_deg=1.0,
    sig_thresh=2.0,
    resid_sigma_smooth_deg=1.0,
    sca_fit_lat_min=10.0,
    sca_domain_lat_min=10.0,
    hilat_cut_deg=10.0,
    clip_counts_pctl=99.9,
    outdir=None,
    wcs=None,
    debug_subtraction=False,
    debug_energy_gev=3.775,
    nnls_weights_mode="poisson",
    smooth_terms=30,
    smooth_max_degree=9,
    smooth_basis="poly",
    smooth_l_max=9,
    smooth_library_max_modes=100,
    mask_lon_max_deg=None,
    mask_lat_max_deg=None,
):
    """
    SCA-like bubbles mask:
      - build residual flux maps in 1–10 GeV after fitting (iem+iso+ps)
      - decompose into hard/soft power laws per pixel
      - threshold hard-component significance (smoothed) and keep connected lobes
    """
    Ectr_gev = Ectr_mev / 1000.0
    print(Ectr_gev)
    use = (Ectr_gev >= e_min_gev) & (Ectr_gev <= e_max_gev)
    idx = np.where(use)[0]
    if idx.size < 2:
        raise RuntimeError("Need at least 2 energy bins in 1–10 GeV for SCA decomposition.")

    k_dbg = int(np.argmin(np.abs(Ectr_gev - debug_energy_gev)))

    ny, nx = roi2d.shape
    nPix = ny * nx

    # Residual flux and its (approx) uncertainty per pixel, per selected energy bin
    R = np.empty((idx.size, nPix), float)
    Sig = np.empty((idx.size, nPix), float)

    if debug_subtraction:
        sum_counts = np.zeros((ny, nx), dtype=float)
        sum_model = np.zeros((ny, nx), dtype=float)
        sum_resid = np.zeros((ny, nx), dtype=float)

    sigma_resid_pix = resid_sigma_smooth_deg / binsz_deg

    # Select smooth proxy basis once (paper-style: keep top N out of a fixed library)
    library = None
    proxies_selected = None
    if str(smooth_basis).lower() == "ylm":
        library = build_smooth_proxies_ylm_2d(
            lon_w,
            lat,
            roi2d,
            l_max=int(smooth_l_max),
            n_max_modes=int(smooth_library_max_modes),
        )
        fitmask_dbg = roi2d & (np.abs(lat) >= sca_fit_lat_min)
        proxies_selected = select_smooth_proxies_by_delta_chi2(
            counts_k=counts[k_dbg],
            mu_gas_k=mu_gas[k_dbg],
            mu_ps_k=mu_ps[k_dbg],
            roi2d=roi2d,
            fitmask2d=fitmask_dbg,
            library=library,
            n_keep=int(smooth_terms),
            eps=1.0,
        )
        print(f"[sca] smooth_basis=ylm: selected {len(proxies_selected)} modes from library of {len(library)}")
    for j, k in enumerate(idx):
        # Step A/B (Ackermann+2017 / Totani-style):
        # Fit trusted components (Gas + PS) together with a set of smooth proxies
        # (to soak up other diffuse emission) but subtract ONLY the trusted part.
        comps = []
        names = []
        trusted = []

        comps.append(mu_gas[k]); names.append("Gas"); trusted.append(True)
        comps.append(mu_ps[k]); names.append("Point sources"); trusted.append(True)

        if str(smooth_basis).lower() == "ylm":
            proxies = [t for _, t in (proxies_selected or [])]
        else:
            proxies = build_smooth_proxies_2d(
                lon_w,
                lat,
                roi2d,
                roi_lon_deg=float(np.nanmax(np.abs(lon_w[roi2d]))),
                roi_lat_deg=float(np.nanmax(np.abs(lat[roi2d]))),
                n_terms=int(smooth_terms),
                max_total_degree=int(smooth_max_degree),
            )

        # Allow signed smooth coefficients by including +/- template columns with non-negative amplitudes.
        for i, t in enumerate(proxies):
            comps.append(t); names.append(f"Smooth{i:02d}+"); trusted.append(False)
            comps.append(-t); names.append(f"Smooth{i:02d}-"); trusted.append(False)

        # --- Fit mask: exclude the plane from the subtraction fit (disk drives imperfections)
        fitmask2d = roi2d & (np.abs(lat) >= sca_fit_lat_min)
        A = fit_bin_weighted_nnls(
            counts[k],
            comps,
            fitmask2d,
            normalize_components=True,
            weights_mode=nnls_weights_mode,
        )

        model_all = np.zeros_like(counts[k], dtype=float)
        model_trusted = np.zeros_like(counts[k], dtype=float)
        components_scaled = {}
        for a, mu, nm, is_trusted in zip(A, comps, names, trusted):
            scaled = a * mu
            model_all += scaled
            components_scaled[nm] = scaled
            if is_trusted:
                model_trusted += scaled

        smooth_model = model_all - model_trusted

        # Residual used for SCA decomposition: subtract ONLY trusted components
        resid_counts = counts[k] - model_trusted
        resid_after_all = counts[k] - model_all

        if debug_subtraction:
            sum_counts += np.nan_to_num(counts[k], nan=0.0, posinf=0.0, neginf=0.0)
            sum_model += np.nan_to_num(model_trusted, nan=0.0, posinf=0.0, neginf=0.0)
            sum_resid += np.nan_to_num(resid_counts, nan=0.0, posinf=0.0, neginf=0.0)

        denom = expo[k] * omega * dE_mev[k]  # cm^2 s * sr * MeV

        # --- SCA domain: build residual flux only at high latitudes (paper goal)
        sca_domain = roi2d & (np.abs(lat) >= sca_domain_lat_min)

        # --- robust bad-pixel masking ---
        good = sca_domain & np.isfinite(denom) & (denom > 0)
        bad = np.zeros_like(good, dtype=bool)

        # very low denom tail -> huge flux / unstable significance
        denom_roi = denom[sca_domain & np.isfinite(denom) & (denom > 0)]
        if denom_roi.size:
            denom_min = float(np.percentile(denom_roi, 0.5))
            bad |= denom < denom_min

        if k == k_dbg:
            m = good  # domain & denom>0 & not outlier

            def pstats(name, arr):
                x = np.asarray(arr)[m]
                x = x[np.isfinite(x)]
                if x.size == 0:
                    print(f"[units] {name}: EMPTY")
                    return
                q = np.percentile(x, [0, 1, 10, 50, 90, 99, 100])
                print(f"[units] {name}: "
                      f"min={q[0]:.3g} p1={q[1]:.3g} p10={q[2]:.3g} p50={q[3]:.3g} "
                      f"p90={q[4]:.3g} p99={q[5]:.3g} max={q[6]:.3g}")

            print("\n" + "=" * 80)
            print(f"[units] DEBUG @ k={k}  Ectr={Ectr_gev[k]:.6g} GeV")
            print(f"[units] binsz_deg={binsz_deg}  resid_sigma_smooth_deg={resid_sigma_smooth_deg}  sig_smooth_deg={sig_smooth_deg}")
            print(f"[units] dE_mev={dE_mev[k]:.6g}")

            # Exposure / omega / denom sanity
            pstats("expo[k] (cm^2 s?)", expo[k])
            pstats("omega (sr)", omega)
            pstats("denom = expo*omega*dE (cm^2 s sr MeV)", denom)

            # Counts sanity
            pstats("counts[k] (counts)", counts[k])
            for nm, mu in zip(names, comps):
                pstats(f"{nm} mu (counts)", mu)

            # Fit coefficients
            print("[units] NNLS amplitudes A:", {nm: float(a) for nm, a in zip(names, A)})
            # How big are the scaled components compared to counts?
            for nm in names:
                pstats(f"{nm} scaled = A*mu (counts)", components_scaled[nm])

            pstats("model_trusted (counts)", model_trusted)
            pstats("smooth_model (counts)", smooth_model)
            pstats("model_all (counts)", model_all)
            pstats("resid_after_trusted (counts)", resid_counts)
            pstats("resid_after_all (counts)", resid_after_all)

            # Convert model & counts into "flux-like" units for scale sanity
            counts_flux = np.full((ny, nx), np.nan, dtype=float)
            model_flux = np.full((ny, nx), np.nan, dtype=float)
            resid_flux_dbg = np.full((ny, nx), np.nan, dtype=float)
            counts_flux[m] = counts[k][m] / denom[m]
            model_flux[m] = model_trusted[m] / denom[m]
            resid_flux_dbg[m] = resid_counts[m] / denom[m]

            pstats("counts/denom (ph cm^-2 s^-1 sr^-1 MeV^-1)", counts_flux)
            pstats("model/denom  (ph cm^-2 s^-1 sr^-1 MeV^-1)", model_flux)
            pstats("resid/denom  (ph cm^-2 s^-1 sr^-1 MeV^-1)", resid_flux_dbg)

            # Sigma sanity
            sigma_counts_dbg = np.sqrt(np.maximum(counts[k], 1.0))
            sigma_flux_dbg = np.full((ny, nx), np.nan, dtype=float)
            sigma_flux_dbg[m] = sigma_counts_dbg[m] / denom[m]
            pstats("sigma_counts = sqrt(counts) (counts)", sigma_counts_dbg)
            pstats("sigma_flux = sigma_counts/denom (ph cm^-2 s^-1 sr^-1 MeV^-1)", sigma_flux_dbg)

            # Effective per-pixel significance in this one bin (not the SCA multi-bin significance)
            sig1 = np.full((ny, nx), np.nan, dtype=float)
            sig1[m] = resid_counts[m] / np.maximum(sigma_counts_dbg[m], 1.0)
            pstats("single-bin resid/sqrt(model) (sigma)", sig1)

            print("=" * 80 + "\n")

        if debug_subtraction:
            good_mask = good

            def _sum(x2d, mask2d):
                z = np.asarray(x2d, dtype=float)[mask2d]
                z = z[np.isfinite(z)]
                return float(np.nansum(z))

            roi_mask = roi2d
            fit_mask = fitmask2d
            dom_mask = good_mask

            # Unscaled component sums (mu templates as read)
            mu_unscaled = {nm: mu for nm, mu in zip(names, comps)}
            # Scaled component sums (after NNLS amplitude)
            mu_scaled = components_scaled

            data_roi = _sum(counts[k], roi_mask)
            data_fit = _sum(counts[k], fit_mask)
            data_dom = _sum(counts[k], dom_mask)

            model_roi = _sum(model_trusted, roi_mask)
            model_fit = _sum(model_trusted, fit_mask)
            model_dom = _sum(model_trusted, dom_mask)

            resid_roi = _sum(resid_counts, roi_mask)
            resid_fit = _sum(resid_counts, fit_mask)
            resid_dom = _sum(resid_counts, dom_mask)

            # Print one compact line per energy bin
            if j == 0:
                print("[counts-by-bin] Columns are sums over mask (ROI / FIT / GOOD):")
                print("[counts-by-bin] E_GeV  data  |  GAS(un,sc)  PS(un,sc)  |  model_trusted  resid")

            def _fmt(v):
                return f"{v:.3g}"

            def _pair(nm, mask):
                return (
                    _sum(mu_unscaled[nm], mask),
                    _sum(mu_scaled[nm], mask),
                )

            gas_nm = "Gas"
            ps_nm = "Point sources"

            gas_roi_u, gas_roi_s = _pair(gas_nm, roi_mask) if gas_nm in mu_unscaled else (np.nan, np.nan)
            gas_fit_u, gas_fit_s = _pair(gas_nm, fit_mask) if gas_nm in mu_unscaled else (np.nan, np.nan)
            gas_dom_u, gas_dom_s = _pair(gas_nm, dom_mask) if gas_nm in mu_unscaled else (np.nan, np.nan)

            ps_roi_u, ps_roi_s = _pair(ps_nm, roi_mask)
            ps_fit_u, ps_fit_s = _pair(ps_nm, fit_mask)
            ps_dom_u, ps_dom_s = _pair(ps_nm, dom_mask)

            print(
                f"[counts-by-bin] {Ectr_gev[k]:7.3f}  "
                f"data={_fmt(data_roi)}/{_fmt(data_fit)}/{_fmt(data_dom)}  |  "
                f"GAS={_fmt(gas_roi_u)},{_fmt(gas_roi_s)}/{_fmt(gas_fit_u)},{_fmt(gas_fit_s)}/{_fmt(gas_dom_u)},{_fmt(gas_dom_s)}  "
                f"PS={_fmt(ps_roi_u)},{_fmt(ps_roi_s)}/{_fmt(ps_fit_u)},{_fmt(ps_fit_s)}/{_fmt(ps_dom_u)},{_fmt(ps_dom_s)}  |  "
                f"model={_fmt(model_roi)}/{_fmt(model_fit)}/{_fmt(model_dom)}  "
                f"resid={_fmt(resid_roi)}/{_fmt(resid_fit)}/{_fmt(resid_dom)}"
            )


        # --- DEBUG PLOTS: BEFORE SCA hard/soft fit (i.e., before building R/Sig arrays)
        if debug_subtraction and (outdir is not None) and (wcs is not None) and (k == k_dbg):
            out_png = os.path.join(outdir, f"debug_subtraction_E{Ectr_gev[k]:.3f}GeV.png")
            plot_subtraction_diagnostics(
                out_png=out_png,
                wcs=wcs,
                roi2d=roi2d,
                lon_w=lon_w,
                lat=lat,
                title_prefix=f"Subtraction diagnostics @ E={Ectr_gev[k]:.3f} GeV",
                counts_k=counts[k],
                expo_k=expo[k],
                omega=omega,
                dE_mev_k=dE_mev[k],
                components_scaled={
                    "Gas": components_scaled.get("Gas", np.zeros((ny, nx))),
                    "Point sources": components_scaled.get("Point sources", np.zeros((ny, nx))),
                    "Smooth proxies (sum)": smooth_model,
                },
                model_counts=model_trusted,
                resid_counts=resid_counts,
                binsz_deg=binsz_deg,
            )
            print("✓ Wrote", out_png)

            out_png2 = os.path.join(outdir, f"debug_subtraction_masked_E{Ectr_gev[k]:.3f}GeV.png")
            plot_subtraction_diagnostics_masked(
                out_png=out_png2,
                wcs=wcs,
                good_mask2d=good,
                title_prefix=f"Masked subtraction diagnostics @ E={Ectr_gev[k]:.3f} GeV (SCA domain & denom>0 & ~outlier)",
                counts_k=counts[k],
                expo_k=expo[k],
                omega=omega,
                dE_mev_k=dE_mev[k],
                components_scaled={
                    "Gas": components_scaled.get("Gas", np.zeros((ny, nx))),
                    "Point sources": components_scaled.get("Point sources", np.zeros((ny, nx))),
                    "Smooth proxies (sum)": smooth_model,
                },
                model_counts=model_trusted,
                resid_counts=resid_counts,
                binsz_deg=binsz_deg,
            )
            print("✓ Wrote", out_png2)


        # --- Build the residual flux and its uncertainty for the SCA decomposition.
        #
        # Important: the paper-style significance is effectively defined on *smoothed*
        # residual maps (they mention O(100) photons in the smoothing kernel). If we
        # only smooth the final significance map, we do NOT get the expected S/N boost
        # for extended structures.
        #
        # So: smooth residual counts and smooth the Poisson variance estimate before
        # converting to flux.
        resid_counts_dom = np.full((ny, nx), np.nan, dtype=float)
        resid_counts_dom[good] = resid_counts[good]
        resid_counts_sm = smooth_nan_2d(resid_counts_dom, sigma_pix=sigma_resid_pix)

        # Use model-based variance (requested: sqrt(model_counts + 1)), smoothed with
        # the same kernel as the residual.
        model_for_var = np.full((ny, nx), np.nan, dtype=float)
        model_for_var[good] = np.maximum(model_all[good], 0.0) + 1.0
        var_counts_sm = smooth_nan_2d(model_for_var, sigma_pix=sigma_resid_pix)
        sigma_counts_sm = np.sqrt(np.maximum(var_counts_sm, 0.0))

        resid_flux_sm = np.full((ny, nx), np.nan, dtype=float)
        resid_flux_sm[good] = resid_counts_sm[good] / denom[good]

        # Fill outside-domain pixels with zero residual and infinite sigma (zero weight)
        # to prevent NaNs from poisoning the weighted hard/soft fit.
        resid_flux_f = np.zeros((ny, nx), dtype=float)
        sigma_flux_f = np.full((ny, nx), np.inf, dtype=float)
        resid_flux_f[good] = resid_flux_sm[good]
        sigma_flux_f[good] = sigma_counts_sm[good] / denom[good]

        R[j] = resid_flux_f.reshape(nPix)
        Sig[j] = sigma_flux_f.reshape(nPix)

    # Hard/soft decomposition per pixel
    E0_gev = 1.0
    fH = (Ectr_gev[idx] / E0_gev) ** (-hard_index)
    fS = (Ectr_gev[idx] / E0_gev) ** (-soft_index)
    H, S, sigma_H = _fit_two_powerlaws_weighted(R, Sig, fH, fS)

    Hmap = H.reshape(ny, nx)
    Smap = S.reshape(ny, nx)
    sigmaH = sigma_H.reshape(ny, nx)

    sig = np.full((ny, nx), np.nan, dtype=float)
    ok = roi2d & (np.abs(lat) >= sca_domain_lat_min) & np.isfinite(Hmap) & np.isfinite(sigmaH) & (sigmaH > 0)
    sig[ok] = Hmap[ok] / sigmaH[ok]

    # Kill pathological pixels dominating the smoothing
    # (paper-style boundaries are from extended structures, not single pixels)
    SIG_CLIP = 10.0
    sig_clip = np.array(sig, copy=True)
    mclip = roi2d & np.isfinite(sig_clip)
    sig_clip[mclip] = np.clip(sig_clip[mclip], -SIG_CLIP, SIG_CLIP)

    # Smooth significance and threshold
    sigma_sig_pix = sig_smooth_deg / binsz_deg
    sig_sm = smooth_nan_2d(sig_clip, sigma_pix=sigma_sig_pix)
    mask_window = np.ones_like(roi2d, dtype=bool)
    if mask_lon_max_deg is not None:
        mask_window &= (np.abs(lon_w) <= float(mask_lon_max_deg))
    if mask_lat_max_deg is not None:
        mask_window &= (np.abs(lat) <= float(mask_lat_max_deg))

    above = (
        roi2d
        & mask_window
        & (np.abs(lat) >= sca_domain_lat_min)
        & np.isfinite(sig_sm)
        & (sig_sm >= sig_thresh)
    )

    # Connected components; pick largest lobe in north and south *within high-lat cut*
    lab, nlab = label(above.astype(np.int8))
    keep = np.zeros((ny, nx), dtype=bool)
    dom = roi2d & mask_window & (np.abs(lat) >= sca_domain_lat_min) & np.isfinite(sig_sm)
    if np.any(dom):
        mx = float(np.nanmax(sig_sm[dom]))
        p50 = float(np.nanpercentile(sig_sm[dom], 50))
        p90 = float(np.nanpercentile(sig_sm[dom], 90))
        p99 = float(np.nanpercentile(sig_sm[dom], 99))
    else:
        mx, p50, p90, p99 = np.nan, np.nan, np.nan, np.nan

    frac = float(np.nanmean(above[roi2d]))
    print(
        f"[debug] sig_sm in-domain: max={mx:.3f} p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}; "
        f"finite={int(np.sum(dom))} above={int(np.sum(above))} frac_above={frac:.3e} thresh={sig_thresh}"
    )

    if nlab == 0:
        return keep, sig_sm, above, Hmap, Smap

    north_best, south_best = 0, 0
    north_area, south_area = -1, -1
    MIN_AREA = int((2.0 / binsz_deg) ** 2)
    for lbl in range(1, nlab + 1):
        m_lbl = (lab == lbl)
        m_hilat = m_lbl & roi2d & (np.abs(lat) >= hilat_cut_deg)
        a = int(np.sum(m_hilat))
        if a < MIN_AREA:
            continue
        mean_lat = float(np.mean(lat[m_hilat]))
        if mean_lat >= 0 and a > north_area:
            north_area, north_best = a, lbl
        if mean_lat < 0 and a > south_area:
            south_area, south_best = a, lbl

    if north_best:
        keep |= (lab == north_best)
    if south_best:
        keep |= (lab == south_best)

    if debug_subtraction and (outdir is not None) and (wcs is not None):
        out_png = os.path.join(outdir, "debug_subtraction_sum_1to10GeV.png")
        plot_subtraction_diagnostics(
            out_png=out_png,
            wcs=wcs,
            roi2d=roi2d,
            lon_w=lon_w,
            lat=lat,
            title_prefix="Subtraction diagnostics: sum over 1–10 GeV",
            counts_k=sum_counts,
            expo_k=np.nanmean(expo[idx], axis=0),
            omega=omega,
            dE_mev_k=float(np.nansum(dE_mev[idx])),
            components_scaled={},
            model_counts=sum_model,
            resid_counts=sum_resid,
            binsz_deg=binsz_deg,
        )
        print("✓ Wrote", out_png)

    return keep, sig_sm, above, Hmap, Smap

def build_smooth_proxies_2d(
    lon_w,
    lat,
    roi2d,
    *,
    roi_lon_deg,
    roi_lat_deg,
    n_terms=30,
    max_total_degree=9,
):
    """Simple smooth proxy basis on a CAR patch.

    Generates low-order 2D polynomial basis functions in scaled lon/lat, ordered
    by increasing total degree, and normalizes each basis on the ROI.

    These act as a proxy for large-scale diffuse emission (IC, isotropic, Loop I,
    etc.) so the gas+PS amplitudes are not biased.
    """
    x = lon_w / float(roi_lon_deg)
    y = lat / float(roi_lat_deg)

    basis = []
    for d in range(max_total_degree + 1):
        for p in range(d + 1):
            q = d - p
            if p == 0 and q == 0:
                t = np.ones_like(x, dtype=float)
            else:
                t = (x ** p) * (y ** q)

            m = roi2d & np.isfinite(t)
            if not np.any(m):
                continue
            t0 = np.array(t, copy=True)
            t0[~m] = 0.0
            # Normalize to unit RMS on the ROI mask
            rms = float(np.sqrt(np.mean(t0[m] ** 2)))
            if not np.isfinite(rms) or rms <= 0:
                continue
            basis.append(t0 / rms)
            if len(basis) >= n_terms:
                return basis


def _real_sph_harm(l, m, theta, phi):
    """Real-valued spherical harmonics from complex Y_lm (scipy.special.sph_harm)."""
    y = sph_harm(m, l, phi, theta)
    if m == 0:
        return np.real(y)
    if m > 0:
        return np.sqrt(2.0) * ((-1) ** m) * np.real(y)
    # m < 0
    mm = -m
    y2 = sph_harm(mm, l, phi, theta)
    return np.sqrt(2.0) * ((-1) ** mm) * np.imag(y2)


def build_smooth_proxies_ylm_2d(
    lon_w,
    lat,
    roi2d,
    *,
    l_max=9,
    n_max_modes=100,
):
    """Build a library of real spherical harmonic maps Y_lm on the ROI patch.

    Returns a list of (name, template2d) in a deterministic order, up to
    n_max_modes entries.
    """
    lon_rad = np.deg2rad(lon_w)
    lon_rad = np.mod(lon_rad, 2.0 * np.pi)
    lat_rad = np.deg2rad(lat)
    theta = 0.5 * np.pi - lat_rad
    phi = lon_rad

    out = []
    for l in range(int(l_max) + 1):
        for m in range(-l, l + 1):
            y = _real_sph_harm(l, m, theta, phi)
            msk = roi2d & np.isfinite(y)
            if not np.any(msk):
                continue
            y0 = np.array(y, copy=True)
            y0[~msk] = 0.0
            rms = float(np.sqrt(np.mean(y0[msk] ** 2)))
            if not np.isfinite(rms) or rms <= 0:
                continue
            y0 = y0 / rms
            out.append((f"Y{l}_{m}", y0))
            if len(out) >= int(n_max_modes):
                return out
    return out


def select_smooth_proxies_by_delta_chi2(
    *,
    counts_k,
    mu_gas_k,
    mu_ps_k,
    roi2d,
    fitmask2d,
    library,
    n_keep,
    eps=1.0,
):
    """Rank candidate smooth modes by delta-chi2 improvement added on top of gas+PS.

    We:
      1) fit gas+PS with weighted NNLS
      2) compute residual r
      3) for each candidate template t, compute best signed coefficient a and
         delta-chi2 improvement analytically under weighted least squares.
    """
    A0 = fit_bin_weighted_nnls(
        counts_k,
        [mu_gas_k, mu_ps_k],
        fitmask2d,
        eps=eps,
        normalize_components=True,
        weights_mode="poisson",
    )
    model0 = A0[0] * mu_gas_k + A0[1] * mu_ps_k
    r = counts_k - model0

    m = fitmask2d & np.isfinite(r) & np.isfinite(counts_k)
    if not np.any(m):
        return []

    w = 1.0 / np.sqrt(np.maximum(counts_k[m], 0.0) + eps)
    w2 = w * w
    r_m = r[m]

    scored = []
    for name, t2d in library:
        t_m = t2d[m]
        if not np.all(np.isfinite(t_m)):
            continue
        denom = float(np.sum(w2 * (t_m ** 2)))
        if not np.isfinite(denom) or denom <= 0:
            continue
        num = float(np.sum(w2 * r_m * t_m))
        dchi2 = (num * num) / denom
        if np.isfinite(dchi2) and dchi2 > 0:
            scored.append((dchi2, name, t2d))

    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[: int(n_keep)]
    return [(name, t) for _, name, t in keep]

    return basis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=False)
    ap.add_argument("--expo", required=False)
    ap.add_argument("--templates-dir", required=False)
    ap.add_argument("--ext-mask", required=False, help="extended-source mask FITS True=keep")
    ap.add_argument("--ps-mask", required=False, help="optional point-source mask FITS True=keep")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "plots_fig1"))

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--disk-cut", type=float, default=10.0)
    ap.add_argument("--sigma-smooth-deg", type=float, default=1)

    ap.add_argument("--sca-e-min-gev", type=float, default=1.0)
    ap.add_argument("--sca-e-max-gev", type=float, default=10.0)
    ap.add_argument("--sca-fit-lat-min", type=float, default=10.0)
    ap.add_argument("--sca-domain-lat-min", type=float, default=10.0)
    ap.add_argument("--sca-hard-index", type=float, default=1.9)
    ap.add_argument("--sca-soft-index", type=float, default=2.4)
    ap.add_argument("--sca-sig-smooth-deg", type=float, default=1.0)
    ap.add_argument("--sca-sig-thresh", type=float, default=1.0)
    ap.add_argument("--sca-mask-lon-max-deg", type=float, default=25.0)
    ap.add_argument("--sca-mask-lat-max-deg", type=float, default=50.0)
    ap.add_argument("--sca-smooth-terms", type=int, default=30)
    ap.add_argument("--sca-smooth-max-degree", type=int, default=9)
    ap.add_argument("--sca-smooth-basis", type=str, default="ylm")
    ap.add_argument("--sca-smooth-l-max", type=int, default=9)
    ap.add_argument("--sca-smooth-library-max-modes", type=int, default=100)
    ap.add_argument("--sca-debug-subtraction", action="store_true")

    ap.add_argument("--user-mask-north-verts", type=str, default=None)
    ap.add_argument("--user-mask-south-verts", type=str, default=None)
    ap.add_argument("--products-prefix", type=str, default="bubbles_vertices_sca")

    args = ap.parse_args()

    _run_sca_default(args)
    return


def _run_sca_default(args):
    ctx = _stage_load_inputs(args)
    counts = ctx["counts"]
    expo = ctx["expo"]
    wcs = ctx["wcs"]
    Ectr_mev = ctx["Ectr_mev"]
    dE_mev = ctx["dE_mev"]
    omega = ctx["omega"]
    lon = ctx["lon"]
    lat = ctx["lat"]
    roi2d = ctx["roi2d"]
    templates_dir = ctx["templates_dir"]

    counts_m, expo_m, mask_all = _stage_apply_keep_masks(args=args, counts=counts, expo=expo, roi2d=roi2d, templates_dir=templates_dir)
    mu_gas_m, mu_ps_m = _stage_load_mu_templates(templates_dir=templates_dir, mask_all=mask_all, counts_shape=counts.shape)
    mu_dummy = np.zeros_like(mu_gas_m)

    connected_mask, sig_sm, above_thresh, Hmap, Smap = derive_bubbles_mask_sca(
        counts=counts_m,
        expo=expo_m,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        lon_w=lon,
        lat=lat,
        mu_gas=mu_gas_m,
        mu_ics=mu_dummy,
        mu_iso=mu_dummy,
        mu_ps=mu_ps_m,
        roi2d=roi2d,
        binsz_deg=args.binsz,
        e_min_gev=args.sca_e_min_gev,
        e_max_gev=args.sca_e_max_gev,
        hard_index=args.sca_hard_index,
        soft_index=args.sca_soft_index,
        sig_smooth_deg=args.sca_sig_smooth_deg,
        sig_thresh=args.sca_sig_thresh,
        resid_sigma_smooth_deg=args.sigma_smooth_deg,
        sca_fit_lat_min=args.sca_fit_lat_min,
        sca_domain_lat_min=args.sca_domain_lat_min,
        hilat_cut_deg=args.disk_cut,
        outdir=args.outdir,
        wcs=wcs,
        debug_subtraction=args.sca_debug_subtraction,
        smooth_terms=args.sca_smooth_terms,
        smooth_max_degree=args.sca_smooth_max_degree,
        smooth_basis=args.sca_smooth_basis,
        smooth_l_max=args.sca_smooth_l_max,
        smooth_library_max_modes=args.sca_smooth_library_max_modes,
        mask_lon_max_deg=args.sca_mask_lon_max_deg,
        mask_lat_max_deg=args.sca_mask_lat_max_deg,
    )

    ny, nx = roi2d.shape
    connected_mask = _stage_optional_user_polygon(args=args, wcs=wcs, ny=ny, nx=nx, connected_mask=connected_mask)
    full_mask = _stage_write_products(
        args=args,
        wcs=wcs,
        roi2d=roi2d,
        lat=lat,
        connected_mask=connected_mask,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
    )
    _stage_write_debug_maps(args=args, wcs=wcs, sig_sm=sig_sm, Hmap=Hmap, Smap=Smap)
    _stage_plot_sca_summary(args=args, wcs=wcs, roi2d=roi2d, sig_sm=sig_sm, Hmap=Hmap, Smap=Smap, full_mask=full_mask)


if __name__ == "__main__":
    main()
