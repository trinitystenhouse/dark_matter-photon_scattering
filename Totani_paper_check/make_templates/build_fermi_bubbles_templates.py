#!/usr/bin/env python3
"""
build_fermi_bubbles_templates_sca_hilat.py

Derive the Fermi bubbles *high-latitude* template (|b| > 10 deg) using the
Spectral Components Analysis (SCA) procedure described in arXiv:1704.03910,
Sec. 5.1.1–5.1.3 and Fig. 8.

Pipeline (matches the paper-level recipe):
1) For each energy bin in 1–10 GeV:
   - Fit gas-correlated (mu_gas) + IC (mu_ics) + isotropic (mu_iso) + point sources (mu_ps)
     to counts (weighted NNLS), within the SCA ROI (default |l|<45, |b|<60).
   - Compute residual flux map.
   - Estimate per-pixel statistical uncertainty (Poisson; with mild smoothing).
2) Per pixel, decompose residual flux vs energy into:
      R(E) ≈ H * E^{-1.9} + S * E^{-2.4}
   via weighted least squares, yielding H and sigma_H.
3) Build hard-component significance map: sig = H / sigma_H.
   Smooth sig with 1 deg Gaussian. Threshold at 2σ.
4) Keep only connected regions; select the largest connected region in the
   north and south hemispheres separately (bubble lobes).
5) Split into high latitude |b|>10 deg template (this script outputs *only* that).

Outputs (written to OUTDIR):
  - bubbles_hilat_mask.fits              [dimensionless; 1 inside template]
  - mu_bubbles_hilat_counts.fits         [counts]
  - bubbles_hilat_dnde.fits              [ph cm^-2 s^-1 sr^-1 MeV^-1]
  - bubbles_hilat_E2dnde.fits            [MeV cm^-2 s^-1 sr^-1]
  - debug_sca_hard_sig_smoothed.fits      [sigma units]
  - debug_sca_threshold_connected.fits    [0/1 mask after threshold+connected]

Notes:
- This builds the *spatial* template as in Fig. 8. In the fit, the spectrum is
  handled by per-energy amplitudes (your later fitting code).
"""

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

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

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


def plot_counts_after_masks(
    *,
    out_png,
    wcs,
    good_mask2d,
    title,
    counts_k,
):
    img = np.array(counts_k, dtype=float, copy=True)
    img[~good_mask2d] = np.nan

    plt.figure(figsize=(7.2, 4.8))
    ax = plt.subplot(1, 1, 1, projection=wcs)

    vmin, vmax = 0.0, float(np.nanpercentile(img[np.isfinite(img)], 99.5)) if np.any(np.isfinite(img)) else 1.0
    im = ax.imshow(img, origin="lower", norm=LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, 1.0)))
    ax.set_title(title)
    ax.set_xlabel("l")
    ax.set_ylabel("b")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="counts (masked, log scale)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    """Ω_pix ≈ Δl Δb cos(b) for CAR projection."""
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), dtype=float)


def smooth_nan_2d(x2d, sigma_pix):
    m = np.isfinite(x2d)
    if not np.any(m):
        return np.full_like(x2d, np.nan, dtype=float)
    w = m.astype(float)
    xs = gaussian_filter(np.where(m, x2d, 0.0), sigma_pix)
    ws = gaussian_filter(w, sigma_pix)
    y = np.full_like(x2d, np.nan, dtype=float)
    ok = ws > 0
    y[ok] = xs[ok] / ws[ok]
    return y


def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    """Interpolate exposure planes onto target energy centers in log(E)."""
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


def write_primary_with_bunit(path, data, hdr, bunit, comments=()):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=hdr)
    hdu.header["BUNIT"] = bunit
    for c in comments:
        c_ascii = str(c).encode("ascii", "replace").decode("ascii")
        hdu.header.add_comment(c_ascii)
    hdu.writeto(path, overwrite=True)


def read_vertices_lonlat(path):
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


def polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_lonlat_deg):
    lon = np.mod(verts_lonlat_deg[:, 0], 360.0)
    lat = verts_lonlat_deg[:, 1]
    x, y = wcs.world_to_pixel_values(lon, lat)
    poly = Path(np.vstack([x, y]).T)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    return poly.contains_points(pts).reshape(ny, nx)


def build_flat_bubbles_template_from_vertices(*, wcs, roi2d, verts_north, verts_south):
    ny, nx = roi2d.shape
    m = np.zeros((ny, nx), dtype=bool)
    if verts_north is not None:
        m |= polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_north)
    if verts_south is not None:
        m |= polygon_mask_from_lonlat_vertices(wcs, ny, nx, verts_south)
    m &= roi2d
    return m


def write_flat_bubbles_template_products(
    *,
    outdir,
    prefix,
    hdr,
    wcs,
    roi2d,
    mask2d,
    expo,
    omega,
    dE_mev,
    Ectr_mev,
):
    nE, ny, nx = expo.shape
    if mask2d.shape != (ny, nx):
        raise RuntimeError("mask2d shape mismatch")

    m = np.asarray(mask2d, dtype=bool) & np.asarray(roi2d, dtype=bool)
    if not np.any(m):
        raise RuntimeError(f"Empty bubbles mask within ROI for prefix='{prefix}'")

    T = np.zeros((ny, nx), float)
    T[m] = 1.0
    T[~roi2d] = 0.0

    s = float(np.nansum(T[m]))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Normalization failed (template sum <= 0).")
    T /= s

    dnde = np.empty((nE, ny, nx), float)
    for k in range(nE):
        dnde[k] = T / (omega * dE_mev[k])

    E2dnde = dnde * (Ectr_mev[:, None, None] ** 2)
    mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    out_mask = os.path.join(outdir, f"{prefix}_mask.fits")
    out_tmpl = os.path.join(outdir, f"{prefix}_template.fits")
    out_mu = os.path.join(outdir, f"mu_{prefix}_counts.fits")
    out_dnde = os.path.join(outdir, f"{prefix}_dnde.fits")
    out_e2 = os.path.join(outdir, f"{prefix}_E2dnde.fits")

    write_primary_with_bunit(out_mask, m.astype(np.int16), hdr, "dimensionless")
    write_primary_with_bunit(out_tmpl, T, hdr, "dimensionless")
    write_primary_with_bunit(out_mu, mu, hdr, "counts")
    write_primary_with_bunit(out_dnde, dnde, hdr, "ph cm-2 s-1 sr-1 MeV-1")
    write_primary_with_bunit(out_e2, E2dnde, hdr, "MeV cm-2 s-1 sr-1")
    return mu


def fit_components_with_flat_bubbles(
    *,
    counts,
    roi2d,
    mu_gas,
    mu_ics,
    mu_iso,
    mu_ps,
    mu_nfw,
    mu_loopi,
    mu_bubbles_flat,
    weights_mode="uniform",
    include_ics=False,
    include_loopi=False,
):
    nE, ny, nx = counts.shape
    if mu_nfw is None:
        raise RuntimeError("mu_nfw must be provided to include GC excess in the fit")

    if include_loopi and (mu_loopi is None):
        raise RuntimeError("include_loopi=True but mu_loopi is None")

    fitmask2d = roi2d  # includes disk (|b|<10°) as requested
    names = ["Gas"]
    if include_ics:
        names.append("ICS")
    names += ["Isotropic", "Point sources", "NFW"]
    if include_loopi:
        names.append("Loop I")
    names.append("BubblesFlat")

    A = np.zeros((nE, len(names)), dtype=float)

    for k in range(nE):
        good = fitmask2d & np.isfinite(counts[k])
        if not np.any(good):
            continue
        comps = [mu_gas[k]]
        if include_ics:
            comps.append(mu_ics[k])
        comps += [mu_iso[k], mu_ps[k], mu_nfw[k]]
        if include_loopi:
            comps.append(mu_loopi[k])
        comps.append(mu_bubbles_flat[k])
        A[k] = fit_bin_weighted_nnls(
            counts[k],
            comps,
            good,
            eps=1.0,
            normalize_components=True,
            weights_mode=weights_mode,
        )
    return names, A


def _fit_cellwise_and_build_bestfit_bubble_counts(
    *,
    counts_k,
    roi2d,
    lon_w,
    lat,
    cell_deg,
    mu_gas_k,
    mu_iso_k,
    mu_ps_k,
    mu_nfw_k,
    mu_bflat_k,
    mu_ics_k=None,
    mu_loopi_k=None,
    weights_mode="uniform",
):
    ny, nx = roi2d.shape
    out = np.full((ny, nx), np.nan, dtype=float)

    l_edges = np.arange(-float(np.nanmax(np.abs(lon_w[roi2d]))), float(np.nanmax(np.abs(lon_w[roi2d]))) + 1e-6, float(cell_deg))
    b_edges = np.arange(-float(np.nanmax(np.abs(lat[roi2d]))), float(np.nanmax(np.abs(lat[roi2d]))) + 1e-6, float(cell_deg))

    for l0 in l_edges[:-1]:
        l1 = l0 + float(cell_deg)
        in_l = (lon_w >= l0) & (lon_w < l1)
        for b0 in b_edges[:-1]:
            b1 = b0 + float(cell_deg)
            cell_mask = roi2d & in_l & (lat >= b0) & (lat < b1)
            if not np.any(cell_mask):
                continue

            comps = [mu_gas_k]
            if mu_ics_k is not None:
                comps.append(mu_ics_k)
            comps += [mu_iso_k, mu_ps_k, mu_nfw_k]
            if mu_loopi_k is not None:
                comps.append(mu_loopi_k)
            comps.append(mu_bflat_k)

            A = fit_bin_weighted_nnls(
                counts_k,
                comps,
                cell_mask & np.isfinite(counts_k),
                eps=1.0,
                normalize_components=True,
                weights_mode=weights_mode,
            )

            # Rebuild the cell model in the same order as comps above.
            # Determine index of BubblesFlat (last component).
            idx_bflat = len(comps) - 1

            model_cell = np.zeros(int(np.sum(cell_mask)), dtype=float)
            ii = 0
            model_cell += A[ii] * mu_gas_k[cell_mask]
            ii += 1
            if mu_ics_k is not None:
                model_cell += A[ii] * mu_ics_k[cell_mask]
                ii += 1
            model_cell += A[ii] * mu_iso_k[cell_mask]; ii += 1
            model_cell += A[ii] * mu_ps_k[cell_mask]; ii += 1
            model_cell += A[ii] * mu_nfw_k[cell_mask]; ii += 1
            if mu_loopi_k is not None:
                model_cell += A[ii] * mu_loopi_k[cell_mask]
                ii += 1

            bubble_cell = A[idx_bflat] * mu_bflat_k[cell_mask]
            model_cell += bubble_cell

            resid_cell = counts_k[cell_mask] - model_cell
            out[cell_mask] = bubble_cell + resid_cell

    return out


def _plot_bestfit_bubble_flux(
    *,
    out_png,
    wcs,
    roi2d,
    disk_cut_deg,
    lat,
    flux2d,
    flat_boundary2d,
    smooth_sigma_deg,
    binsz_deg,
    title,
):
    sigma_pix = float(smooth_sigma_deg) / float(binsz_deg)
    img = np.array(flux2d, dtype=float, copy=True)
    img[~roi2d] = np.nan
    img[np.abs(lat) < float(disk_cut_deg)] = np.nan
    img_sm = smooth_nan_2d(img, sigma_pix=sigma_pix)
    img_sm[np.abs(lat) < float(disk_cut_deg)] = np.nan

    dom = roi2d & (np.abs(lat) >= float(disk_cut_deg)) & np.isfinite(img_sm)
    if np.any(dom):
        lim = float(np.nanpercentile(np.abs(img_sm[dom]), 99.0))
        lim = max(lim, 1e-20)
    else:
        lim = 1.0

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111, projection=wcs)
    im = ax.imshow(img_sm, origin="lower", cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax.contour(
        np.where(roi2d & (np.abs(lat) >= float(disk_cut_deg)), flat_boundary2d.astype(float), np.nan),
        levels=[0.5],
        colors=["w"],
        linewidths=1.6,
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("l")
    ax.set_ylabel("b")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"flux [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$]")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_component_spectra(
    *,
    out_png,
    Ectr_gev,
    names,
    A,
    title,
):
    plt.figure(figsize=(7.2, 4.8))
    for i, nm in enumerate(names):
        plt.plot(Ectr_gev, A[:, i], marker="o", linewidth=1.3, markersize=3.5, label=nm)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("E [GeV]")
    plt.ylabel("fit amplitude")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


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

def stats(name, a, roi2d):
    x = a[roi2d]
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(name, "EMPTY")
        return
    p50, p90, p99, p999 = np.percentile(x, [50, 90, 99, 99.9])
    print(f"[debug] {name}: min={x.min():.3g}  p50={p50:.3g}  p90={p90:.3g}  p99={p99:.3g}  p99.9={p999:.3g}  max={x.max():.3g}")


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

        # extreme counts outliers (PS wings / edge artifacts)
        # cnt_roi = counts[k][sca_domain]
        # cnt_roi = cnt_roi[np.isfinite(cnt_roi)]
        # if cnt_roi.size:
        #     ccut = float(np.percentile(cnt_roi, clip_counts_pctl)) if (clip_counts_pctl is not None) else float(
        #         np.percentile(cnt_roi, 99.9)
        #     )
        #     bad |= counts[k] > ccut

        # good &= ~bad

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plot",
        action="store_true",
        help="If set, write diagnostic plots (PNG) to outdir.",
    )

    ap.add_argument("--counts", default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"))
    ap.add_argument("--expcube", default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"))

    ap.add_argument("--mu-gas", default=os.path.join(DATA_DIR, "processed", "templates", "mu_gas_counts.fits"))
    ap.add_argument("--mu-ics", default=os.path.join(DATA_DIR, "processed", "templates", "mu_ics_counts.fits"))
    ap.add_argument("--mu-iso", default=os.path.join(DATA_DIR, "processed", "templates", "mu_iso_counts.fits"))
    ap.add_argument("--mu-ps",  default=os.path.join(DATA_DIR, "processed", "templates", "mu_ps_counts.fits"))
    ap.add_argument("--mu-nfw", default=os.path.join(DATA_DIR, "processed", "templates", "mu_nfw_counts.fits"))
    ap.add_argument("--mu-loopI", default=os.path.join(DATA_DIR, "processed", "templates", "mu_loopI_counts.fits"))

    ap.add_argument("--outdir", default=os.path.join(DATA_DIR, "processed", "templates"))
    ap.add_argument("--binsz", type=float, default=0.125)

    # Paper-style SCA ROI
    ap.add_argument("--sca-roi-lon", type=float, default=45.0)
    ap.add_argument("--sca-roi-lat", type=float, default=60.0)

    ap.add_argument("--flat-roi-lon", type=float, default=60.0)
    ap.add_argument("--flat-roi-lat", type=float, default=60.0)

    # SCA spectral decomposition settings
    ap.add_argument("--sca-e-min-gev", type=float, default=1.0)
    ap.add_argument("--sca-e-max-gev", type=float, default=10.0)
    ap.add_argument("--sca-hard-index", type=float, default=1.9)
    ap.add_argument("--sca-soft-index", type=float, default=2.4)
    ap.add_argument("--sca-sig-smooth-deg", type=float, default=1.0)
    ap.add_argument("--sca-sig-thresh", type=float, default=0.2)
    ap.add_argument("--sca-resid-sigma-smooth-deg", type=float, default=1.0)
    ap.add_argument(
        "--sca-fit-lat-min",
        type=float,
        default=10.0,
        help="Exclude |b|<this (deg) from NNLS subtraction fit (masks the GC/plane).",
    )
    ap.add_argument(
        "--sca-domain-lat-min",
        type=float,
        default=None,
        help="Restrict SCA residuals/significance to |b|>=this (deg). Default: hilat-cut-deg.",
    )
    ap.add_argument(
        "--sca-clip-counts-pctl",
        type=float,
        default=99.9,
        help="Mask pixels with counts above this percentile within the SCA domain (100 disables).",
    )
    ap.add_argument(
        "--sca-nnls-weights",
        choices=["poisson", "uniform"],
        default="poisson",
        help="Weighting for NNLS subtraction fit. 'poisson' downweights bright pixels; 'uniform' can help ISO not go to zero.",
    )
    ap.add_argument(
        "--debug-subtraction",
        action="store_true",
        help="Write diagnostic PNGs showing data and subtraction components for one energy bin (before SCA spectral fit).",
    )
    ap.add_argument(
        "--debug-energy-gev",
        type=float,
        default=3.775,
        help="Energy (GeV) whose bin will be used for subtraction debug plots (choose within 1–10 GeV).",
    )

    ap.add_argument(
        "--sca-smooth-terms",
        type=int,
        default=30,
        help="Number of smooth proxy templates used in the subtraction fit (Step B).",
    )
    ap.add_argument(
        "--sca-smooth-max-degree",
        type=int,
        default=9,
        help="Max total polynomial degree for smooth proxy basis (Step B).",
    )

    ap.add_argument(
        "--flat-vertices-template",
        action="store_true",
        help="If set, build a flat (isotropic) bubbles template with sharp edges defined by bubble-vertex polygons, and fit it together with other components over the full ROI.",
    )
    ap.add_argument(
        "--bubble-verts-north",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "bubble_vertices_north.txt"),
        help="Path to north bubble polygon vertices text file (lon lat per line).",
    )
    ap.add_argument(
        "--bubble-verts-south",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "bubble_vertices_south.txt"),
        help="Path to south bubble polygon vertices text file (lon lat per line).",
    )
    ap.add_argument(
        "--flat-products-prefix",
        type=str,
        default="bubbles_vertices_flat_full",
        help="Prefix for written flat-vertex bubbles template products.",
    )

    ap.add_argument(
        "--flat-fit-include-ics",
        action="store_true",
        help="Include the ICS template in the flat-vertices fit (default: off, to match 'no other halo components').",
    )

    ap.add_argument(
        "--flat-fit-include-loopI",
        action="store_true",
        help="Include the Loop I template in the flat-vertices fit.",
    )

    ap.add_argument(
        "--flat-nnls-weights",
        choices=["uniform", "poisson"],
        default="uniform",
        help="Weighting for the flat-vertices normalization fit. 'uniform' matches Totani-style cell-wise fitting; 'poisson' downweights bright pixels.",
    )

    ap.add_argument("--flat-debug-templates", action="store_true")
    ap.add_argument("--flat-debug-energy-gev", type=float, default=3.775)

    ap.add_argument("--flat-cell-deg", type=float, default=10.0)
    ap.add_argument("--flat-plot", action="store_true")
    ap.add_argument("--flat-plot-disk-cut-deg", type=float, default=10.0)
    ap.add_argument("--flat-plot-smooth-deg", type=float, default=1.0)
    ap.add_argument(
        "--flat-plot-energies-gev",
        type=float,
        nargs=2,
        default=[1.5, 4.3],
    )

    # High-lat split (what you want to plot like Fig. 8 bottom-left)
    ap.add_argument("--hilat-cut-deg", type=float, default=10.0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load counts + energies ---
    with fits.open(args.counts) as h:
        hdr = h[0].header
        counts = h[0].data.astype(float)
        eb = h["EBOUNDS"].data

    wcs = WCS(hdr).celestial
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])

    Emin_kev = eb["E_MIN"].astype(float)
    Emax_kev = eb["E_MAX"].astype(float)
    Ectr_kev = np.sqrt(Emin_kev * Emax_kev)
    Emin_mev = Emin_kev / 1000.0
    Emax_mev = Emax_kev / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    nE = len(Ectr_mev)

    print("[E] counts EBOUNDS:")
    print("[E]   Emin_keV:", np.round(Emin_kev, 3))
    print("[E]   Emax_keV:", np.round(Emax_kev, 3))
    print("[E]   Ectr_keV:", np.round(Ectr_kev, 3))
    print("[E]   Emin_MeV:", np.round(Emin_mev, 6))
    print("[E]   Emax_MeV:", np.round(Emax_mev, 6))
    print("[E]   Ectr_MeV:", np.round(Ectr_mev, 6))
    print("[E]   Ectr_GeV:", np.round(Ectr_mev / 1000.0, 6))
    print("[E]   dE_MeV  :", np.round(dE_mev, 6))

    use = (Ectr_mev/1000 >= args.sca_e_min_gev) & (Ectr_mev/1000 <= args.sca_e_max_gev)
    print("[debug] # bins in 1–10 GeV:", int(np.sum(use)))
    print("[debug] selected Ectr_gev:", Ectr_mev[use]/1000)


    # --- exposure ---
    with fits.open(args.expcube) as h:
        expo_raw = h[0].data.astype(float)
        E_expo_mev = None
        if "ENERGIES" in h:
            col0 = h["ENERGIES"].columns.names[0]
            E_expo_mev = np.array(h["ENERGIES"].data[col0], dtype=float)
            print("[E] expcube ENERGIES (as stored, assumed MeV):")
            print("[E]   N:", int(E_expo_mev.size))
            print("[E]   min/max:", float(np.nanmin(E_expo_mev)), float(np.nanmax(E_expo_mev)))
            print("[E]   first/last:", float(E_expo_mev[0]), float(E_expo_mev[-1]))
        else:
            print("[E] expcube has no ENERGIES extension")

    print("[E] expcube planes (raw):", expo_raw.shape)

    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != (nE, ny, nx):
        raise RuntimeError(f"Exposure shape {expo.shape} not compatible with {(nE, ny, nx)}")

    print("[E] expcube planes (resampled):", expo.shape)

    # --- lon/lat, omega ---
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon_w = ((lon + 180.0) % 360.0) - 180.0
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # --- ROI mask ---
    # For SCA pipeline we use the SCA ROI, but for the flat-vertices normalization fit
    # we use a separate (typically larger) ROI.
    roi2d_sca = (np.abs(lon_w) <= args.sca_roi_lon) & (np.abs(lat) <= args.sca_roi_lat)
    roi2d_flat = (np.abs(lon_w) <= args.flat_roi_lon) & (np.abs(lat) <= args.flat_roi_lat)

    # --- read mu templates ---
    def _read_mu(path):
        with fits.open(path) as h:
            d = h[0].data.astype(float)
        if d.shape != (nE, ny, nx):
            raise RuntimeError(f"{path} has shape {d.shape}, expected {(nE, ny, nx)}")
        return d

    mu_gas = _read_mu(args.mu_gas)
    mu_ics = _read_mu(args.mu_ics)
    mu_iso = _read_mu(args.mu_iso)
    mu_ps  = _read_mu(args.mu_ps)
    mu_nfw = _read_mu(args.mu_nfw)
    mu_loopi = None
    if args.flat_fit_include_loopI:
        mu_loopi = _read_mu(args.mu_loopI)

    if args.flat_vertices_template:
        print(
            "[flat-vertices] Fitting component normalizations over full flat ROI (includes disk; no extended-source masking), skipping SCA thresholding/hi-lat mask"
        )
        # Flat (isotropic) bubbles template within vertex polygons
        verts_n = read_vertices_lonlat(args.bubble_verts_north) if args.bubble_verts_north else None
        verts_s = read_vertices_lonlat(args.bubble_verts_south) if args.bubble_verts_south else None
        flat_mask2d = build_flat_bubbles_template_from_vertices(
            wcs=wcs,
            roi2d=roi2d_flat,
            verts_north=verts_n,
            verts_south=verts_s,
        )

        mu_bflat = write_flat_bubbles_template_products(
            outdir=args.outdir,
            prefix=str(args.flat_products_prefix),
            hdr=hdr,
            wcs=wcs,
            roi2d=roi2d_flat,
            mask2d=flat_mask2d,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
        )

        if args.flat_debug_templates:
            Ectr_gev = Ectr_mev / 1000.0
            k_dbg = int(np.argmin(np.abs(Ectr_gev - float(args.flat_debug_energy_gev))))
            m = roi2d_flat & np.isfinite(counts[k_dbg])

            def _tmpl_stats(label, arr2d):
                x = np.asarray(arr2d, dtype=float)[m]
                x = x[np.isfinite(x)]
                if x.size == 0:
                    print(f"[flat-debug] {label}: EMPTY")
                    return
                s = float(np.nansum(x))
                mn = float(np.nanmin(x))
                mx = float(np.nanmax(x))
                print(f"[flat-debug] {label}: sum={s:.6g} min={mn:.6g} max={mx:.6g}")
                if not np.isfinite(s) or s == 0.0:
                    print(f"[flat-debug] WARNING: {label} has zero sum in ROI; NNLS amplitude will be forced to 0")

            print(f"[flat-debug] Template stats in ROI @ E={Ectr_gev[k_dbg]:.3f} GeV")
            _tmpl_stats("counts", counts[k_dbg])
            _tmpl_stats("mu_gas", mu_gas[k_dbg])
            _tmpl_stats("mu_ics", mu_ics[k_dbg])
            _tmpl_stats("mu_iso", mu_iso[k_dbg])
            _tmpl_stats("mu_ps", mu_ps[k_dbg])
            _tmpl_stats("mu_nfw", mu_nfw[k_dbg])
            if mu_loopi is not None:
                _tmpl_stats("mu_loopI", mu_loopi[k_dbg])
            _tmpl_stats("mu_bubbles_flat", mu_bflat[k_dbg])

        # Fit all components over the full ROI, including the Galactic disk region.
        names, A = fit_components_with_flat_bubbles(
            counts=counts,
            roi2d=roi2d_flat,
            mu_gas=mu_gas,
            mu_ics=mu_ics,
            mu_iso=mu_iso,
            mu_ps=mu_ps,
            mu_nfw=mu_nfw,
            mu_loopi=mu_loopi,
            mu_bubbles_flat=mu_bflat,
            weights_mode=str(args.flat_nnls_weights),
            include_ics=bool(args.flat_fit_include_ics),
            include_loopi=bool(args.flat_fit_include_loopI),
        )

        if args.flat_plot:
            Ectr_gev = Ectr_mev / 1000.0
            out_png = os.path.join(args.outdir, f"spectra_{args.flat_products_prefix}.png")
            _plot_component_spectra(
                out_png=out_png,
                Ectr_gev=Ectr_gev,
                names=names,
                A=A,
                title="Best-fit spectra (flat bubbles template)",
            )
            print("✓ Wrote", out_png)

            flat_boundary2d = flat_mask2d.astype(bool)
            disk_cut = float(args.flat_plot_disk_cut_deg)
            for Eplot in list(args.flat_plot_energies_gev):
                k = int(np.argmin(np.abs(Ectr_gev - float(Eplot))))
                bestfit_bubble_counts = _fit_cellwise_and_build_bestfit_bubble_counts(
                    counts_k=counts[k],
                    roi2d=roi2d_flat,
                    lon_w=lon_w,
                    lat=lat,
                    cell_deg=float(args.flat_cell_deg),
                    mu_gas_k=mu_gas[k],
                    mu_iso_k=mu_iso[k],
                    mu_ps_k=mu_ps[k],
                    mu_nfw_k=mu_nfw[k],
                    mu_bflat_k=mu_bflat[k],
                    mu_ics_k=(mu_ics[k] if args.flat_fit_include_ics else None),
                    mu_loopi_k=(mu_loopi[k] if args.flat_fit_include_loopI else None),
                    weights_mode=str(args.flat_nnls_weights),
                )

                denom = expo[k] * omega * dE_mev[k]
                flux = np.full((ny, nx), np.nan, dtype=float)
                good = roi2d_flat & np.isfinite(bestfit_bubble_counts) & np.isfinite(denom) & (denom > 0)
                flux[good] = bestfit_bubble_counts[good] / denom[good]

                out_png2 = os.path.join(args.outdir, f"bestfit_bubbles_{Ectr_gev[k]:.3f}GeV_{args.flat_products_prefix}.png")
                _plot_bestfit_bubble_flux(
                    out_png=out_png2,
                    wcs=wcs,
                    roi2d=roi2d_flat,
                    disk_cut_deg=disk_cut,
                    lat=lat,
                    flux2d=flux,
                    flat_boundary2d=flat_boundary2d,
                    smooth_sigma_deg=float(args.flat_plot_smooth_deg),
                    binsz_deg=float(args.binsz),
                    title=f"Best-fit bubbles image (E~{Ectr_gev[k]:.2f} GeV)",
                )
                print("✓ Wrote", out_png2)

        out_txt = os.path.join(args.outdir, f"fit_amplitudes_{args.flat_products_prefix}.txt")
        with open(out_txt, "w") as f:
            f.write("# Per-energy weighted NNLS fit amplitudes over full ROI\n")
            f.write("# Components: " + " ".join(names) + "\n")
            f.write("# Columns: Ectr_GeV  " + "  ".join([f"A_{n}" for n in names]) + "\n")
            for k in range(nE):
                f.write(
                    f"{Ectr_mev[k]/1000.0:.8g}  "
                    + "  ".join([f"{A[k, i]:.8g}" for i in range(A.shape[1])])
                    + "\n"
                )
        print("✓ Wrote:")
        print(" ", out_txt)
        return

    sca_domain_lat_min = args.hilat_cut_deg if (args.sca_domain_lat_min is None) else float(args.sca_domain_lat_min)

    # --- SCA mask (connected hard component above threshold) ---
    connected_mask, sig_sm, above_thresh, Hmap, Smap = derive_bubbles_mask_sca(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        lon_w=lon_w,
        lat=lat,
        mu_gas=mu_gas,
        mu_ics=mu_ics,
        mu_iso=mu_iso,
        mu_ps=mu_ps,
        roi2d=roi2d_sca,
        binsz_deg=args.binsz,
        e_min_gev=args.sca_e_min_gev,
        e_max_gev=args.sca_e_max_gev,
        hard_index=args.sca_hard_index,
        soft_index=args.sca_soft_index,
        sig_smooth_deg=args.sca_sig_smooth_deg,
        sig_thresh=args.sca_sig_thresh,
        resid_sigma_smooth_deg=args.sca_resid_sigma_smooth_deg,
        sca_fit_lat_min=args.sca_fit_lat_min,
        sca_domain_lat_min=sca_domain_lat_min,
        hilat_cut_deg=args.hilat_cut_deg,
        clip_counts_pctl=args.sca_clip_counts_pctl,
        outdir=args.outdir,
        wcs=wcs,
        debug_subtraction=args.debug_subtraction,
        debug_energy_gev=args.debug_energy_gev,
        nnls_weights_mode=args.sca_nnls_weights,
        smooth_terms=args.sca_smooth_terms,
        smooth_max_degree=args.sca_smooth_max_degree,
    )


    # --- High latitude only (|b| > 10 deg) ---
    hilat = connected_mask & (np.abs(lat) >= args.hilat_cut_deg) & roi2d

    if not np.any(hilat):
        print(f"[dbg] sig_sm max={float(np.nanmax(sig_sm)):.3f}, thresh={args.sca_sig_thresh}")
        print(f"[dbg] above pixels={int(np.sum(above_thresh))}, keep pixels={int(np.sum(connected_mask))}")
        print(
            f"[dbg] lat stats in ROI: min={float(np.nanmin(lat[roi2d])):.3f}, max={float(np.nanmax(lat[roi2d])):.3f}"
        )
        if np.any(connected_mask & roi2d):
            abslat_keep = np.abs(lat[connected_mask & roi2d])
            print(
                "[dbg] keep |lat| percentiles: "
                f"p50={float(np.nanpercentile(abslat_keep, 50)):.3f}, "
                f"p90={float(np.nanpercentile(abslat_keep, 90)):.3f}, "
                f"p99={float(np.nanpercentile(abslat_keep, 99)):.3f}"
            )
        print(f"[dbg] hilat_cut_deg={args.hilat_cut_deg} -> hilat pixels={int(np.sum(hilat))}")

        lat_abs = np.abs(lat[roi2d])
        if float(np.nanmax(lat_abs)) <= (np.pi + 0.1):
            print("[dbg] WARNING: lat values look like radians, not degrees.")
        raise RuntimeError("High-lat bubbles mask is empty; try lowering --sca-sig-thresh or check inputs.")

    # Spatial template (binary), normalized over its own support (like your other templates)
    T = np.zeros((ny, nx), float)
    T[hilat] = 1.0
    T[~roi2d] = 0.0

    norm_mask = hilat  # normalize over the high-lat bubble region
    s = float(np.nansum(T[norm_mask]))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("Normalization failed (template sum <= 0).")
    T /= s

    # Broadcast into energy cube in your repo conventions
    dnde = np.empty((nE, ny, nx), float)
    for k in range(nE):
        dnde[k] = T / (omega * dE_mev[k])

    E2dnde = dnde * (Ectr_mev[:, None, None] ** 2)
    mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]

    # --- Write outputs ---
    out_mask = os.path.join(args.outdir, "bubbles_hilat_mask.fits")
    out_mu   = os.path.join(args.outdir, "mu_bubbles_hilat_counts.fits")
    out_dnde = os.path.join(args.outdir, "bubbles_hilat_dnde.fits")
    out_e2   = os.path.join(args.outdir, "bubbles_hilat_E2dnde.fits")

    out_sig  = os.path.join(args.outdir, "debug_sca_hard_sig_smoothed.fits")
    out_conn = os.path.join(args.outdir, "debug_sca_threshold_connected.fits")

    write_primary_with_bunit(
        out_mask, T, hdr, "dimensionless",
        comments=[
            "Fermi bubbles template (HIGH LATITUDE ONLY) from SCA hard-component significance thresholding.",
            f"SCA ROI: |l|<{args.sca_roi_lon} deg, |b|<{args.sca_roi_lat} deg; high-lat cut |b|>={args.hilat_cut_deg} deg",
            f"Hard/soft indices: {args.sca_hard_index}/{args.sca_soft_index}; sig smooth {args.sca_sig_smooth_deg} deg; thresh {args.sca_sig_thresh} sigma",
            "Normalized so sum(template over high-lat bubble region) = 1.",
        ],
    )

    write_primary_with_bunit(
        out_mu, mu, hdr, "counts",
        comments=[
            "Expected counts for high-lat Fermi bubbles spatial template (spectrum fitted elsewhere).",
        ],
    )
    write_primary_with_bunit(
        out_dnde, dnde, hdr, "ph cm-2 s-1 sr-1 MeV-1",
        comments=["High-lat Fermi bubbles spatial template as dN/dE cube (repo convention)."],
    )
    write_primary_with_bunit(
        out_e2, E2dnde, hdr, "MeV cm-2 s-1 sr-1",
        comments=["High-lat Fermi bubbles spatial template as E^2 dN/dE cube (repo convention)."],
    )

    write_primary_with_bunit(
        out_sig, sig_sm, hdr, "sigma",
        comments=["Debug: smoothed hard-component significance map used for thresholding."],
    )
    write_primary_with_bunit(
        out_conn, connected_mask.astype(np.float32), hdr, "dimensionless",
        comments=["Debug: connected hard-component mask after thresholding (before |b| split)."],
    )

    # Debug text
    with open(os.path.join(args.outdir, "debug_bubbles_sca_hilat.txt"), "w") as f:
        f.write("# SCA high-lat bubbles template build\n")
        f.write(f"sca_roi_lon {args.sca_roi_lon}\n")
        f.write(f"sca_roi_lat {args.sca_roi_lat}\n")
        f.write(f"sca_e_min_gev {args.sca_e_min_gev}\n")
        f.write(f"sca_e_max_gev {args.sca_e_max_gev}\n")
        f.write(f"hard_index {args.sca_hard_index}\n")
        f.write(f"soft_index {args.sca_soft_index}\n")
        f.write(f"sig_smooth_deg {args.sca_sig_smooth_deg}\n")
        f.write(f"sig_thresh {args.sca_sig_thresh}\n")
        f.write(f"hilat_cut_deg {args.hilat_cut_deg}\n")
        f.write(f"template_sum_normregion {s}\n")
        f.write(f"hilat_pixels {int(np.sum(hilat))}\n")

    print("✓ Wrote:")
    print(" ", out_mask)
    print(" ", out_mu)
    print(" ", out_dnde)
    print(" ", out_e2)
    print(" ", out_sig)
    print(" ", out_conn)

    if args.plot:
        dom = roi2d & (np.abs(lat) >= sca_domain_lat_min)
        Hdom = Hmap[dom]
        Hdom = Hdom[np.isfinite(Hdom)]
        Sdom = Smap[dom]
        Sdom = Sdom[np.isfinite(Sdom)]

        def _sym_vmax(x, q=99.0, fallback=1.0):
            if x.size == 0:
                return fallback
            v = float(np.nanpercentile(np.abs(x), q))
            if not np.isfinite(v) or v <= 0:
                return fallback
            return v

        hv = _sym_vmax(Hdom, q=99.0, fallback=1.0)
        sv = _sym_vmax(Sdom, q=99.0, fallback=1.0)

        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(2, 2, 1, projection=wcs)
        im1 = ax1.imshow(sig_sm, origin="lower", vmin=0.0, vmax=10.0)
        ax1.set_title("SCA hard significance (smoothed)")
        ax1.set_xlabel("l"); ax1.set_ylabel("b")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="sigma")

        ax2 = plt.subplot(2, 2, 2, projection=wcs)
        im2 = ax2.imshow(Hmap, origin="lower", vmin=-hv, vmax=hv)
        ax2.set_title("SCA hard component H (at 1 GeV)")
        ax2.set_xlabel("l"); ax2.set_ylabel("b")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(2, 2, 3, projection=wcs)
        im3 = ax3.imshow(Smap, origin="lower", vmin=-sv, vmax=sv)
        ax3.set_title("SCA soft component S (at 1 GeV)")
        ax3.set_xlabel("l"); ax3.set_ylabel("b")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = plt.subplot(2, 2, 4, projection=wcs)
        im4 = ax4.imshow(hilat.astype(float), origin="lower", vmin=0, vmax=1)
        ax4.set_title(f"High-lat bubbles mask |b|>{args.hilat_cut_deg}°")
        ax4.set_xlabel("l"); ax4.set_ylabel("b")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="mask")

        out_png = os.path.join(args.outdir, "bubbles_sca_hilat_diagnostics.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(" ", out_png)


if __name__ == "__main__":
    main()
