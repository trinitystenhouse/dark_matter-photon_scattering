#!/usr/bin/env python3
"""
Sanity-check NFW template FITS products:
  - nfw*_dnde.fits
  - nfw*_E2dnde.fits
  - mu_nfw*_counts.fits

Checks:
  1) shapes match, finite values, non-negative
  2) ROI masking behaviour (outside ROI ~ 0)
  3) normalization: sum ROI pixels of spatial map == 1 (reconstructed from dnde)
  4) consistency: E2dnde == dnde * E^2
  5) consistency: mu == dnde * expo * omega * dE  (if expo provided)
  6) symmetry-ish: I(l,b) vs I(-l,b) and vs I(l,-b) inside ROI (approx)
  7) optional radial monotonicity: azimuthal average vs psi from GC decreases

Usage:
  python check_nfw_outputs.py --dnde path/to/nfw_dnde.fits --e2 path/to/nfw_E2dnde.fits --mu path/to/mu_nfw_counts.fits \
      --expo path/to/expcube.fits --binsz 0.125 --roi-lon 60 --roi-lat 60

Notes:
- This script reconstructs the spatial template from dnde by: spatial ~ dnde[k]*omega*dE / Phi_bin[k]
  It assumes your Phi_bin is flat (1/nE), as in your builder.
"""

import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import inspect
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from totani_helpers.totani_io import (
    pixel_solid_angle_map,
    read_expcube_energies_mev,
    resample_exposure_logE_interp,
)


# -------------------------
# Helpers
# -------------------------

def _finite_stats(x, name):
    x = np.asarray(x)
    fin = np.isfinite(x)
    if fin.any():
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        mean = float(np.nanmean(x))
    else:
        mn = mx = mean = float("nan")
    return {
        "name": name,
        "shape": tuple(x.shape),
        "finite_frac": float(fin.mean()),
        "min": mn,
        "max": mx,
        "mean": mean,
        "neg_frac": float((x < 0).mean()) if x.size else float("nan"),
        "nan_frac": float(np.isnan(x).mean()) if x.size else float("nan"),
    }

def _print_stats(stats):
    print(f"[{stats['name']}] shape={stats['shape']} finite={stats['finite_frac']:.6f} "
          f"nan={stats['nan_frac']:.6f} neg={stats['neg_frac']:.6f} "
          f"min={stats['min']:.6e} max={stats['max']:.6e} mean={stats['mean']:.6e}")

def read_ebounds_from_counts_ccube(hdul):
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

def _roi_mask_from_wcs(wcs_cel, ny, nx, roi_lon, roi_lat):
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs_cel.pixel_to_world_values(xx, yy)
    lon = ((lon + 180.0) % 360.0) - 180.0
    roi = (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)
    return roi, lon, lat

def _azimuthal_profile(map2d, lon_deg, lat_deg, psi_bins_deg):
    """
    Azimuthal average in bins of psi = angular separation from (0,0) (GC).
    Uses spherical law of cosines: cos psi = cos b cos l (with l,b in rad).
    """
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)
    cospsi = np.cos(b) * np.cos(l)
    cospsi = np.clip(cospsi, -1.0, 1.0)
    psi = np.rad2deg(np.arccos(cospsi))

    prof = np.full(len(psi_bins_deg) - 1, np.nan, float)
    for i in range(len(prof)):
        m = (psi >= psi_bins_deg[i]) & (psi < psi_bins_deg[i + 1]) & np.isfinite(map2d)
        if np.any(m):
            prof[i] = np.nanmean(map2d[m])
    return prof


def read_coefficients(coeff_file, component_name):
    """
    Read coefficients from fitting output file.
    Returns: (Ectr_gev, coefficients) arrays
    """
    Ectr_gev = []
    coeffs = []

    with open(coeff_file, "r") as f:
        header = f.readline().strip()
        # Header usually looks like:
        #   "# k  Ectr(GeV)  GAS  ICS  ISO  PS  LOOPI  FB_FLAT  NFW"
        # Strip leading comment marker if present.
        if header.startswith("#"):
            header = header.lstrip("#").strip()

        cols = header.split()
        if len(cols) < 3:
            raise ValueError(f"Malformed coefficient header: '{header}'")

        def _norm(s):
            return "".join(ch for ch in str(s).lower() if ch.isalnum())

        comp_req = _norm(component_name)

        # Data lines: k, Ectr(GeV), then one column per component.
        # Support case-insensitive and "contains" matching (e.g. NFW matches nfw_rho2.5_g1.25)
        norm_cols = [_norm(c) for c in cols]
        if comp_req in norm_cols:
            comp_col = norm_cols.index(comp_req)
        else:
            hits = [i for i, nc in enumerate(norm_cols) if (comp_req in nc)]
            if len(hits) == 1:
                comp_col = hits[0]
            else:
                raise ValueError(
                    f"Component '{component_name}' not found in coefficient file. Available: {cols[2:]}"
                )

        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= comp_col:
                continue
            Ectr_gev.append(float(parts[1]))
            coeffs.append(float(parts[comp_col]))

    return np.array(Ectr_gev), np.array(coeffs)


def plot_scaled_flux(
    Ectr_gev,
    coeffs,
    mu_counts,
    expo,
    omega,
    dE_mev,
    roi2d,
    plot_dir,
    component_name,
    coeff_mode="multiplier",
    show=False,
):
    """Plot the scaled flux from counts-space template.

    We compute ROI-averaged intensity per bin using the same estimator as the fitter:
      I_k = sum( model_counts ) / sum( expo * omega * dE )
      E2_k = I_k * Ectr^2

    where model_counts = coeff_k * mu_counts[k].
    """
    Ectr_gev = np.asarray(Ectr_gev, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)

    nE = int(mu_counts.shape[0])
    if coeffs.size != nE:
        raise ValueError(f"coeff length {coeffs.size} != template nE {nE}")

    Ectr_mev = Ectr_gev * 1000.0

    E2_template = np.full(nE, np.nan, dtype=float)
    E2_scaled = np.full(nE, np.nan, dtype=float)
    for k in range(nE):
        m = roi2d & np.isfinite(mu_counts[k])
        if not np.any(m):
            continue
        denom = float(np.nansum((expo[k] * omega * dE_mev[k])[m]))
        if not np.isfinite(denom) or denom <= 0:
            continue
        C_tpl = float(np.nansum(mu_counts[k][m]))
        E2_template[k] = (C_tpl / denom) * (Ectr_mev[k] ** 2)

        if coeff_mode == "multiplier":
            # coeff is dimensionless multiplier: model_counts = coeff * mu
            C_scaled = float(coeffs[k]) * C_tpl
        elif coeff_mode == "counts":
            # coeff is total counts attributed to component in ROI for this bin.
            C_scaled = float(coeffs[k])
        else:
            raise ValueError(f"Unknown coeff_mode='{coeff_mode}'")

        E2_scaled[k] = (C_scaled / denom) * (Ectr_mev[k] ** 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Top panel: Coefficients
    ax1.plot(Ectr_gev, coeffs, 'o-', lw=2, label='Fit coefficients')
    ax1.axhline(1.0, color='k', ls='--', alpha=0.3, label='Coefficient = 1')
    ax1.set_xscale('log')
    ax1.set_xlabel('Energy [GeV]')
    ax1.set_ylabel('Fit Coefficient')
    ax1.set_title(f'{component_name}: Fit Coefficients vs Energy')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    
    # Bottom panel: Scaled flux
    ax2.plot(Ectr_gev, E2_scaled, "o-", lw=2, label="Scaled flux (coeff x template)")
    ax2.plot(Ectr_gev, E2_template, "s--", lw=1.5, alpha=0.6, label="Template (unscaled)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Energy [GeV]')
    ax2.set_ylabel('E² dN/dE [MeV cm⁻² s⁻¹ sr⁻¹]')
    ax2.set_title(f'{component_name}: Scaled Flux (ROI-averaged)')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    outpath = os.path.join(plot_dir, f"{component_name.lower()}_scaled_flux.png")
    fig.savefig(outpath, dpi=200)
    print(f"✓ Saved scaled flux plot: {outpath}")
    if show:
        plt.show()
    plt.close(fig)


def plot_scaled_spatial_maps(
    *,
    k,
    wcs,
    roi2d,
    coeff_k,
    mu_k,
    expo_k,
    omega,
    dE_mev_k,
    Ectr_mev_k,
    plot_dir,
    component_name,
    coeff_mode="multiplier",
    map_norm="log",
    show=False,
):
    """Plot scaled spatial E^2 dN/dE map for one energy bin."""
    denom = expo_k * omega * dE_mev_k
    E2_map = np.full_like(mu_k, np.nan, dtype=float)
    ok = np.isfinite(denom) & (denom > 0) & np.isfinite(mu_k)
    if np.any(ok):
        if coeff_mode == "multiplier":
            # model_counts_map = coeff * mu
            counts_map = float(coeff_k) * mu_k
        elif coeff_mode == "counts":
            # model_counts_map = C_tot * (mu / sum(mu in ROI))
            s = float(np.nansum(mu_k[roi2d & np.isfinite(mu_k)]))
            if not np.isfinite(s) or s <= 0:
                counts_map = np.full_like(mu_k, np.nan, dtype=float)
            else:
                counts_map = float(coeff_k) * (mu_k / s)
        else:
            raise ValueError(f"Unknown coeff_mode='{coeff_mode}'")

        I = counts_map[ok] / denom[ok]
        E2_map[ok] = I * (Ectr_mev_k ** 2)

    E2_plot = np.where(roi2d, E2_map, np.nan)

    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection=wcs)

    if map_norm == "log":
        finite = np.isfinite(E2_plot) & (E2_plot > 0)
        if np.any(finite):
            vmin = float(np.nanpercentile(E2_plot[finite], 5))
            vmax = float(np.nanpercentile(E2_plot[finite], 99.5))
            vmin = max(vmin, vmax * 1e-6) if np.isfinite(vmax) and vmax > 0 else vmin
            norm = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, max(vmin, 1e-30)))
        else:
            norm = None
        im = ax.imshow(E2_plot, origin="lower", cmap="magma", norm=norm)
    else:
        im = ax.imshow(E2_plot, origin="lower", cmap="magma")

    ax.set_title(f"{component_name} scaled E2 dN/dE (k={k:02d})")
    ax.set_xlabel("l")
    ax.set_ylabel("b")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$")
    fig.tight_layout()
    outpath = os.path.join(plot_dir, f"{component_name.lower()}_scaled_E2_map_k{k:02d}.png")
    fig.savefig(outpath, dpi=200)
    print(f"✓ Saved scaled spatial map: {outpath}")
    if show:
        plt.show()
    plt.close(fig)


# -------------------------
# Main checks
# -------------------------

def main():
    repo_dir = os.environ.get("REPO_PATH")
    data_dir = os.path.join(repo_dir, "fermi_data", "totani") if repo_dir else None
    default_counts = os.path.join(data_dir, "processed", "counts_ccube_1000to1000000.fits") if data_dir else None
    default_expo = os.path.join(data_dir, "processed", "expcube_1000to1000000.fits") if data_dir else None
    default_dnde = os.path.join(data_dir, "processed", "templates", "nfw_rho2.5_g1.25_pheno_dnde.fits") if data_dir else None
    default_e2 = os.path.join(data_dir, "processed", "templates", "nfw_rho2.5_g1.25_pheno_E2dnde.fits") if data_dir else None
    default_mu = os.path.join(data_dir, "processed", "templates", "mu_nfw_rho2.5_g1.25_pheno_counts.fits") if data_dir else None
    default_coeffs = os.path.join(repo_dir, "Totani_paper_check", "fig2_3", "fit_coefficients_fig2_highlat.txt")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dnde", default=default_dnde, help="nfw*_dnde.fits")
    ap.add_argument("--e2", default=default_e2, help="nfw*_E2dnde.fits")
    ap.add_argument("--mu", default=default_mu, help="mu_nfw*_counts.fits")
    ap.add_argument("--counts", default=default_counts, help="counts CCUBE used to build templates (has EBOUNDS + WCS)")
    ap.add_argument("--expo", default=default_expo, help="expcube used to build mu (optional but recommended)")
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--eps", type=float, default=1e-6, help="tolerance for consistency checks (relative)")
    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument(
        "--plot-dir",
        default=os.path.join(os.path.dirname(__file__), "plots_check_nfw"),
        help="Directory to save diagnostic plots",
    )
    ap.add_argument("--plot-k", type=int, default=None, help="Energy-bin index to use for map plots (default: middle bin)")
    ap.add_argument("--plot-lonwin", type=float, default=5.0, help="Longitude half-width (deg) for latitude profile I(b)")
    ap.add_argument("--map-norm", choices=["linear", "log"], default="log", help="Map color normalization")
    ap.add_argument("--show", action="store_true", help="Show plots interactively (in addition to saving)")
    ap.add_argument("--coeff-file", default=default_coeffs, help="Coefficient file from fitting (e.g., fit_coefficients_fig2_highlat.txt)")
    ap.add_argument("--component-name", default="NFW", help="Component name in coefficient file (default: NFW)")
    ap.add_argument(
        "--coeff-mode",
        choices=["multiplier", "counts"],
        default="counts",
        help="How to interpret the coefficient column: 'multiplier' means model=coeff*mu; 'counts' means coeff is total counts in ROI per bin.",
    )

    args = ap.parse_args()

    for name in ["counts", "dnde", "e2", "mu"]:
        p = getattr(args, name)
        if p is None or not os.path.exists(str(p)):
            raise SystemExit(
                f"Missing required file --{name}. Either set REPO_PATH so defaults resolve, or pass --{name} explicitly."
            )

    if args.expo is not None and (not os.path.exists(str(args.expo))):
        raise SystemExit(f"Exposure file not found: {args.expo}")

    # --- read energies and WCS from dnde header (should match counts header) ---
    # Read WCS + image shape from COUNTS ccube (authoritative)
    with fits.open(args.counts) as hc:
        hdr = hc[0].header
        wcs = WCS(hdr).celestial
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
        Emin_mev, Emax_mev, Ectr_mev, dE_mev = read_ebounds_from_counts_ccube(hc)
    nE = len(Ectr_mev)


    # --- load arrays ---
    with fits.open(args.dnde) as h:
        dnde = np.array(h[0].data, dtype=float)
    with fits.open(args.e2) as h:
        e2 = np.array(h[0].data, dtype=float)
    with fits.open(args.mu) as h:
        mu = np.array(h[0].data, dtype=float)

    print("== Basic stats ==")
    _print_stats(_finite_stats(dnde, "dnde"))
    _print_stats(_finite_stats(e2, "E2dnde"))
    _print_stats(_finite_stats(mu, "mu_counts"))

    # --- shape checks ---
    assert dnde.shape == (nE, ny, nx), f"dnde shape {dnde.shape} != {(nE, ny, nx)}"
    assert e2.shape == (nE, ny, nx), f"e2 shape {e2.shape} != {(nE, ny, nx)}"
    assert mu.shape == (nE, ny, nx), f"mu shape {mu.shape} != {(nE, ny, nx)}"
    print("✓ Shapes match expected (nE, ny, nx)")

    # --- non-negativity (allow tiny negative from numerics) ---
    for name, arr in [("dnde", dnde), ("E2dnde", e2), ("mu", mu)]:
        minv = float(np.nanmin(arr))
        if minv < -1e-12:
            print(f"⚠ {name} has significantly negative values: min={minv:.3e}")
        else:
            print(f"✓ {name} non-negative (within tolerance)")

    # --- ROI mask + outside ROI should be ~0 for all k (since spatial was masked then normalised) ---
    roi, lon_deg, lat_deg = _roi_mask_from_wcs(wcs, ny, nx, args.roi_lon, args.roi_lat)

    outside = ~roi
    outside_level = float(np.nanmax(np.abs(dnde[:, outside]))) if np.any(outside) else 0.0
    print(f"[ROI] max |dnde| outside ROI = {outside_level:.3e}")
    if outside_level > 0:
        # allow tiny numerical noise
        inside_peak = float(np.nanmax(dnde[:, roi])) if np.any(roi) else 1.0
        if inside_peak > 0 and outside_level / inside_peak > 1e-6:
            print("⚠ significant leakage outside ROI (check masking / WCS / ROI cut)")
        else:
            print("✓ Outside ROI is ~0 (relative)")

    # --- E2 consistency: E2dnde == dnde * E^2 ---
    pred_e2 = dnde * (Ectr_mev[:, None, None] ** 2)
    denom = np.nanmax(np.abs(e2))
    rel = float(np.nanmax(np.abs(e2 - pred_e2)) / (denom if denom > 0 else 1.0))
    print(f"[CONSIST] max rel |E2dnde - dnde*E^2| = {rel:.3e}")
    if rel > args.eps:
        print("⚠ E2dnde mismatch (check Ectr units / which energy used)")
    else:
        print("✓ E2dnde matches dnde * E^2")

    # --- reconstruct spatial template from dnde for one representative bin ---
    # You built with Phi_bin flat => Phi_bin[k] = 1/nE
    phi_k = 1.0 / nE
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    

    if args.plot_k is None:
        k0 = int(np.clip(nE // 2, 0, nE - 1))
    else:
        k0 = int(np.clip(args.plot_k, 0, nE - 1))
    spatial_rec = dnde[k0] * omega * dE_mev[k0] / phi_k  # should be nfw_spatial (unit-sum over ROI pixels)
    spatial_rec[~roi] = 0.0

    spatial_sum = float(np.nansum(spatial_rec))
    print(f"[NORM] reconstructed spatial sum over ROI pixels (k={k0}) = {spatial_sum:.8f} (expected ~1)")
    if not np.isfinite(spatial_sum) or abs(spatial_sum - 1.0) > 5e-3:
        print("⚠ spatial normalisation not ~1. This can happen if omega estimate differs from your builder.")
        print("   If you want an exact check, import and use the *same* pixel_solid_angle_map() as in the builder.")
    else:
        print("✓ spatial normalisation looks OK")

    # --- if exposure provided: check mu consistency ---
    if args.expo is not None:
        with fits.open(args.expo) as he:
            expo_raw = np.array(he[0].data, dtype=np.float64)
            E_expo_mev = read_expcube_energies_mev(he)

        expo = resample_exposure_logE_interp(expo_raw, E_expo_mev, Ectr_mev)

        print("[DBG] resampler:", resample_exposure_logE_interp.__name__)
        print("[DBG] resampler file:", inspect.getsourcefile(resample_exposure_logE_interp))
        print("[DBG] expo per-bin sum:", np.nansum(expo, axis=(1, 2)))

        pred_mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
        # Per-bin relative mismatch (L1-weighted is robust)
        num = np.nansum(np.abs(mu - pred_mu), axis=(1, 2))
        den = np.nansum(np.abs(mu), axis=(1, 2)) + 1e-30
        rel_per_bin = num / den

        print("[CONSIST] rel mismatch per energy bin:")
        for k, r in enumerate(rel_per_bin):
            print(f"  k={k:02d}  Ectr={Ectr_mev[k]:.1f} MeV  rel={r:.4f}")

        # Also check if it's basically a single global scale factor:
        a = np.nansum(mu * pred_mu) / (np.nansum(pred_mu * pred_mu) + 1e-30)
        rel_after_scale = np.nanmax(np.abs(mu - a * pred_mu)) / (np.nanmax(np.abs(mu)) + 1e-30)
        print(f"[CONSIST] best-fit global scale a={a:.6g}, max-rel after scaling={rel_after_scale:.4e}")

        denom = np.nanmax(np.abs(mu))
        rel_mu = float(np.nanmax(np.abs(mu - pred_mu)) / (denom if denom > 0 else 1.0))
        print(f"[CONSIST] max rel |mu - dnde*expo*omega*dE| = {rel_mu:.3e}")
        if rel_mu > 5e-3:
            print("⚠ mu mismatch. Most common causes: different omega calc, different exposure resampling, or dtype issues.")
        else:
            print("✓ mu matches dnde*expo*omega*dE (within tolerance)")
        
        mu_bin = np.nansum(mu, axis=(1,2))
        pred_bin = np.nansum(pred_mu, axis=(1,2))
        ratio = mu_bin / (pred_bin + 1e-30)

        print("[CONSIST] sum(mu) per bin vs sum(pred_mu) per bin:")
        for k in range(nE):
            print(f"  k={k:02d} Ectr={Ectr_mev[k]:.1f} MeV  mu={mu_bin[k]:.6e}  pred={pred_bin[k]:.6e}  ratio={ratio[k]:.4f}")

        print("[DBG] Ectr_mev:", Ectr_mev)
        print("[DBG] E_expo_mev min/max:", np.min(E_expo_mev), np.max(E_expo_mev))
        print("[DBG] E_expo_mev first/last:", E_expo_mev[0], E_expo_mev[-1])

        # After resampling:
        print("[DBG] expo_resampled per-bin sum:", np.nansum(expo, axis=(1,2)))

        # Effective exposure implied by mu and dnde
        expo_eff = np.full_like(mu, np.nan, dtype=float)

        den = dnde * omega[None,:,:] * dE_mev[:,None,None]
        m = (den > 0) & np.isfinite(den) & np.isfinite(mu)

        expo_eff[m] = mu[m] / den[m]

        # Compare per bin inside ROI
        print("[EXPO] expo_eff / expo stats per bin (ROI):")
        for k in range(nE):
            mk = roi & np.isfinite(expo_eff[k]) & np.isfinite(expo[k]) & (expo[k] > 0)
            r = expo_eff[k][mk] / expo[k][mk]
            med = np.nanmedian(r)
            p16 = np.nanpercentile(r, 16)
            p84 = np.nanpercentile(r, 84)
            print(f"  k={k:02d} Ectr={Ectr_mev[k]:.1f} MeV  median={med:.4f}  (16–84%: {p16:.4f}–{p84:.4f})")

        if args.plot_dir is not None:
            os.makedirs(args.plot_dir, exist_ok=True)
            out_base = os.path.splitext(os.path.basename(args.dnde))[0]

            fig = plt.figure(figsize=(6.8, 4.8))
            ax = fig.add_subplot(111)

            pos = spatial_rec[np.isfinite(spatial_rec) & (spatial_rec > 0)]
            if pos.size:
                vmin = float(np.nanpercentile(pos, 1))
                vmax = float(np.nanpercentile(pos, 99.9))
                if not np.isfinite(vmin) or vmin <= 0:
                    vmin = float(np.nanmin(pos))
                if not np.isfinite(vmax) or vmax <= vmin:
                    vmax = float(np.nanmax(pos))
            else:
                vmin, vmax = 1e-30, 1.0

            norm = None
            if args.map_norm == "log":
                norm = LogNorm(vmin=vmin, vmax=vmax)

            im = ax.imshow(
                spatial_rec,
                origin="lower",
                cmap="magma",
                norm=norm,
                vmin=(None if norm is not None else 0.0),
            )
            ax.set_title(f"Reconstructed spatial (k={k0}, Ectr={Ectr_mev[k0]/1000.0:.3g} GeV)")
            ax.set_xlabel("pixel x")
            ax.set_ylabel("pixel y")
            fig.colorbar(im, ax=ax, label="arb.")
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_spatial_k{k0:02d}.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)

            psi_bins = np.linspace(0.0, min(args.roi_lon, args.roi_lat), 41)
            prof = _azimuthal_profile(spatial_rec, lon_deg, lat_deg, psi_bins)
            psi_ctr = 0.5 * (psi_bins[:-1] + psi_bins[1:])

            fig = plt.figure(figsize=(6.8, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(psi_ctr, prof, lw=2.0)
            ax.set_xlabel(r"$\psi$ from GC  [deg]")
            ax.set_ylabel("mean intensity (arb.)")
            ax.set_title("Azimuthal mean profile")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_profile_psi.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)

            mlat = roi & (np.abs(lon_deg) <= float(args.plot_lonwin))
            lat_bins = np.linspace(-args.roi_lat, args.roi_lat, 81)
            lat_ctr = 0.5 * (lat_bins[:-1] + lat_bins[1:])
            prof_b = np.full(len(lat_ctr), np.nan, float)
            for i in range(len(lat_ctr)):
                mm = mlat & (lat_deg >= lat_bins[i]) & (lat_deg < lat_bins[i + 1])
                if np.any(mm):
                    prof_b[i] = np.nanmean(spatial_rec[mm])

            fig = plt.figure(figsize=(6.8, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(lat_ctr, prof_b, lw=2.0)
            ax.set_xlabel("b  [deg]")
            ax.set_ylabel("mean intensity (arb.)")
            ax.set_title(f"Latitude profile (|l|<={args.plot_lonwin:g} deg)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_profile_lat.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)

            fig = plt.figure(figsize=(6.8, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(Ectr_mev / 1000.0, rel_per_bin, lw=2.0)
            ax.set_xscale("log")
            ax.set_xlabel("Ectr  [GeV]")
            ax.set_ylabel("rel L1 mismatch per bin")
            ax.set_title(r"$||\mu-\mu_{pred}||_1 / ||\mu||_1$")
            ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"{out_base}_mu_mismatch.png"), dpi=200)
            if args.show:
                plt.show()
            plt.close(fig)


    # --- symmetry checks on reconstructed spatial (inside ROI only) ---
    # These are approximate because your lon grid may not be perfectly symmetric, but it’s still a useful smoke test.
    # We'll compare to flipped arrays.
    if spatial_rec.shape == (ny, nx):
        lr = np.nanmax(np.abs(spatial_rec - spatial_rec[:, ::-1]))
        ns = np.nanmax(np.abs(spatial_rec - spatial_rec[::-1, :]))
        peak = float(np.nanmax(spatial_rec)) if np.any(roi) else 0.0
        print(f"[SYM] max |I - I_flip_lon| = {lr:.3e}  (rel {lr/(peak if peak>0 else 1):.3e})")
        print(f"[SYM] max |I - I_flip_lat| = {ns:.3e}  (rel {ns/(peak if peak>0 else 1):.3e})")

    # --- radial monotonicity (azimuthal profile vs psi) ---
    psi_bins = np.linspace(0.0, min(args.roi_lon, args.roi_lat), 31)  # 30 bins out to ROI edge-ish
    prof = _azimuthal_profile(spatial_rec, lon_deg, lat_deg, psi_bins)
    # check if it mostly decreases (allow some noise)
    diffs = np.diff(prof)
    frac_increasing = float(np.mean(diffs > 0)) if np.isfinite(diffs).any() else float("nan")
    print(f"[RADIAL] frac of bins where profile increases outward = {frac_increasing:.3f} (expect small)")

    # --- Plot scaled flux if coefficient file provided ---
    if args.coeff_file is not None:
        if not os.path.exists(args.coeff_file):
            print(f"\n⚠ Coefficient file not found: {args.coeff_file}")
        elif args.expo is None:
            print("\n⚠ --expo is required to compute scaled E2 dN/dE from counts-space templates.")
        else:
            print(f"\n[SCALED FLUX] Reading coefficients from: {args.coeff_file}")
            try:
                Ectr_gev_coeff, coeffs = read_coefficients(args.coeff_file, args.component_name)
                print(f"[SCALED FLUX] Component: {args.component_name}")
                print(f"[SCALED FLUX] Coefficient range: {np.nanmin(coeffs):.3e} to {np.nanmax(coeffs):.3e}")
                plot_scaled_flux(
                    Ectr_gev_coeff,
                    coeffs,
                    mu,
                    expo,
                    omega,
                    dE_mev,
                    roi,
                    args.plot_dir,
                    args.component_name,
                    coeff_mode=args.coeff_mode,
                    show=args.show,
                )

                # Spatial map (scaled E2 dN/dE) for one representative energy bin
                k_map = args.plot_k
                if k_map is None:
                    k_map = int(nE // 2)
                k_map = int(np.clip(k_map, 0, nE - 1))
                plot_scaled_spatial_maps(
                    k=k_map,
                    wcs=wcs,
                    roi2d=roi,
                    coeff_k=coeffs[k_map],
                    mu_k=mu[k_map],
                    expo_k=expo[k_map],
                    omega=omega,
                    dE_mev_k=float(dE_mev[k_map]),
                    Ectr_mev_k=float(Ectr_mev[k_map]),
                    plot_dir=args.plot_dir,
                    component_name=args.component_name,
                    coeff_mode=args.coeff_mode,
                    map_norm=args.map_norm,
                    show=args.show,
                )
            except Exception as e:
                print(f"⚠ Error plotting scaled flux: {e}")

    print("\n✓ Done. If anything flagged ⚠, paste the relevant lines and I'll tell you exactly what to change.")

if __name__ == "__main__":
    main()
