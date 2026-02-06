#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

# Optional NNLS (recommended)
try:
    from scipy.optimize import nnls
    HAVE_NNLS = True
except Exception:
    HAVE_NNLS = False


# -------------------------
# IO + geometry helpers
# -------------------------
def read_counts_and_ebounds(counts_path):
    with fits.open(counts_path) as h:
        counts = h[0].data.astype(float)   # (nE, ny, nx)
        hdr = h[0].header
        eb  = h["EBOUNDS"].data

    Emin = eb["E_MIN"].astype(float) / 1000 # MeV
    Emax = eb["E_MAX"].astype(float) / 1000 # MeV
    Ectr = np.sqrt(Emin * Emax)       # MeV
    dE   = (Emax - Emin)              # MeV
    return counts, hdr, Emin, Emax, Ectr, dE


def read_exposure(expo_path):
    with fits.open(expo_path) as h:
        expo_raw = h[0].data.astype(float)  # (Ne, ny, nx)
        E_expo = None
        if "ENERGIES" in h:
            col0 = h["ENERGIES"].columns.names[0]
            E_expo = np.array(h["ENERGIES"].data[col0], dtype=float)  # MeV
    return expo_raw, E_expo


def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    """Log-energy linear interpolation onto target energies."""
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


def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    """Ω_pix ≈ Δl Δb cos(b) for CAR (matches your template convention)."""
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), float)  # sr


def lonlat_grids(wcs, ny, nx):
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon = ((lon + 180) % 360) - 180
    return lon, lat


def build_roi_box_mask(lon, lat, roi_lon=60.0, roi_lat=60.0):
    return (np.abs(lon) <= roi_lon) & (np.abs(lat) <= roi_lat)


def load_mask_any_shape(mask_path, nE, ny, nx):
    m = fits.getdata(mask_path).astype(bool)
    if m.shape == (nE, ny, nx):
        return m
    if m.shape == (ny, nx):
        return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
    raise RuntimeError(f"Mask shape {m.shape} incompatible with {(nE, ny, nx)}")


def bunit_str(hdr):
    return str(hdr.get("BUNIT", "")).lower().replace("**", "")


# -------------------------
# Template conversion
# -------------------------
def template_to_mu_and_E2(template, bunit, Ectr_mev, dE_mev, expo, omega):
    """
    Returns:
      mu_counts : predicted counts per pixel per bin
      E2cube    : E^2 dN/dE [MeV cm^-2 s^-1 sr^-1] per pixel per bin

    Accepts:
      - counts (mu_*_counts.fits)
      - dN/dE  [ph cm^-2 s^-1 sr^-1 MeV^-1]
      - E^2 dN/dE [MeV cm^-2 s^-1 sr^-1]
    """
    bu = (bunit or "").lower().replace("**", "")
    T = template.astype(float)

    # Counts template
    if "counts" in bu:
        mu = T
        denom = expo * omega[None, :, :] * dE_mev[:, None, None]
        dnde = np.full_like(mu, np.nan, float)
        ok = denom > 0
        dnde[ok] = mu[ok] / denom[ok]
        E2 = dnde * (Ectr_mev[:, None, None] ** 2)
        return mu, E2

    # E^2 dN/dE template
    looks_E2 = ("mev" in bu) and ("sr-1" in bu) and ("mev-1" not in bu) and ("ph" not in bu)
    if looks_E2:
        E2 = T
        dnde = E2 / (Ectr_mev[:, None, None] ** 2)
        mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
        return mu, E2

    # dN/dE template (or unknown -> assume dN/dE)
    dnde = T
    E2 = dnde * (Ectr_mev[:, None, None] ** 2)
    mu = dnde * expo * omega[None, :, :] * dE_mev[:, None, None]
    return mu, E2


# -------------------------
# Spectra + fitting
# -------------------------
def data_E2_spectrum(counts, expo, omega, dE_mev, Ectr_mev, mask3d):
    """
    Exposure-weighted estimator:
      I_k = sum C / sum(expo * omega * dE)
    then E^2 I_k and Poisson error.
    """
    nE = counts.shape[0]
    y = np.full(nE, np.nan)
    yerr = np.full(nE, np.nan)
    for k in range(nE):
        m = mask3d[k]
        C = np.nansum(counts[k][m])
        D = np.nansum((expo[k] * omega * dE_mev[k])[m])
        if D <= 0:
            continue
        I = C / D
        Ierr = np.sqrt(C) / D if C > 0 else 0.0
        y[k] = I * (Ectr_mev[k] ** 2)
        yerr[k] = Ierr * (Ectr_mev[k] ** 2)
    return y, yerr


def model_E2_spectrum_from_mu(coeff_cells, cells, mu_list, expo, omega, dE_mev, Ectr_mev, mask3d):
    """Compute E2 spectrum of fitted model (or any subset of components) using counts-space estimator.

    Returns E2 spectrum for each component and for the full model, computed as:
      I_k = sum(model_counts) / sum(expo*omega*dE)
      E2_k = I_k * Ectr^2
    """
    nE = mu_list[0].shape[0]
    ncomp = len(mu_list)

    y_comp = np.full((ncomp, nE), np.nan)
    y_model = np.full(nE, np.nan)

    for k in range(nE):
        m = mask3d[k]
        D = np.nansum((expo[k] * omega * dE_mev[k])[m])
        if D <= 0:
            continue

        Cj = np.zeros(ncomp, float)
        for ci, cell2d in enumerate(cells):
            cm = m & cell2d
            if not np.any(cm):
                continue
            for j in range(ncomp):
                Cj[j] += np.nansum((coeff_cells[ci, k, j] * mu_list[j][k])[cm])

        Ctot = float(np.sum(Cj))
        I = Ctot / D
        y_model[k] = I * (Ectr_mev[k] ** 2)
        for j in range(ncomp):
            Ij = Cj[j] / D
            y_comp[j, k] = Ij * (Ectr_mev[k] ** 2)

    return y_comp, y_model


def raw_counts_spectrum(counts, mask3d):
    nE = counts.shape[0]
    C = np.full(nE, np.nan)
    Cerr = np.full(nE, np.nan)
    for k in range(nE):
        ck = np.nansum(counts[k][mask3d[k]])
        C[k] = ck
        Cerr[k] = np.sqrt(ck) if ck >= 0 else np.nan
    return C, Cerr


def omega_weighted_mean(map2d, omega, mask2d):
    w = omega * mask2d
    wsum = np.nansum(w)
    if wsum <= 0:
        return np.nan
    return np.nansum(map2d * w) / wsum


def fit_per_bin_weighted_nnls(counts, mu_list, mask3d):
    """
    Per-energy-bin weighted NNLS:
      counts[k] ~ sum_j a_kj * mu_j[k]
    Returns coeff[nE, ncomp]
    """
    nE = counts.shape[0]
    ncomp = len(mu_list)
    coeff = np.zeros((nE, ncomp), float)

    for k in range(nE):
        m = mask3d[k]
        y = counts[k][m].ravel()
        if y.size == 0:
            continue

        X = np.vstack([mu[k][m].ravel() for mu in mu_list]).T  # (Np, ncomp)

        # Poisson-ish weights: downweight noisy high-count pixels slightly
        w = 1.0 / np.maximum(y, 1.0)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw

        if HAVE_NNLS:
            c, _ = nnls(Xw, yw)
        else:
            c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            c = np.clip(c, 0.0, None)

        coeff[k] = c

    return coeff


def fit_per_bin_weighted_nnls_cellwise(counts, mu_list, mask3d, lon, lat, roi_lon, roi_lat, cell_deg=10.0, weighting="uniform"):
    nE = counts.shape[0]
    ncomp = len(mu_list)

    l_edges = np.arange(-roi_lon, roi_lon + 1e-6, cell_deg)
    b_edges = np.arange(-roi_lat, roi_lat + 1e-6, cell_deg)

    # list of (cell_mask_2d, coeff_cell[nE,ncomp])
    cells = []
    for l0 in l_edges[:-1]:
        l1 = l0 + cell_deg
        in_l = (lon >= l0) & (lon < l1)
        for b0 in b_edges[:-1]:
            b1 = b0 + cell_deg
            cell2d = in_l & (lat >= b0) & (lat < b1)
            if not np.any(cell2d):
                continue
            cells.append(cell2d)

    coeff_cells = np.zeros((len(cells), nE, ncomp), float)

    for ci, cell2d in enumerate(cells):
        for k in range(nE):
            m2d = mask3d[k] & cell2d
            y = counts[k][m2d].ravel()
            if y.size == 0:
                continue

            X = np.vstack([mu[k][m2d].ravel() for mu in mu_list]).T

            if weighting == "uniform":
                Xw = X
                yw = y
            elif weighting == "poisson":
                # chi^2 approx with Var~y
                sw = 1.0 / np.sqrt(np.maximum(y, 1.0))
                Xw = X * sw[:, None]
                yw = y * sw
            else:
                raise ValueError(f"Unknown weighting='{weighting}'")

            if HAVE_NNLS:
                c, _ = nnls(Xw, yw)
            else:
                c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                c = np.clip(c, 0.0, None)

            coeff_cells[ci, k] = c

    return cells, coeff_cells


# -------------------------
# Template auto-discovery
# -------------------------
def pick_existing(template_dir, candidates):
    for name in candidates:
        p = os.path.join(template_dir, name)
        if os.path.exists(p):
            return p
    return None


def resolve_templates(template_dir):
    """
    Prefer counts templates. Fall back to intensity templates if needed.
    """
    spec = {}

    spec["IEM"] = pick_existing(template_dir, [
        "mu_iem_counts.fits",
        "mu_iem.fits",
        "mu_iem_counts.fits.gz",
        "template_iem_intensity.fits",
        "iem_dnde.fits",
        "iem_E2dnde.fits",
    ])

    spec["ISO"] = pick_existing(template_dir, [
        "mu_iso_counts.fits",
        "mu_iso.fits",
        "mu_iso_counts.fits.gz",
        "template_iso_intensity.fits",
        "iso_dnde.fits",
        "iso_E2dnde.fits",
    ])

    spec["PS"] = pick_existing(template_dir, [
        "mu_ps_counts.fits",
        "mu_ps.fits",
        "ps_dnde.fits",
        "ps_E2dnde.fits",
    ])

    spec["NFW"] = pick_existing(template_dir, [
        "mu_nfw_counts.fits",
        "mu_nfw.fits",
        "nfw_dnde.fits",
        "nfw_E2dnde.fits",
    ])

    spec["LOOPI"] = pick_existing(template_dir, [
        "mu_loopI_counts.fits",
        "mu_loopi_counts.fits",
        "loopI_dnde.fits",
        "loopI_E2dnde.fits",
    ])

    spec["BUB_POS"] = pick_existing(template_dir, [
        "mu_bubbles_pos_counts.fits",
        "bubbles_pos_dnde.fits",
        "bubbles_pos_E2dnde.fits",
    ])

    spec["BUB_NEG"] = pick_existing(template_dir, [
        "mu_bubbles_neg_counts.fits",
        "bubbles_neg_dnde.fits",
        "bubbles_neg_E2dnde.fits",
    ])

    spec["FB_FLAT"] = pick_existing(template_dir, [
        "mu_bubbles_flat_counts.fits",
        "bubbles_flat_dnde.fits",
        "bubbles_flat_E2dnde.fits",
    ])

    missing = [k for k, v in spec.items() if v is None]
    if missing:
        raise RuntimeError(
            "Missing templates for: " + ", ".join(missing) +
            f"\nLooked in: {template_dir}\n"
            "Expected files like mu_iem_counts.fits, mu_iso_counts.fits, mu_ps_counts.fits, mu_nfw_counts.fits, "
            "mu_loopI_counts.fits, mu_bubbles_flat_counts.fits"
        )
    return spec


# -------------------------
# Plot makers
# -------------------------
def plot_raw_counts(Ectr_gev, C, Cerr, outpath, title):
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(Ectr_gev, C, yerr=Cerr, fmt="o", capsize=2)
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Counts per energy bin")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("✓ wrote", outpath)


def plot_fit_fig2(Ectr_gev, data_y, data_yerr, comp_specs, outpath, title):
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for lab, y in comp_specs.items():
        ax.plot(Ectr_gev, y, marker="o", label=lab)

    ax.errorbar(Ectr_gev, data_y, yerr=data_yerr, fmt="o", capsize=2, label="data")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel(r"$E^2\,dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("✓ wrote", outpath)


# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Make BOTH: raw counts plot + Totani Fig2-style fitted component plot.")
    ap.add_argument(
        "--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"),
        help="counts_ccube_*.fits",
    )
    ap.add_argument(
        "--expo",
        default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"),
        help="expcube_*.fits",
    )
    ap.add_argument(
        "--templates-dir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
        help="directory containing templates (mu_*_counts.fits etc)",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig2_3"),
    )
    ap.add_argument("--binsz", type=float, default=0.125)

    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--cell-deg", type=float, default=10.0)
    ap.add_argument(
        "--weighting",
        default="uniform",
        choices=["uniform", "poisson"],
        help="NNLS weighting scheme. 'uniform' is unweighted; 'poisson' uses 1/sqrt(counts) weights.",
    )

    # For the FIT plot (you can change default to 0 if you want no disk cut)
    ap.add_argument("--disk-cut-fit", type=float, default=10.0, help="disk cut |b|>=X applied ONLY in fit plot")
    ap.add_argument("--mask-fit", default=None, help="optional mask FITS for fit plot (2D or 3D), True=keep")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load counts + EBOUNDS
    counts, hdr, Emin, Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    wcs = WCS(hdr).celestial

    # Exposure
    expo_raw, E_expo_mev = read_exposure(args.expo)
    if expo_raw.shape[1:] != (ny, nx):
        raise RuntimeError(f"Exposure grid {expo_raw.shape[1:]} != counts grid {(ny, nx)}")
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape[0] != nE:
        raise RuntimeError("Exposure resampling did not produce same nE as counts")

    # Solid angle map
    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    # Lon/lat and ROI box
    lon, lat = lonlat_grids(wcs, ny, nx)
    roi2d = build_roi_box_mask(lon, lat, args.roi_lon, args.roi_lat)

    # X axis in GeV (IMPORTANT)
    Ectr_gev = Ectr_mev / 1000.0

    # -------------------------
    # (A) Raw counts UNMASKED plot
    # "unmasked" here = ROI box only (no disk cut, no source mask)
    # -------------------------
    mask3d_raw = roi2d[None, :, :] & np.ones((nE, ny, nx), bool)
    C_raw, C_raw_err = raw_counts_spectrum(counts, mask3d_raw)

    out_counts = os.path.join(args.outdir, "raw_counts_unmasked_roi.png")
    plot_raw_counts(
        Ectr_gev, C_raw, C_raw_err, out_counts,
        title=rf"Raw counts per bin (ROI: |l|≤{args.roi_lon}°, |b|≤{args.roi_lat}°; no disk/source mask)"
    )

    # -------------------------
    # (B) Fit + Totani Fig.2-style plot
    # mask for fit = ROI box + disk cut + optional source mask
    # -------------------------
    # -------------------------
    # (B) Fit + Totani Fig.2/Fig.3-style plots
    #   Fig 2: |b| >= disk_cut
    #   Fig 3: |b| <  disk_cut
    # Both use the same templates + optional source mask
    # -------------------------

    # Optional source/extended mask (True=keep)
    if args.mask_fit is not None:
        srcmask = load_mask_any_shape(args.mask_fit, nE, ny, nx)
    else:
        srcmask = np.ones((nE, ny, nx), bool)

    # Resolve templates automatically
    tpl = resolve_templates(args.templates_dir)
    labels = ["IEM", "ISO", "PS", "LOOPI", "FB_FLAT", "NFW"]

    # Load templates -> mu + E2 maps ONCE
    mu_list = []
    E2_list = []
    for lab in labels:
        path = tpl[lab]
        with fits.open(path) as h:
            T = h[0].data.astype(float)
            bu = bunit_str(h[0].header)

        if T.shape != (nE, ny, nx):
            raise RuntimeError(f"{lab} template shape {T.shape} != counts shape {(nE, ny, nx)}\nFile: {path}")

        mu, E2 = template_to_mu_and_E2(T, bu, Ectr_mev, dE_mev, expo, omega)
        mu_list.append(mu)
        E2_list.append(E2)
        print(f"[template] {lab}: {os.path.basename(path)} (BUNIT='{bu}')")

    def run_fit_and_plot(region_name, region2d, out_png, out_coeff_txt, cells, coeff_cells):
        """
        region2d: boolean keep-mask (ny,nx)
        Applies srcmask (3d) AND region2d (2d) to form final fit mask.
        """
        mask3d_plot = srcmask & region2d[None, :, :]

        # Data E2 spectrum in this region
        data_y, data_yerr = data_E2_spectrum(counts, expo, omega, dE_mev, Ectr_mev, mask3d_plot)

        # Component spectra from fitted counts (more robust closure than averaging E2 maps)
        y_comp, y_model = model_E2_spectrum_from_mu(
            coeff_cells,
            cells,
            mu_list,
            expo,
            omega,
            dE_mev,
            Ectr_mev,
            mask3d_plot,
        )

        comp_specs = {lab: y_comp[j] for j, lab in enumerate(labels)}
        comp_specs["MODEL_SUM"] = y_model

        frac = np.full(nE, np.nan)
        for k in range(nE):
            if np.isfinite(data_y[k]) and data_y[k] != 0 and np.isfinite(y_model[k]):
                frac[k] = (data_y[k] - y_model[k]) / data_y[k]

        print("[closure]", region_name)
        for k in range(nE):
            if not np.isfinite(frac[k]):
                continue
            print(f"  k={k:02d} E={Ectr_gev[k]:.3g} GeV frac_resid={(100*frac[k]):+.2f}%")

        # Plot
        plot_fit_fig2(
            Ectr_gev, data_y, data_yerr, comp_specs, out_png,
            title=region_name
        )

        # Save coeffs
        # Save approximate per-bin coefficients by averaging cell coefficients weighted by number of used pixels.
        with open(out_coeff_txt, "w") as f:
            f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
            for k in range(nE):
                wsum = 0.0
                csum = np.zeros(len(labels), float)
                for ci, cell2d in enumerate(cells):
                    cm = mask3d_plot[k] & cell2d
                    npx = int(np.count_nonzero(cm))
                    if npx <= 0:
                        continue
                    wsum += npx
                    csum += npx * coeff_cells[ci, k]
                cmean = csum / wsum if wsum > 0 else np.zeros(len(labels), float)
                f.write(f"{k:02d} {Ectr_gev[k]:.6g} " + " ".join(f"{cmean[j]:.6g}" for j in range(len(labels))) + "\n")
        print("✓ wrote", out_coeff_txt)

    # Define regions
    disk_cut = float(args.disk_cut_fit)
    fig2_region2d = roi2d                              # Fig 2: include disk
    fig3_region2d = roi2d & (np.abs(lat) >= disk_cut)  # Fig 3: exclude disk

    # Fit mask is the SAME for both figures: ROI including disk
    fit_mask3d = srcmask & roi2d[None, :, :]
    cells, coeff_cells = fit_per_bin_weighted_nnls_cellwise(
        counts,
        mu_list,
        fit_mask3d,
        lon,
        lat,
        args.roi_lon,
        args.roi_lat,
        cell_deg=args.cell_deg,
        weighting=args.weighting,
    )

    out_fig2 = os.path.join(args.outdir, "totani_fig2_fit_components.png")
    out_c2   = os.path.join(args.outdir, "fit_coefficients_fig2_highlat.txt")
    run_fit_and_plot(
        rf"|l|≤{args.roi_lon}°\n|b|≤{args.roi_lat}°",
        fig2_region2d,
        out_fig2,
        out_c2,
        cells,
        coeff_cells,
    )

    out_fig3 = os.path.join(args.outdir, "totani_fig3_fit_components.png")
    out_c3   = os.path.join(args.outdir, "fit_coefficients_fig3_disk.txt")
    run_fit_and_plot(
        rf"|l|≤{args.roi_lon}°\n{disk_cut}°≤|b|≤{args.roi_lat}°\n(fit including |b|<{disk_cut}°)",
        fig3_region2d,
        out_fig3,
        out_c3,
        cells,
        coeff_cells,
    )


if __name__ == "__main__":
    main()
