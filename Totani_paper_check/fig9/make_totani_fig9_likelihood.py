#!/usr/bin/env python3

import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import nnls

from scipy.optimize import minimize

REPO_DIR = os.environ["REPO_PATH"]
DATA_DIR = os.path.join(REPO_DIR, "fermi_data", "totani")

R_SUN_KPC = 8.25
RS_KPC = 20.0
S_MAX_KPC = 50.0


def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), dtype=float)


def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    if expo_raw.shape[0] == len(E_tgt_mev):
        return expo_raw
    if E_expo_mev is None:
        raise RuntimeError("Exposure planes != counts planes and EXPO has no ENERGIES table.")

    order = np.argsort(E_expo_mev)
    E_expo_mev = E_expo_mev[order]
    expo_raw = expo_raw[order].astype(float)

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
        out[j] = (1.0 - w[j]) * flat[i0[j]] + w[j] * flat[i1[j]]

    return out.reshape(len(E_tgt_mev), ny, nx)


def gNFW_rho(r_kpc, gamma):
    x = r_kpc / RS_KPC
    return 1.0 / (np.power(x, gamma) * np.power(1.0 + x, 3.0 - gamma))


def los_halo_intensity(lon_deg, lat_deg, *, gamma, rho_power, n_s=256, chunk=8):
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)

    s = np.linspace(0.0, S_MAX_KPC, n_s, dtype=np.float32)
    ds = float(s[1] - s[0])

    cospsi = (np.cos(b) * np.cos(l)).astype(np.float32)

    intensity = np.zeros_like(cospsi, dtype=np.float32)
    for i in range(0, len(s), chunk):
        s_chunk = s[i : i + chunk][:, None, None]
        r2 = (
            (R_SUN_KPC * R_SUN_KPC)
            + (s_chunk * s_chunk)
            - (2.0 * R_SUN_KPC) * s_chunk * cospsi[None, :, :]
        )
        r = np.sqrt(r2, dtype=np.float32)
        rho = gNFW_rho(r, gamma=gamma)
        intensity += np.sum(np.power(rho, rho_power), axis=0, dtype=np.float32) * ds

    return intensity.astype(float)


def load_mu_template(path, expected_shape=None):
    with fits.open(path) as h:
        data = h[0].data.astype(float)
        bunit = h[0].header.get("BUNIT", "")
    if expected_shape is not None and data.shape != expected_shape:
        raise RuntimeError(f"{path} has shape {data.shape}, expected {expected_shape}")
    return data, bunit


def poisson_loglike(counts_2d, mu_2d, mask_2d):
    m = mask_2d
    C = counts_2d[m].astype(float)
    M = mu_2d[m].astype(float)
    M = np.clip(M, 1e-30, None)
    return float(np.sum(C * np.log(M) - M))


def nnls_fit(counts_2d, mu_list_2d, mask_2d):
    m = mask_2d.ravel()
    y = counts_2d.ravel()[m]
    X = np.vstack([mu.ravel()[m] for mu in mu_list_2d]).T
    c, _ = nnls(X, y)
    return c


def poisson_mle_fit(counts_2d, mu_list_2d, mask_2d, x0=None):
    """Poisson MLE for non-negative linear template amplitudes.

    Maximizes ln L = sum_i [ C_i ln M_i - M_i ] (up to constant),
    where M_i = sum_j a_j mu_{j,i}.
    """

    m = mask_2d
    C = counts_2d[m].astype(float)
    Mus = [mu[m].astype(float) for mu in mu_list_2d]
    ncomp = len(Mus)

    if x0 is None:
        # least-squares seed
        x0 = nnls_fit(counts_2d, mu_list_2d, mask_2d)
    x0 = np.asarray(x0, float)
    if x0.shape != (ncomp,):
        x0 = np.zeros(ncomp, float)

    def nll(a):
        a = np.clip(a, 0.0, None)
        M = np.zeros_like(C)
        for aj, muj in zip(a, Mus):
            if aj != 0.0:
                M += aj * muj
        M = np.clip(M, 1e-30, None)
        return float(np.sum(M - C * np.log(M)))

    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * ncomp,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    a = np.clip(res.x, 0.0, None)
    # Evaluate ll at optimum
    M = np.zeros_like(C)
    for aj, muj in zip(a, Mus):
        if aj != 0.0:
            M += aj * muj
    ll = float(np.sum(C * np.log(np.clip(M, 1e-30, None)) - M))
    return a, ll


def profile_halo_envelope(counts_2d, mu_base_list_2d, mu_halo_2d, mask_2d, a_best, ll_nohalo, ll_best):
    # 95% top region (1 dof Wilks): delta(2 ln L)=3.84 => delta ln L=1.92
    ll_cut = ll_best - 1.92

    # scan amplitude around best-fit in log space + include 0
    # Keep this small-ish for runtime.
    if not np.isfinite(a_best) or a_best <= 0:
        a_grid = np.concatenate([np.array([0.0]), np.logspace(-8, -2, 40)])
    else:
        a_min = max(a_best / 30.0, 0.0)
        a_max = a_best * 30.0
        lo = max(a_min, 1e-20)
        hi = max(a_max, lo * 10)
        a_grid = np.unique(np.concatenate([np.array([0.0, a_best]), np.logspace(np.log10(lo), np.log10(hi), 60)]))

    dlnl_vals = []
    for a in a_grid:
        # Fix halo amplitude and refit baseline via Poisson MLE.
        # This is the profile likelihood in a.
        C_adj = counts_2d - a * mu_halo_2d
        C_adj = np.clip(C_adj, 0.0, None)
        base_a, _ = poisson_mle_fit(C_adj, mu_base_list_2d, mask_2d)

        mu_model = a * mu_halo_2d
        for aj, muj in zip(base_a, mu_base_list_2d):
            if aj != 0.0:
                mu_model = mu_model + aj * muj

        ll = poisson_loglike(counts_2d, mu_model, mask_2d)
        if ll >= ll_cut:
            dlnl_vals.append(ll - ll_nohalo)

    if len(dlnl_vals) == 0:
        return np.nan, np.nan

    return float(np.min(dlnl_vals)), float(np.max(dlnl_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--counts",
        default=os.path.join(DATA_DIR, "processed", "counts_ccube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--expo",
        default=os.path.join(DATA_DIR, "processed", "expcube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--templates-dir",
        default=os.path.join(DATA_DIR, "processed", "templates"),
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "plots_fig9"),
    )

    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut-fit", type=float, default=10.0)
    ap.add_argument("--mask-fit", default=None)

    ap.add_argument("--halo-gamma", type=float, default=1.25)
    ap.add_argument("--halo-powers", default="2.5,2,1")

    ap.add_argument("--n-s", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with fits.open(args.counts) as h:
        hdr = h[0].header
        counts = h[0].data.astype(float)
        eb = h["EBOUNDS"].data

    wcs = WCS(hdr).celestial
    ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])

    Emin_mev = eb["E_MIN"].astype(float) / 1000.0
    Emax_mev = eb["E_MAX"].astype(float) / 1000.0
    dE_mev = (Emax_mev - Emin_mev)
    Ectr_mev = np.sqrt(Emin_mev * Emax_mev)
    Ectr_gev = Ectr_mev / 1000.0
    nE = len(Ectr_mev)

    with fits.open(args.expo) as h:
        expo_raw = h[0].data.astype(float)
        E_expo_mev = None
        if "ENERGIES" in h:
            col0 = h["ENERGIES"].columns.names[0]
            E_expo_mev = np.array(h["ENERGIES"].data[col0], dtype=float)

    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != (nE, ny, nx):
        raise RuntimeError(f"Exposure shape {expo.shape} not compatible with {(nE, ny, nx)}")

    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    lon_w = ((lon + 180.0) % 360.0) - 180.0

    omega = pixel_solid_angle_map(wcs, ny, nx, args.binsz)

    roi2d = (np.abs(lon_w) <= args.roi_lon) & (np.abs(lat) <= args.roi_lat)

    if args.mask_fit is None:
        srcmask2d = np.ones((ny, nx), bool)
    else:
        with fits.open(args.mask_fit) as h:
            m = h[0].data
        if m.ndim == 3:
            m = m[0]
        srcmask2d = np.isfinite(m) & (m != 0)

    fit_mask2d = roi2d & srcmask2d

    # baseline templates (counts-space mu)
    tdir = args.templates_dir
    mu_iem, _ = load_mu_template(os.path.join(tdir, "mu_iem_counts.fits"), expected_shape=(nE, ny, nx))
    mu_iso, _ = load_mu_template(os.path.join(tdir, "mu_iso_counts.fits"), expected_shape=(nE, ny, nx))
    mu_ps, _ = load_mu_template(os.path.join(tdir, "mu_ps_counts.fits"), expected_shape=(nE, ny, nx))
    mu_loopi, _ = load_mu_template(os.path.join(tdir, "mu_loopI_counts.fits"), expected_shape=(nE, ny, nx))
    mu_fbflat, _ = load_mu_template(os.path.join(tdir, "mu_bubbles_flat_counts.fits"), expected_shape=(nE, ny, nx))

    base_mu_list = [mu_iem, mu_iso, mu_ps, mu_loopi, mu_fbflat]

    # halo spatial models
    halo_powers = [float(x) for x in args.halo_powers.split(",")]

    # build spatial templates once (ROI-normalized)
    halo_spatials = {}
    for p in halo_powers:
        h2d = los_halo_intensity(lon_w, lat, gamma=args.halo_gamma, rho_power=p, n_s=args.n_s, chunk=args.chunk)
        h2d[~roi2d] = 0.0
        s = float(np.nansum(h2d))
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError(f"Halo spatial template (power={p}) is zero in ROI")
        halo_spatials[p] = h2d / s

    # simple flat spectrum normalized to unit total flux (like other builders)
    Phi_bin = np.ones(nE, float)
    Phi_bin /= Phi_bin.sum()

    # convert spatial->mu cube: mu[k] = (Phi_bin[k] * spatial / (omega*dE)) * expo*omega*dE = Phi_bin[k]*spatial*expo
    halo_mu = {}
    for p in halo_powers:
        mu = np.empty((nE, ny, nx), float)
        for k in range(nE):
            mu[k] = Phi_bin[k] * halo_spatials[p] * expo[k]
        halo_mu[p] = mu

    # results arrays
    dlnl_best = {p: np.full(nE, np.nan) for p in halo_powers}
    dlnl_lo = {p: np.full(nE, np.nan) for p in halo_powers}
    dlnl_hi = {p: np.full(nE, np.nan) for p in halo_powers}

    for k in range(nE):
        Ck = counts[k]
        base_2d = [mu[k] for mu in base_mu_list]

        c0, ll0 = poisson_mle_fit(Ck, base_2d, fit_mask2d)

        for p in halo_powers:
            halo_2d = halo_mu[p][k]

            c1, ll1 = poisson_mle_fit(Ck, base_2d + [halo_2d], fit_mask2d)

            dlnl_best[p][k] = ll1 - ll0

            a_best = float(c1[-1])
            lo, hi = profile_halo_envelope(Ck, base_2d, halo_2d, fit_mask2d, a_best, ll0, ll1)
            dlnl_lo[p][k] = lo
            dlnl_hi[p][k] = hi

    # plot
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.6, 4.8))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_xlabel("photon energy  [GeV]")
    ax.set_ylabel(r"$\ln(L) - \ln(L_{\rm no-halo})$")

    ax.set_ylim(-20, 120)

    styles = {
        2.5: dict(color="k", ls="-.", marker="^") ,
        2.0: dict(color="r", ls="-", marker="o"),
        1.0: dict(color="b", ls="--", marker="s"),
    }

    for p in halo_powers:
        st = styles.get(p, dict(color=None, ls="-", marker=None))
        ax.plot(Ectr_gev, dlnl_best[p], label=f"NFW-ρ^{p:g}", lw=2.0, **{k: v for k, v in st.items() if k in ["color", "ls", "marker"]})
        ax.plot(Ectr_gev, dlnl_lo[p], lw=0.8, alpha=0.6, **{k: v for k, v in st.items() if k in ["color", "ls"]})
        ax.plot(Ectr_gev, dlnl_hi[p], lw=0.8, alpha=0.6, **{k: v for k, v in st.items() if k in ["color", "ls"]})

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out_png = os.path.join(args.outdir, "totani_fig9_lnl_diff.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_txt = os.path.join(args.outdir, "totani_fig9_lnl_diff.txt")
    with open(out_txt, "w") as f:
        f.write("# k Ectr_GeV " + " ".join([f"dlnl_best_p{p:g} dlnl_lo_p{p:g} dlnl_hi_p{p:g}" for p in halo_powers]) + "\n")
        for k in range(nE):
            row = [f"{k:02d}", f"{Ectr_gev[k]:.6g}"]
            for p in halo_powers:
                row += [f"{dlnl_best[p][k]:.8g}", f"{dlnl_lo[p][k]:.8g}", f"{dlnl_hi[p][k]:.8g}"]
            f.write(" ".join(row) + "\n")

    print("✓ wrote", out_png)
    print("✓ wrote", out_txt)


if __name__ == "__main__":
    main()
