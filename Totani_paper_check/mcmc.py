#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import emcee

from astropy.io import fits
from astropy.wcs import WCS

from helpers.path_utils import resolve_path
from totani_helpers.exposure import resample_exposure

# ============================================================
# FILES
# ============================================================

COUNTS = "../fermi_data/processed/counts_ccube_1to1000.fits"
EXPO   = "../fermi_data/processed/expcube_1to1000.fits"

MU_IEM = "../fermi_data/processed/templates/mu_iem_counts.fits"
MU_ISO = "../fermi_data/processed/templates/mu_iso_counts.fits"
MU_PS  = "../fermi_data/processed/templates/mu_ps_counts.fits"
MU_NFW = "../fermi_data/processed/templates/mu_nfw_counts.fits"

OUTPNG = "fig2_mcmc_templates.png"

COUNTS = resolve_path(COUNTS, start=__file__)
EXPO = resolve_path(EXPO, start=__file__)
MU_IEM = resolve_path(MU_IEM, start=__file__)
MU_ISO = resolve_path(MU_ISO, start=__file__)
MU_PS = resolve_path(MU_PS, start=__file__)
MU_NFW = resolve_path(MU_NFW, start=__file__)
OUTPNG = resolve_path(OUTPNG, start=__file__)

# ============================================================
# CONSTANTS
# ============================================================

BINSZ_DEG = 0.125
PIX_SR = np.deg2rad(BINSZ_DEG)**2
DISK_CUT_DEG = 10.0

# ============================================================
# MCMC DEFINITIONS
# ============================================================

def normalise_template(mu, mask):
    """
    Normalise counts template so mean over ROI = 1 count.
    """
    norm = np.nanmean(mu[mask])
    return mu / norm, norm

def log_prior(theta):
    # log-flat prior on amplitudes
    if np.all((-20 < theta) & (theta < 20)):
        return 0.0
    return -np.inf


def log_likelihood(theta, counts, templates):
    A = np.exp(theta)
    mu = np.sum(A[:, None] * templates, axis=0)

    if np.any(mu <= 0):
        return -np.inf

    return np.sum(counts * np.log(mu) - mu)


def log_posterior(theta, counts, templates):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, counts, templates)


def run_mcmc_bin(counts_2d, templates_3d, mask_2d,
                 nwalkers=24, nsteps=1500, burnin=500):

    m = mask_2d.ravel()
    counts = counts_2d.ravel()[m]
    templates = templates_3d.reshape(4, -1)[:, m]

    # Safe initial guess
    A0 = np.maximum(
        np.sum(counts) / np.maximum(np.sum(templates, axis=1), 1.0),
        1e-6
    )
    theta0 = np.log(A0)

    p0 = theta0 + 1e-2 * np.random.randn(nwalkers, 4)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        4,
        log_posterior,
        args=(counts, templates)
    )

    sampler.run_mcmc(p0, nsteps, progress=True)

    chain = sampler.get_chain(discard=burnin, flat=True)
    return chain


def fit_all_bins_mcmc(counts, mu_iem, mu_iso, mu_ps, mu_nfw, analysis_mask):

    nE = counts.shape[0]
    med = np.zeros((nE, 4))
    lo  = np.zeros((nE, 4))
    hi  = np.zeros((nE, 4))

    for k in range(nE):
        print(f"MCMC bin {k+1}/{nE}")

        templates = np.stack([
            mu_iem[k],
            mu_iso[k],
            mu_ps[k],
            mu_nfw[k],
        ])

        chain = run_mcmc_bin(
            counts[k],
            templates,
            analysis_mask
        )

        A = np.exp(chain)

        med[k] = np.percentile(A, 50, axis=0)
        lo[k]  = np.percentile(A, 16, axis=0)
        hi[k]  = np.percentile(A, 84, axis=0)

    return med, lo, hi

# ============================================================
# LOAD DATA
# ============================================================

with fits.open(COUNTS) as h:
    counts = h[0].data.astype(float)
    hdr = h[0].header
    eb = h["EBOUNDS"].data

with fits.open(EXPO) as h:
    expo_raw = h[0].data.astype(float)
    E_exp_mev = h["ENERGIES"].data[h["ENERGIES"].columns.names[0]]

E_cnt_mev = np.sqrt(eb["E_MIN"] * eb["E_MAX"])
expo = resample_exposure(expo_raw, E_exp_mev, E_cnt_mev)

mu_iem = fits.getdata(MU_IEM)
mu_iso = fits.getdata(MU_ISO)
mu_ps  = fits.getdata(MU_PS)
mu_nfw = fits.getdata(MU_NFW)

nE, ny, nx = counts.shape

mu_iem_n = np.zeros_like(mu_iem)
mu_iso_n = np.zeros_like(mu_iso)
mu_ps_n  = np.zeros_like(mu_ps)
mu_nfw_n = np.zeros_like(mu_nfw)

norm_iem = np.zeros(nE)
norm_iso = np.zeros(nE)
norm_ps  = np.zeros(nE)
norm_nfw = np.zeros(nE)

# ============================================================
# GEOMETRY & MASK
# ============================================================

wcs = WCS(hdr).celestial
yy, xx = np.mgrid[:ny, :nx]
lon, lat = wcs.pixel_to_world_values(xx, yy)

roi_mask  = (np.abs(lon) <= 60) & (np.abs(lat) <= 60)
disk_mask = np.abs(lat) >= DISK_CUT_DEG
analysis_mask = roi_mask & disk_mask

for k in range(nE):
    mu_iem_n[k], norm_iem[k] = normalise_template(mu_iem[k], analysis_mask)
    mu_iso_n[k], norm_iso[k] = normalise_template(mu_iso[k], analysis_mask)
    mu_ps_n[k],  norm_ps[k]  = normalise_template(mu_ps[k],  analysis_mask)
    mu_nfw_n[k], norm_nfw[k] = normalise_template(mu_nfw[k], analysis_mask)


# ============================================================
# ENERGY BINS
# ============================================================

Emin = eb["E_MIN"] / 1000.0
Emax = eb["E_MAX"] / 1000.0
dE   = Emax - Emin
Ectr = np.sqrt(Emin * Emax) / 1000


# ============================================================
# RUN MCMC
# ============================================================

med, lo, hi = fit_all_bins_mcmc(
    counts,
    mu_iem_n,
    mu_iso_n,
    mu_ps_n,
    mu_nfw_n,
    analysis_mask
)

# ============================================================
# COUNTS → MEAN FLUX
# ============================================================

# amplitudes are now counts per bin per unit-normalised template
counts_per_bin = med * np.vstack([
    norm_iem,
    norm_iso,
    norm_ps,
    norm_nfw
]).T

# denom = expo * PIX_SR * dE[:, None, None]
# mean_denom = np.nanmean(denom[:, analysis_mask], axis=1)
# mean_denom = mean_denom[:, None]  # FIX broadcasting

mean_denom = np.nanmean(
    expo * PIX_SR * dE[:, None, None],
    axis=(1,2)
)

mean_flux = counts_per_bin / mean_denom[:, None]

err_lo = np.maximum(med - lo, 0.0) / mean_denom
err_hi = np.maximum(hi - med, 0.0) / mean_denom

# ============================================================
# SAVE RESULTS
# ============================================================

np.savez(
    "mcmc_results_fig2.npz",
    Ectr=Ectr,
    med=med,
    lo=lo,
    hi=hi,
    mean_flux=mean_flux,
    err_lo=err_lo,
    err_hi=err_hi,
)

print("✓ Saved MCMC summary: mcmc_results_fig2.npz")

# ============================================================
# PLOT (TOTANI FIG. 2 STYLE)
# ============================================================

E2 = (Ectr * 1e3)**2  # MeV²

labels  = ['Gas + ICS', 'Isotropic', 'Point sources', r'NFW-$\rho^{2.5}$']
markers = ['s', '<', 'D', 'o']
colors  = ['black', 'brown', 'green', 'red']

plt.figure(figsize=(7, 6))

for j in range(4):
    good = (
        np.isfinite(mean_flux[:, j]) &
        np.isfinite(err_lo[:, j]) &
        np.isfinite(err_hi[:, j]) &
        (mean_flux[:, j] > 0)
    )

    plt.errorbar(
        Ectr[good],
        E2[good] * mean_flux[good, j],
        yerr=[
            E2[good] * err_lo[good, j],
            E2[good] * err_hi[good, j],
        ],
        fmt=markers[j] + '--',
        color=colors[j],
        label=labels[j],
        capsize=2,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Photon energy [GeV]")
plt.ylabel(r"$E^2\,dN/dE\;[\mathrm{MeV\,cm^{-2}\,s^{-1}\,sr^{-1}}]$")
plt.legend(frameon=False)
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPNG, dpi=200)
plt.show()

print("✓ Fig.2 MCMC plot written:", OUTPNG)
