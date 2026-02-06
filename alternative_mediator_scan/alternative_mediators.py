#!/usr/bin/env python3
"""
gamma_dm_fit.py

End-to-end toy pipeline for gamma–DM scattering:
- Choose mediator structure -> dsigma/dOmega(E, cosθ)
- Integrate over angles -> sigma_tot(E)
- Compute optical depth tau(E) = n_chi * sigma_tot(E) * L
- Attenuate a baseline photon spectrum
- Chi^2 test vs data
- Plot results

UNITS:
- Energies in GeV
- Cross sections returned by dsigma/dOmega are in GeV^-2
- Convert to cm^2 via: 1 GeV^-2 = 0.389379e-27 cm^2
- n_chi = rho_dm / m_chi where rho rho_dm in GeV/cm^3 and m_chi in GeV -> 1/cm^3
- L is in cm
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Optional SciPy for fitting and p-values; code works without it.
try:
    from scipy.optimize import curve_fit
    from scipy.stats import chi2 as chi2_dist
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------
# Constants
# -----------------------------
import numpy as np

# constants (if you already have them, reuse)
ALPHA_EM = 1.0 / 137.035999084
G_W = 0.653
M_W = 80.379
MPL_REDUCED = 2.435e18  # GeV (reduced Planck mass)


def mediator_title(args):
    """Pretty titles for plot labels."""
    titles = {
        "scalar": r"Scalar mediator: $\phi\,\bar\chi\chi$ and $\phi\,F_{\mu\nu}F^{\mu\nu}$",
        "pseudoscalar": r"Pseudoscalar mediator: $a\,\bar\chi i\gamma_5\chi$ and $a\,F_{\mu\nu}\tilde F^{\mu\nu}$",
        "thomson": r"Vector / millicharge (Thomson-like Compton scattering)",
        "rayleigh_even": r"Rayleigh operator (CP-even): $\bar\chi\chi\,F_{\mu\nu}F^{\mu\nu}/\Lambda^3$",
        "rayleigh_odd": r"Rayleigh operator (CP-odd): $\bar\chi\chi\,F_{\mu\nu}\tilde F^{\mu\nu}/\Lambda^3$",
        "graviton": r"Graviton exchange (schematic, forward-peaked)",
    }
    return titles.get(args.mediator, args.mediator)


def compute_yeff(args):
    """
    Compute y_eff according to the definition for each mediator structure.
    Returns:
      y_eff_value (float),
      y_eff_unit (str),
      y_eff_latex (str) -> formatted for legends/titles
    """
    if args.mediator == "scalar":
        # y_eff = g_chi * c_gamma
        y = args.gchi * args.cgamma
        unit = r"\mathrm{GeV}^{-1}"
        ytex = rf"$y_\mathrm{{eff}}={y:.2g}\ {unit}$"
        return y, unit, ytex

    if args.mediator == "pseudoscalar":
        # y_eff = g_chi * c_tilde
        y = args.gchi * args.ctilde
        unit = r"\mathrm{GeV}^{-1}"
        ytex = rf"$y_\mathrm{{eff}}={y:.2g}\ {unit}$"
        return y, unit, ytex

    if args.mediator == "thomson":
        # choose a consistent definition: y_eff = q_eff^2 (dimensionless)
        y = args.qeff**2
        unit = r""
        ytex = rf"$y_\mathrm{{eff}}=q_\mathrm{{eff}}^2={y:.2g}$"
        return y, unit, ytex

    if args.mediator in ("rayleigh_even", "rayleigh_odd"):
        # y_eff = 1/Lambda^3
        y = 1.0 / (args.Lambda**3)
        unit = r"\mathrm{GeV}^{-3}"
        ytex = rf"$y_\mathrm{{eff}}=\Lambda^{{-3}}={y:.2g}\ {unit}$"
        return y, unit, ytex

    if args.mediator == "graviton":
        # no free coupling; define y_eff as 1/M_Pl^2 as a scale (GeV^-2)
        y = 1.0 / (MPL_REDUCED**2)
        unit = r"\mathrm{GeV}^{-2}"
        ytex = rf"$y_\mathrm{{eff}}\equiv M_\mathrm{{Pl}}^{{-2}}={y:.2g}\ {unit}$"
        return y, unit, ytex

    # fallback
    y = np.nan
    unit = ""
    ytex = r"$y_\mathrm{eff}=\mathrm{n/a}$"
    return y, unit, ytex

GEV2_TO_CM2 = 0.389379e-27
CM_PER_GPC = 3.085677581e27
CM_PER_KPC = 3.085677581e21

def data_format(data, spacing):
    """
    Load spectral data
    Selects only some points for plotting
    assumes data has columns:
       E(GeV)   flux   flux_err
    data : string - file name
    spacing    : energy spacing between plotted points (GeV)
    returns E_plot, F_plot, F_err_plot (GeV, GeV/cm^2/s, GeV/cm^2/s)
    """

    # Ensure sorted
    idx_sort = np.argsort(data[:, 0])
    E_data = data[:, 0][idx_sort]
    F_data = data[:, 1][idx_sort]
    F_err  = data[:, 2][idx_sort]

    # Make a linearly spaced grid (SPACING GeV spacing) 
    # for plotting clarity
    E_min = np.floor(E_data.min() / spacing) * spacing
    E_max = np.ceil(E_data.max()  / spacing) * spacing
    E_lin = np.arange(E_min, E_max +  spacing, spacing)

    # For each desired energy, pick the nearest actual data point
    selected_indices = []
    for E_target in E_lin:
        idx = np.argmin(np.abs(E_data - E_target))
        selected_indices.append(idx)

    selected_indices = np.unique(selected_indices)

    # Force inclusion of the 175 GeV point
    idx_175 = np.argmin(np.abs(E_data - 175.0))
    selected_indices = np.unique(np.append(selected_indices, idx_175))

    # Build reduced arrays
    E_plot = E_data[selected_indices]
    F_plot = F_data[selected_indices]
    F_err_plot = F_err[selected_indices]

    print("Selected E values:", E_plot)

    return E_plot, F_plot, F_err_plot
# -----------------------------
# Kinematics (heavy target approx)
# -----------------------------
def t_hat(Eg: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """t ≃ -2 Eg^2 (1 - cosθ) [GeV^2]."""
    return -2.0 * Eg**2 * (1.0 - cos_theta)

def denom_propagator(m_med: float, Eg: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """(m_med^2 - t)^2 with t negative -> (m_med^2 + |t|)^2 [GeV^4]."""
    return (m_med**2 - t_hat(Eg, cos_theta))**2

# -----------------------------
# Effective couplings + dsigma/dOmega (toy normalisations)
# -----------------------------
def yeff_scalar(g_chi: float, c_gamma: float) -> float:
    """y_eff = g_chi * c_gamma  [GeV^-1]."""
    return g_chi * c_gamma

def dsigma_domega_scalar(Eg, cos_theta, m_chi, m_med, g_chi, c_gamma, norm=1.0):
    y = yeff_scalar(g_chi, c_gamma)  # GeV^-1
    ang = (1.0 + cos_theta**2)
    prop = denom_propagator(m_med, Eg, cos_theta)
    return norm * (y**2 / m_chi**2) * (Eg**4 * ang) / prop  # GeV^-2

def yeff_pseudoscalar(g_chi: float, c_tilde_gamma: float) -> float:
    """y_eff = g_chi * c_tilde_gamma [GeV^-1]."""
    return g_chi * c_tilde_gamma

def dsigma_domega_pseudoscalar(Eg, cos_theta, m_chi, m_med, g_chi, c_tilde_gamma, norm=1.0):
    y = yeff_pseudoscalar(g_chi, c_tilde_gamma)
    sin2 = (1.0 - cos_theta**2)
    prop = denom_propagator(m_med, Eg, cos_theta)
    return norm * (y**2 / m_chi**2) * (Eg**4 * sin2) / prop  # GeV^-2

def dsigma_domega_thomson(Eg, cos_theta, m_chi, q_eff, norm=1.0):
    """Unpolarized Thomson-like: dσ/dΩ ≈ q_eff^4/(16π^2 m_chi^2) * (1+cos^2θ)/2."""
    ang = 0.5 * (1.0 + cos_theta**2)
    return norm * (q_eff**4 / (16.0 * np.pi**2 * m_chi**2)) * ang  # GeV^-2

def yeff_rayleigh(Lambda: float) -> float:
    """y_eff = 1/Λ^3 [GeV^-3]."""
    return 1.0 / (Lambda**3)

def dsigma_domega_rayleigh_even(Eg, cos_theta, m_chi, Lambda, norm=1.0):
    ang = (1.0 + cos_theta**2)
    return norm * (Eg**4 / (m_chi**2 * Lambda**6)) * ang  # GeV^-2

def dsigma_domega_rayleigh_odd(Eg, cos_theta, m_chi, Lambda, norm=1.0):
    sin2 = (1.0 - cos_theta**2)
    return norm * (Eg**4 / (m_chi**2 * Lambda**6)) * sin2  # GeV^-2

def dsigma_domega_graviton(Eg, cos_theta, m_chi, norm=1.0, regulator=1e-6):
    """Very schematic: ~ Eg^2/Mpl^4 * P(cosθ)/(1-cosθ)^2."""
    P = (1.0 + cos_theta**2)
    denom = (1.0 - cos_theta + regulator)**2
    return norm * (Eg**2 / (MPL_REDUCED**4)) * P / denom  # GeV^-2

# -----------------------------
# Baseline spectrum models
# -----------------------------
def baseline_powerlaw(E, A, gamma, E0):
    return A * (E / E0)**(-gamma)

def baseline_cutoffpl(E, A, gamma, Ecut, E0):
    return A * (E / E0)**(-gamma) * np.exp(-E / Ecut)

# -----------------------------
# Data i/o
# -----------------------------
def load_csv_threecol(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", comments="#")
    if raw.ndim == 1 and raw.size == 0:
        raise ValueError(f"Could not read {path}")
    if raw.ndim == 1:
        raw = raw[None, :]
    # If a header exists, genfromtxt can produce nan row; filter:
    raw = raw[~np.isnan(raw).any(axis=1)]
    if raw.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns: E_GeV, flux, flux_err")
    E, F, Ferr = raw[:, 0], raw[:, 1], raw[:, 2]
    m = (E > 0) & np.isfinite(F) & np.isfinite(Ferr) & (Ferr > 0)
    return E[m], F[m], Ferr[m]

# -----------------------------
# Integration over angles
# -----------------------------
def sigma_tot_from_dsigma(
    Eg: np.ndarray,
    dsigma_domega: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_cos: int = 2000
) -> np.ndarray:
    """
    sigma_tot(E) = ∫ dΩ (dσ/dΩ) = 2π ∫_{-1}^{1} dcosθ (dσ/dΩ)
    Returns in GeV^-2.
    """
    cos = np.linspace(-1.0, 1.0, n_cos)
    # Broadcast Eg over cos:
    Eg2 = Eg[:, None]
    cos2 = cos[None, :]
    integrand = dsigma_domega(Eg2, cos2)  # GeV^-2
    # integrate over cos for each E:
    integral_cos = np.trapz(integrand, cos, axis=1)
    return 2.0 * np.pi * integral_cos  # GeV^-2

# -----------------------------
# Optical depth + attenuation
# -----------------------------
@dataclass
class PropagationParams:
    rho_dm_GeV_cm3: float
    m_chi_GeV: float
    L_cm: float

def tau_of_E(Eg: np.ndarray, sigma_tot_GeV2: np.ndarray, prop: PropagationParams) -> np.ndarray:
    n_chi_cm3 = prop.rho_dm_GeV_cm3 / prop.m_chi_GeV
    sigma_cm2 = sigma_tot_GeV2 * GEV2_TO_CM2
    return n_chi_cm3 * sigma_cm2 * prop.L_cm

# -----------------------------
# Chi^2 utilities
# -----------------------------
def chi2_value(y_obs, y_err, y_model) -> float:
    r = (y_obs - y_model) / y_err
    return float(np.sum(r**2))

def chi2_pvalue(chi2: float, dof: int) -> Optional[float]:
    if dof <= 0:
        return None
    if SCIPY_OK:
        return float(chi2_dist.sf(chi2, dof))
    return None

# -----------------------------
# Fit baseline to data
# -----------------------------
def fit_baseline(E, F, Ferr, model: str, E0: float):
    """
    Fit baseline parameters using curve_fit if SciPy exists.
    If SciPy missing, do a crude log-linear fit for pure power law.
    """
    if model == "powerlaw":
        if SCIPY_OK:
            p0 = [np.median(F), 2.0]
            def f(E, A, gamma):
                return baseline_powerlaw(E, A, gamma, E0)
            popt, pcov = curve_fit(f, E, F, p0=p0, sigma=Ferr, absolute_sigma=True, maxfev=20000)
            return ("powerlaw", popt, pcov)
        # Fallback: fit log F = log A - gamma log(E/E0)
        x = np.log(E / E0)
        y = np.log(F)
        w = 1.0 / (Ferr / F)**2
        X = np.vstack([np.ones_like(x), -x]).T
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        A = np.exp(beta[0]); gamma = beta[1]
        return ("powerlaw", np.array([A, gamma]), None)

    if model == "cutoffpl":
        if not SCIPY_OK:
            raise RuntimeError("cutoffpl fit requires SciPy. Install scipy or use --baseline_model powerlaw.")
        p0 = [np.median(F), 2.0, np.max(E)]
        def f(E, A, gamma, Ecut):
            return baseline_cutoffpl(E, A, gamma, Ecut, E0)
        popt, pcov = curve_fit(f, E, F, p0=p0, sigma=Ferr, absolute_sigma=True, maxfev=50000)
        return ("cutoffpl", popt, pcov)

    raise ValueError(f"Unknown baseline model: {model}")

def eval_baseline(E, fit_info, E0: float):
    name, popt, _ = fit_info
    if name == "powerlaw":
        A, gamma = popt
        return baseline_powerlaw(E, A, gamma, E0)
    if name == "cutoffpl":
        A, gamma, Ecut = popt
        return baseline_cutoffpl(E, A, gamma, Ecut, E0)
    raise ValueError(name)

# -----------------------------
# Mediator factory
# -----------------------------
def make_dsigma_callable(args) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns dsigma_domega(Eg[:,None], cos[None,:]) as a vectorized callable.
    """
    mchi = args.mchi
    norm = args.norm

    if args.mediator == "scalar":
        if args.mmed is None or args.gchi is None or args.cgamma is None:
            raise ValueError("scalar requires --mmed --gchi --cgamma")
        return lambda Eg, cos: dsigma_domega_scalar(Eg, cos, mchi, args.mmed, args.gchi, args.cgamma, norm=norm)

    if args.mediator == "pseudoscalar":
        if args.mmed is None or args.gchi is None or args.ctilde is None:
            raise ValueError("pseudoscalar requires --mmed --gchi --ctilde")
        return lambda Eg, cos: dsigma_domega_pseudoscalar(Eg, cos, mchi, args.mmed, args.gchi, args.ctilde, norm=norm)

    if args.mediator == "thomson":
        if args.qeff is None:
            raise ValueError("thomson requires --qeff")
        return lambda Eg, cos: dsigma_domega_thomson(Eg, cos, mchi, args.qeff, norm=norm)

    if args.mediator == "rayleigh_even":
        if args.Lambda is None:
            raise ValueError("rayleigh_even requires --Lambda")
        return lambda Eg, cos: dsigma_domega_rayleigh_even(Eg, cos, mchi, args.Lambda, norm=norm)

    if args.mediator == "rayleigh_odd":
        if args.Lambda is None:
            raise ValueError("rayleigh_odd requires --Lambda")
        return lambda Eg, cos: dsigma_domega_rayleigh_odd(Eg, cos, mchi, args.Lambda, norm=norm)

    if args.mediator == "graviton":
        return lambda Eg, cos: dsigma_domega_graviton(Eg, cos, mchi, norm=norm, regulator=args.regulator)

    raise ValueError(f"Unknown mediator: {args.mediator}")

try:
    from scipy.stats import chi2
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def compute_chi2(F_data, F_model, F_err, n_params):
    """
    Compute chi^2, reduced chi^2 and p-value for a given model.
    """
    F_data  = np.asarray(F_data)
    F_model = np.asarray(F_model)
    F_err   = np.asarray(F_err)

    chi2_val = np.sum(((F_data - F_model) / F_err)**2)
    ndof = len(F_data) - n_params
    chi2_red = chi2_val / ndof if ndof > 0 else np.nan
    p_val = chi2.sf(chi2_val, ndof) if (SCIPY_OK and ndof > 0) else np.nan

    return chi2_val, ndof, chi2_red, p_val


def print_gof(label, F_data, F_model, F_err, n_params):
    chi2_val, ndof, chi2_red, p_val = compute_chi2(F_data, F_model, F_err, n_params)
    if SCIPY_OK:
        print(f"[{label}]  χ² = {chi2_val:.2f}  for {ndof} dof  "
              f"(χ²_red = {chi2_red:.2f},  p = {p_val:.3g})")
    else:
        print(f"[{label}]  χ² = {chi2_val:.2f}  for {ndof} dof  "
              f"(χ²_red = {chi2_red:.2f},  p = n/a (scipy not installed))")
    return chi2_val, ndof


def force_curve_through_point(E_fit, F_model_fit, E0_forced, F_target_at_E0):
    """
    Scale a model curve so that it passes through (E0_forced, F_target_at_E0).
    Uses interpolation on the model evaluated on E_fit grid.
    """
    F_model_at_E0 = np.interp(E0_forced, E_fit, F_model_fit)
    if F_model_at_E0 <= 0:
        return F_model_fit.copy(), 1.0
    scale = F_target_at_E0 / F_model_at_E0
    return scale * F_model_fit, scale

import matplotlib.pyplot as plt

def plot_spectrum(
    E_data,
    E_plot, F_plot, F_err_plot,
    E_fit,
    F_smooth_fit,
    F_dm_fit,
    F_dm_forced_fit,
    res_data_plot,
    res_smooth,
    res_dm,
    res_dm_forced,
    mchi,
    E0_forced,
    yeff_tex,          # <-- NEW: LaTeX string (already formatted)
    mediator_name,     # <-- NEW: nice title string
    outfile="plots/final_plot.png"
):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(7.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- Top panel: flux ---
    ax_top.errorbar(
        E_plot, F_plot*1e6, yerr=F_err_plot*1e6,
        fmt="o", ms=4, lw=1, capsize=2,
        label="FERMI-LAT Data"
    )

    ax_top.plot(
        E_fit, F_smooth_fit*1e6,
        linestyle="--",
        label="Smooth Flux (No Scattering)"
    )

    ax_top.plot(
        E_fit, F_dm_forced_fit*1e6,
        linestyle=":",
        label=f"Forced Fit at {E0_forced:.0f} GeV"
    )

    ax_top.plot(
        E_fit, F_dm_fit*1e6,
        label=rf"Flux with Scattering ({yeff_tex})"
    )

    ax_top.set_ylabel(r'$E^2\,dN/dE$ [GeV cm$^{-2}$ s$^{-1}$] × 10$^{-6}$')
    ax_top.set_xscale("linear")
    ax_top.grid(alpha=0.3, which="both", axis="both")
    ax_top.legend()

    # Add a clear title (structure + mchi + yeff)
    ax_top.set_title(rf"{mediator_name}" "\n" rf"$m_\chi={mchi:.2g}$ GeV, {yeff_tex}")

    # --- Bottom panel: residuals relative to smooth flux ---
    ax_bot.errorbar(
        E_plot, res_data_plot*1e9,
        yerr=F_err_plot*1e9,
        fmt="o", ms=4, lw=1, capsize=2,
        label="Smooth vs Data"
    )

    ax_bot.plot(E_data, res_smooth*1e9, linestyle="--", label="Smooth")
    ax_bot.plot(E_data, res_dm_forced*1e9, linestyle=":", label="Smooth vs Forced-DM")
    ax_bot.plot(E_data, res_dm*1e9, linestyle="-", label=rf"Smooth vs DM ({yeff_tex})")

    ax_bot.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_bot.set_ylabel(r"Residuals ×10$^{-9}$")
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Gamma–DM scattering attenuation + chi2 vs data")

    # Data
    ap.add_argument("--data", required=True, help="txt with columns: E_GeV, flux, flux_err")
    ap.add_argument("--baseline", default=None,
                    help="Optional CSV baseline spectrum with columns: E_GeV, flux. If provided, no fit is done.")

    # Baseline model (if not using --baseline file)
    ap.add_argument("--baseline_model", choices=["powerlaw", "cutoffpl"], default="powerlaw")
    ap.add_argument("--E0", type=float, default=100.0, help="Pivot energy E0 [GeV] for baseline model")
    ap.add_argument("--E0_forced", type=float, default=175.0, help="Pivot energy E0 [GeV] for forced DM model")
    
    # Propagation / DM
    ap.add_argument("--mchi", type=float, required=True, help="DM mass [GeV]")
    ap.add_argument("--rho", type=float, default=1.2e-6, help="DM density [GeV/cm^3] (default cosmic mean)")
    ap.add_argument("--L", type=float, default=12.0, help="Path length [Gpc] if --L_unit=Gpc")
    ap.add_argument("--L_unit", choices=["Gpc", "kpc", "cm"], default="Gpc")

    # Mediator choice
    ap.add_argument("--mediator",
                    choices=["scalar", "pseudoscalar", "thomson", "rayleigh_even", "rayleigh_odd", "graviton"],
                    required=True)
    ap.add_argument("--norm", type=float, default=1.0,
                    help="Overall normalization knob for toy dsigma. Set once you fix conventions.")

    # Mediator params
    ap.add_argument("--mmed", type=float, default=None, help="Mediator mass [GeV] for scalar/pseudoscalar")
    ap.add_argument("--gchi", type=float, default=None, help="g_chi coupling for scalar/pseudoscalar")
    ap.add_argument("--cgamma", type=float, default=None, help="c_gamma [GeV^-1] for scalar")
    ap.add_argument("--ctilde", type=float, default=None, help="c_tilde [GeV^-1] for pseudoscalar")
    ap.add_argument("--qeff", type=float, default=None, help="Effective charge q_eff for thomson")
    ap.add_argument("--Lambda", type=float, default=None, help="Scale Λ [GeV] for Rayleigh operators")
    ap.add_argument("--regulator", type=float, default=1e-6, help="Forward regulator for graviton")

    # Numerics
    ap.add_argument("--ncos", type=int, default=2000, help="Number of cosθ points for angular integration")

    # Output
    ap.add_argument("--outdir", default="out_scatter", help="Output directory")
    ap.add_argument("--tag", default="", help="Tag string appended to output filenames")
    ap.add_argument("--residuals", action="store_true", help="Also save a residual plot")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Length conversion
    if args.L_unit == "Gpc":
        L_cm = args.L * CM_PER_GPC
    elif args.L_unit == "kpc":
        L_cm = args.L * CM_PER_KPC
    else:
        L_cm = args.L

    prop = PropagationParams(rho_dm_GeV_cm3=args.rho, m_chi_GeV=args.mchi, L_cm=L_cm)

    # Load data
    filename = args.data
    data = np.loadtxt(filename)
    E, F, Ferr = data_format(data, 50)

    # Baseline spectrum: either file or fit
    if args.baseline is not None:
        base_raw = np.genfromtxt(args.baseline, delimiter=",", comments="#")
        base_raw = base_raw[~np.isnan(base_raw).any(axis=1)]
        Eb, Fb = base_raw[:, 0], base_raw[:, 1]
        # Interpolate baseline to data energies
        F_base = np.interp(E, Eb, Fb)
        fit_info = None
    else:
        fit_info = fit_baseline(E, F, Ferr, args.baseline_model, args.E0)
        F_base = eval_baseline(E, fit_info, args.E0)

    # Build dsigma/dOmega callable and integrate to sigma_tot(E)
    dsig = make_dsigma_callable(args)
    sigma_tot = sigma_tot_from_dsigma(E, dsig, n_cos=args.ncos)  # GeV^-2

    # Optical depth and attenuated model
    tau = tau_of_E(E, sigma_tot, prop)
    F_att = F_base * np.exp(-tau)

    # Chi^2 tests (baseline vs attenuated)
    chi2_base = chi2_value(F, Ferr, F_base)
    chi2_att = chi2_value(F, Ferr, F_att)
    dof_base = len(E) - (0 if args.baseline is not None else (2 if args.baseline_model == "powerlaw" else 3))
    dof_att = dof_base  # same #params in this toy (we're not fitting mediator params here)

    p_base = chi2_pvalue(chi2_base, dof_base)
    p_att = chi2_pvalue(chi2_att, dof_att)

    print("\n=== Results ===")
    print(f"Mediator: {args.mediator}")
    print(f"mchi = {args.mchi:g} GeV, rho = {args.rho:g} GeV/cm^3, L = {L_cm:.3e} cm")
    print(f"chi2 (baseline)   = {chi2_base:.3f}  dof={dof_base}  p={p_base if p_base is not None else 'n/a'}")
    print(f"chi2 (attenuated) = {chi2_att:.3f}  dof={dof_att}  p={p_att if p_att is not None else 'n/a'}")
    print(f"Delta chi2 (base - att) = {chi2_base - chi2_att:.3f}")
    print(f"max(tau) = {np.max(tau):.3e}, min(tau) = {np.min(tau):.3e}")

    # Save a text summary
    tag = f"_{args.tag}" if args.tag else ""
    summary_path = os.path.join(args.outdir, f"summary_{args.mediator}{tag}.txt")
    with open(summary_path, "w") as f:
        f.write("=== Gamma–DM scattering fit summary ===\n")
        f.write(f"mediator = {args.mediator}\n")
        f.write(f"mchi_GeV = {args.mchi}\n")
        f.write(f"rho_GeV_cm3 = {args.rho}\n")
        f.write(f"L_cm = {L_cm}\n")
        f.write(f"chi2_base = {chi2_base}  dof={dof_base}  p={p_base}\n")
        f.write(f"chi2_att  = {chi2_att}  dof={dof_att}  p={p_att}\n")
        f.write(f"delta_chi2 = {chi2_base - chi2_att}\n")
        f.write(f"tau_min = {np.min(tau)}  tau_max = {np.max(tau)}\n")
        if fit_info is not None:
            f.write(f"baseline_fit = {fit_info[0]} params={fit_info[1].tolist()}\n")
    print(f"Saved: {summary_path}")

    # ----------------------------------------
    # Build fit grids for plotting (smooth curves)
    # ----------------------------------------
    E_data = E
    F_plot = F
    F_err_plot = Ferr

    E_fit = np.linspace(np.min(E_data), np.max(E_data), 600)

    # Evaluate smooth baseline on E_fit
    F_smooth_fit = np.interp(E_fit, E_data, F_base)  # or eval_baseline(E_fit, fit_info, E0) if you have it

    # Evaluate DM model on E_fit (attenuated)
    tau_fit = np.interp(E_fit, E_data, tau)
    F_dm_fit = F_smooth_fit * np.exp(-tau_fit)

    # ----------------------------------------
    # Forced curve through chosen E0
    # ----------------------------------------
    E0_forced = 175.0  # or args.E0_forced
    F_data_at_E0 = np.interp(E0_forced, E_data, F_plot)
    F_dm_forced_fit, forced_scale = force_curve_through_point(
        E_fit, F_dm_fit, E0_forced, F_data_at_E0
    )

    # Also evaluate forced DM at the data energies (for chi2/residuals)
    F_dm_data = F_base * np.exp(-tau)
    F_dm_forced_data = np.interp(E_data, E_fit, F_dm_forced_fit)

    # ----------------------------------------
    # Residuals relative to smooth model
    # ----------------------------------------
    res_data_plot = (F_plot - F_base)                # data - smooth
    res_smooth    = np.zeros_like(E_data)            # smooth - smooth
    res_dm        = (F_dm_data - F_base)             # DM - smooth
    res_dm_forced = (F_dm_forced_data - F_base)      # forced DM - smooth

    # ----------------------------------------
    # Goodness-of-fit prints
    # ----------------------------------------
    n_params_smooth = 2   # powerlaw (A, gamma)  OR 3 for cutoffpl
    print_gof("Smooth", F_plot, F_base, F_err_plot, n_params_smooth)

    # DM models here are NOT refit (so dof is the same as smooth for a GOF comparison)
    print_gof("DM",     F_plot, F_dm_data,        F_err_plot, n_params_smooth)
    print_gof("DM forced", F_plot, F_dm_forced_data, F_err_plot, n_params_smooth)

    # ----------------------------------------
    # Make the plot
    # ----------------------------------------
    medi_name = mediator_title(args)
    y_eff_val, y_eff_unit, y_eff_tex = compute_yeff(args)

    os.makedirs("plots", exist_ok=True)

    plot_spectrum(
        E_data=E_data,
        E_plot=E_data, F_plot=F_plot, F_err_plot=F_err_plot,
        E_fit=E_fit,
        F_smooth_fit=F_smooth_fit,
        F_dm_fit=F_dm_fit,
        F_dm_forced_fit=F_dm_forced_fit,
        res_data_plot=res_data_plot,
        res_smooth=res_smooth,
        res_dm=res_dm,
        res_dm_forced=res_dm_forced,
        mchi=args.mchi,
        E0_forced=args.E0_forced,
        yeff_tex=y_eff_tex,
        mediator_name=medi_name,
        outfile=f"plots/final_plot_{args.mediator}.png"
    )


if __name__ == "__main__":
    main()
