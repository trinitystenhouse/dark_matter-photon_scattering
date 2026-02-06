#!/usr/bin/env python3
"""
alternative_mediators_scan.py

Scan alternative mediator models for gamma–DM scattering, using:
- your data_format() selection for plotting points
- your smooth_flux_model() fit on reduced points
- your get_sigma_spline() approach (unit-normalised sigma shape spline)
- your forced-fit method via A_forced at 175 GeV
- saves ALL plots (one per scan point) + CSV summary

Expected input file columns:
  E(GeV)   flux   flux_err
"""

from __future__ import annotations
import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
from scipy.interpolate import UnivariateSpline
from helpers.trinity_plotting import set_plot_style

set_plot_style(style="dark")

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

# -----------------------------
# Constants / unit conversions
# -----------------------------
# 1 GeV^-2 = 0.389379e-27 cm^2 = 0.389379e-31 m^2
GEV2_TO_CM2 = 0.389379e-27
GEV2_TO_M2  = 0.389379e-31

CM_PER_GPC = 3.085677581e27
CM_PER_KPC = 3.085677581e21

MPL_REDUCED = 2.435e18  # GeV

# -----------------------------
# Your data formatting
# -----------------------------
def data_format(data, spacing):
    """
    Load spectral data
    Selects only some points for plotting
    assumes data has columns:
       E(GeV)   flux   flux_err
    data : np.ndarray
    spacing : energy spacing between plotted points (GeV)
    returns E_plot, F_plot, F_err_plot
    """
    idx_sort = np.argsort(data[:, 0])
    E_data = data[:, 0][idx_sort]
    F_data = data[:, 1][idx_sort]
    F_err  = data[:, 2][idx_sort]

    E_min = np.floor(E_data.min() / spacing) * spacing
    E_max = np.ceil(E_data.max()  / spacing) * spacing
    E_lin = np.arange(E_min, E_max + spacing, spacing)

    selected_indices = []
    for E_target in E_lin:
        idx = np.argmin(np.abs(E_data - E_target))
        selected_indices.append(idx)

    selected_indices = np.unique(selected_indices)

    # Force inclusion of 175 GeV point
    idx_175 = np.argmin(np.abs(E_data - 175.0))
    selected_indices = np.unique(np.append(selected_indices, idx_175))

    E_plot = E_data[selected_indices]
    F_plot = F_data[selected_indices]
    F_err_plot = F_err[selected_indices]

    print("Selected E values:", E_plot)
    return E_plot, F_plot, F_err_plot


def smooth_flux_model(E, N0, E0, gamma):
    """Power-law model: F(E) = N0 * (E/E0)^(-gamma)"""
    return N0 * (E / E0)**(-gamma)


# -----------------------------
# Your goodness-of-fit
# -----------------------------
def compute_chi2(F_data, F_model, F_err, n_params):
    F_data  = np.asarray(F_data)
    F_model = np.asarray(F_model)
    F_err   = np.asarray(F_err)

    chi2_val = np.sum(((F_data - F_model) / F_err)**2)
    ndof = len(F_data) - n_params
    chi2_red = chi2_val / ndof if ndof > 0 else np.nan
    p_val = chi2_dist.sf(chi2_val, ndof) if ndof > 0 else np.nan
    return chi2_val, ndof, chi2_red, p_val


def print_gof(label, F_data, F_model, F_err, n_params):
    chi2_val, ndof, chi2_red, p_val = compute_chi2(F_data, F_model, F_err, n_params)
    print(f"[{label}]  χ² = {chi2_val:.2f}  for {ndof} dof  "
          f"(χ²_red = {chi2_red:.2f},  p = {p_val:.3g})")
    return chi2_val, ndof


# -----------------------------
# Physics: A = rho*L/mchi (your A_def)
# -----------------------------
def A_from_mchi(mchi_GeV: float, rho_GeV_cm3: float, L_cm: float) -> float:
    """
    A = (rho/mchi) * L   with rho in GeV/cm^3, mchi in GeV, L in cm
    Units: 1/cm^2, so that tau = A * sigma_cm2 is dimensionless
    """
    return (rho_GeV_cm3 / mchi_GeV) * L_cm


# -----------------------------
# Alternative mediator dsigma/dOmega (toy-normalised, as before)
# -----------------------------
def t_hat(Eg, cos_theta):
    """t ≃ -2 Eg^2 (1-cosθ) [GeV^2], heavy-target approx."""
    return -2.0 * Eg**2 * (1.0 - cos_theta)

def denom_propagator(m_med, Eg, cos_theta):
    """(m_med^2 - t)^2"""
    return (m_med**2 - t_hat(Eg, cos_theta))**2

def dsigma_domega_scalar(Eg, cos_theta, m_chi, m_med, g_chi, c_gamma, norm=1.0):
    y = g_chi * c_gamma  # GeV^-1
    ang = (1.0 + cos_theta**2)
    prop = denom_propagator(m_med, Eg, cos_theta)
    return norm * (y**2 / m_chi**2) * (Eg**4 * ang) / prop  # GeV^-2

def dsigma_domega_pseudoscalar(Eg, cos_theta, m_chi, m_med, g_chi, c_tilde, norm=1.0):
    y = g_chi * c_tilde
    sin2 = (1.0 - cos_theta**2)
    prop = denom_propagator(m_med, Eg, cos_theta)
    return norm * (y**2 / m_chi**2) * (Eg**4 * sin2) / prop  # GeV^-2

def dsigma_domega_thomson(Eg, cos_theta, m_chi, q_eff, norm=1.0):
    ang = 0.5 * (1.0 + cos_theta**2)           # shape (1, ncos)
    pref = norm * (q_eff**4 / (16.0 * np.pi**2 * m_chi**2))
    return pref * ang * np.ones_like(Eg)       # shape (nE, ncos) after broadcasting

def dsigma_domega_rayleigh_even(Eg, cos_theta, m_chi, Lambda, norm=1.0):
    ang = (1.0 + cos_theta**2)
    return norm * (Eg**4 / (m_chi**2 * Lambda**6)) * ang  # GeV^-2

def dsigma_domega_rayleigh_odd(Eg, cos_theta, m_chi, Lambda, norm=1.0):
    sin2 = (1.0 - cos_theta**2)
    return norm * (Eg**4 / (m_chi**2 * Lambda**6)) * sin2  # GeV^-2

def dsigma_domega_graviton(Eg, cos_theta, m_chi, norm=1.0, regulator=1e-6):
    P = (1.0 + cos_theta**2)
    denom = (1.0 - cos_theta + regulator)**2
    return norm * (Eg**2 / (MPL_REDUCED**4)) * P / denom  # GeV^-2


# -----------------------------
# sigma_tot(E): integrate over angles -> GeV^-2 then to m^2 or cm^2
# -----------------------------
def sigma_tot_vec(E_array_GeV, dsigma_func, ncos=1500) -> np.ndarray:
    """
    Returns sigma_tot(E) in m^2 (to match your get_sigma_spline docstring),
    using:
      sigma_tot = 2π ∫_{-1}^1 dcosθ dσ/dΩ
    where dσ/dΩ is in GeV^-2.
    """
    E = np.asarray(E_array_GeV, dtype=float)
    cos = np.linspace(-1.0, 1.0, ncos)
    Eg2 = E[:, None]
    cos2 = cos[None, :]

    integrand = dsigma_func(Eg2, cos2)  # GeV^-2
    integral_cos = np.trapz(integrand, cos, axis=1)
    sigma_GeV2 = 2.0 * np.pi * integral_cos  # GeV^-2
    return sigma_GeV2 * GEV2_TO_M2  # m^2


# -----------------------------
# Your spline builder (same structure, just plugs into sigma_tot_vec above)
# -----------------------------
def get_sigma_spline(E_array, dsigma_func, mchi, ncos):
    """
    Returns:
        sigma_spline: spline over unit-normalised σ(E) shape
        sigma_max_cm2: max σ(E) in cm^2
    """
    sigma_raw_m2 = sigma_tot_vec(E_array, dsigma_func, ncos=ncos)  # m^2
    sigma_cm2 = sigma_raw_m2 * 1e4  # m^2 -> cm^2
    sigma_max_cm2 = sigma_cm2.max()

    # shape normalised to 1 at max
    sigma_shape = sigma_cm2 / sigma_max_cm2
    sigma_spline = UnivariateSpline(E_array, sigma_shape, s=0, k=3)
    return sigma_spline, sigma_max_cm2


# -----------------------------
# Your flux model with scattering
# -----------------------------
def flux_dm(E, A, N0, E0, gamma, sigma_spline, sigma_max_cm2):
    """
    F_DM(E) = F_smooth(E) * exp(-A * sigma_cm2(E))
    where sigma_cm2(E) = sigma_shape(E) * sigma_max_cm2.
    """
    F_s = smooth_flux_model(E, N0, E0, gamma)
    sigma_cm2 = sigma_spline(E) * sigma_max_cm2
    return F_s * np.exp(-A * sigma_cm2)


# -----------------------------
# Forced fit (your method)
# -----------------------------
def compute_A_forced(E0_forced, E_data, F_data, N0_fit, E0_fit, gamma_fit, sigma_spline, sigma_max_cm2):
    idx_0 = np.argmin(np.abs(E_data - E0_forced))
    F0_data = F_data[idx_0]
    if F0_data <= 0:
        raise ValueError(f"Flux at {E0_forced} GeV must be positive.")

    F0_smooth = smooth_flux_model(E0_forced, N0_fit, E0_fit, gamma_fit)

    sigma0_shape = float(sigma_spline(E0_forced))
    sigma0_cm2 = sigma0_shape * sigma_max_cm2
    if sigma0_cm2 <= 0:
        raise ValueError("sigma(E0) <= 0; check spline / sigma normalisation.")

    # Force: F_smooth * exp(-A_forced * sigma0) = F_data
    A_forced = -np.log(F0_data / F0_smooth) / sigma0_cm2
    return A_forced


# -----------------------------
# Plotting (your style, saves to outfile)
# -----------------------------
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
    y_eff_tex,
    mediator_name,
    outfile
):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(7.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.errorbar(E_plot, F_plot*1e6, yerr=F_err_plot*1e6,
                    fmt="o", ms=4, lw=1, capsize=2, label="FERMI-LAT Data")

    ax_top.plot(E_fit, F_smooth_fit*1e6, linestyle="--",
                label="Smooth Flux (No Scattering)")

    ax_top.plot(E_fit, F_dm_forced_fit*1e6, linestyle=":",
                label=f"Forced Fit at {E0_forced:.0f} GeV")

    ax_top.plot(E_fit, F_dm_fit*1e6,
                label=rf"Flux with Scattering ({y_eff_tex})")

    ax_top.set_ylabel(r'$E^2\,dN/dE$ [GeV cm$^{-2}$ s$^{-1}$] × 10$^{-6}$')
    ax_top.set_xscale("linear")
    ax_top.grid(alpha=0.3, which="both", axis="both")
    ax_top.legend()

    ax_top.set_title(rf"{mediator_name}" "\n" rf"$m_\chi={mchi:.2g}$ GeV, {y_eff_tex}")

    ax_bot.errorbar(E_plot, res_data_plot*1e9, yerr=F_err_plot*1e9,
                    fmt="o", ms=4, lw=1, capsize=2, label="Smooth vs Data")

    ax_bot.plot(E_data, res_smooth*1e9, linestyle="--", label="Smooth")
    ax_bot.plot(E_data, res_dm_forced*1e9, linestyle=":", label="Smooth vs Forced-DM")
    ax_bot.plot(E_data, res_dm*1e9, linestyle="-", label=rf"Smooth vs DM ({y_eff_tex})")

    ax_bot.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_bot.set_ylabel(r"Residuals ×10$^{-9}$")
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


# -----------------------------
# y_eff per mediator + titles
# -----------------------------
def mediator_title(med: str) -> str:
    titles = {
        "scalar": r"Scalar mediator: $\phi\,\bar\chi\chi$ and $\phi\,F_{\mu\nu}F^{\mu\nu}$",
        "pseudoscalar": r"Pseudoscalar: $a\,\bar\chi i\gamma_5\chi$ and $a\,F_{\mu\nu}\tilde F^{\mu\nu}$",
        "thomson": r"Vector / millicharge (Thomson-like)",
        "rayleigh_even": r"Rayleigh (CP-even): $\bar\chi\chi\,F_{\mu\nu}F^{\mu\nu}/\Lambda^3$",
        "rayleigh_odd": r"Rayleigh (CP-odd): $\bar\chi\chi\,F_{\mu\nu}\tilde F^{\mu\nu}/\Lambda^3$",
        "graviton": r"Graviton exchange (schematic)",
    }
    return titles.get(med, med)

def compute_yeff_tex(mediator: str, params: dict) -> str:
    """
    Returns LaTeX string for y_eff in legend/title (using your per-case definitions).
    """
    if mediator == "scalar":
        y = params["gchi"] * params["cgamma"]  # GeV^-1
        return rf"$y_\mathrm{{eff}}={y:.2g}\ \mathrm{{GeV^{{-1}}}}$"
    if mediator == "pseudoscalar":
        y = params["gchi"] * params["ctilde"]  # GeV^-1
        return rf"$y_\mathrm{{eff}}={y:.2g}\ \mathrm{{GeV^{{-1}}}}$"
    if mediator == "thomson":
        y = params["qeff"]**2
        return rf"$y_\mathrm{{eff}}=q_\mathrm{{eff}}^2={y:.2g}$"
    if mediator in ("rayleigh_even", "rayleigh_odd"):
        y = 1.0 / (params["Lambda"]**3)  # GeV^-3
        return rf"$y_\mathrm{{eff}}=\Lambda^{{-3}}={y:.2g}\ \mathrm{{GeV^{{-3}}}}$"
    if mediator == "graviton":
        y = 1.0 / (MPL_REDUCED**2)  # GeV^-2
        return rf"$y_\mathrm{{eff}}\equiv M_\mathrm{{Pl}}^{{-2}}={y:.2g}\ \mathrm{{GeV^{{-2}}}}$"
    return r"$y_\mathrm{eff}=\mathrm{n/a}$"


# -----------------------------
# dsigma factory (depends on scan point)
# -----------------------------
def make_dsigma_func(mediator: str, mchi: float, params: dict, norm: float, regulator: float):
    if mediator == "scalar":
        return lambda Eg, cos: dsigma_domega_scalar(Eg, cos, mchi, params["mmed"], params["gchi"], params["cgamma"], norm=norm)
    if mediator == "pseudoscalar":
        return lambda Eg, cos: dsigma_domega_pseudoscalar(Eg, cos, mchi, params["mmed"], params["gchi"], params["ctilde"], norm=norm)
    if mediator == "thomson":
        return lambda Eg, cos: dsigma_domega_thomson(Eg, cos, mchi, params["qeff"], norm=norm)
    if mediator == "rayleigh_even":
        return lambda Eg, cos: dsigma_domega_rayleigh_even(Eg, cos, mchi, params["Lambda"], norm=norm)
    if mediator == "rayleigh_odd":
        return lambda Eg, cos: dsigma_domega_rayleigh_odd(Eg, cos, mchi, params["Lambda"], norm=norm)
    if mediator == "graviton":
        return lambda Eg, cos: dsigma_domega_graviton(Eg, cos, mchi, norm=norm, regulator=regulator)
    raise ValueError(mediator)


# -----------------------------
# scan parser
# -----------------------------
def parse_scan_spec(spec: str) -> np.ndarray:
    """
    Supported:
      list:0.1,1,10
      log:min:max:n
      lin:min:max:n
      or comma list: 0.1,1,10
    """
    s = spec.strip()
    if s.startswith("list:"):
        vals = s.split("list:", 1)[1]
        return np.array([float(x) for x in vals.split(",") if x.strip() != ""])
    if s.startswith("log:"):
        _, rest = s.split("log:", 1)
        a, b, n = rest.split(":")
        return np.logspace(np.log10(float(a)), np.log10(float(b)), int(n))
    if s.startswith("lin:"):
        _, rest = s.split("lin:", 1)
        a, b, n = rest.split(":")
        return np.linspace(float(a), float(b), int(n))
    if "," in s:
        return np.array([float(x) for x in s.split(",") if x.strip() != ""])
    return np.array([float(s)])

def parse_scan_kv(kv: str):
    if "=" not in kv:
        raise ValueError(f"Bad --scan '{kv}'. Expected param=spec.")
    k, v = kv.split("=", 1)
    return k.strip(), parse_scan_spec(v.strip())


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--filename",
        default=os.path.join(REPO_DIR, "fermi_data", "york", "processed", "spectrum_data.txt"),
        help="Spectrum data",
    )
    ap.add_argument("--spacing", type=float, default=50.0, help="Spacing in GeV for data_format selection")

    ap.add_argument("--mchi", type=float, required=True, help="DM mass [GeV]")
    ap.add_argument("--rho", type=float, default=1.2e-6, help="DM density [GeV/cm^3]")
    ap.add_argument("--L", type=float, default=12.0, help="Path length in units of --L_unit")
    ap.add_argument("--L_unit", choices=["Gpc", "kpc", "cm"], default="Gpc")

    ap.add_argument("--mediator",
                    choices=["scalar", "pseudoscalar", "thomson", "rayleigh_even", "rayleigh_odd", "graviton"],
                    required=True)

    ap.add_argument("--norm", type=float, default=1.0, help="overall dsigma normalisation (toy)")
    ap.add_argument("--regulator", type=float, default=1e-6, help="graviton IR regulator")
    ap.add_argument("--ncos", type=int, default=1500, help="cos(theta) integration resolution")

    ap.add_argument("--E0_forced", type=float, default=175.0)

    # smooth fit p0 (your defaults)
    ap.add_argument("--p0", default="9.6e-7,8.4,0.42",
                    help="Initial guess N0,E0,gamma as comma list")

    # Single-point params (if not scanning)
    ap.add_argument("--mmed", type=float, default=None)
    ap.add_argument("--gchi", type=float, default=None)
    ap.add_argument("--cgamma", type=float, default=None)
    ap.add_argument("--ctilde", type=float, default=None)
    ap.add_argument("--qeff", type=float, default=None)
    ap.add_argument("--Lambda", type=float, default=None)

    # Scan
    ap.add_argument("--scan", action="append", default=[],
                    help="param=log:min:max:n or param=list:a,b,c")
    ap.add_argument("--gchi_max", type=float, default=None, help="Perturbativity cut: skip gchi>gchi_max")
    ap.add_argument("--Lambda_min", type=float, default=None, help="EFT cut: skip Lambda<Lambda_min")

    ap.add_argument("--outdir", default="scan_out")
    ap.add_argument("--save_all_plots", action="store_true", help="Save plot for every scan point")
    ap.add_argument("--plot_best", action="store_true", help="Also save best-point plot (even if not saving all)")
    ap.add_argument(
        "--plot_best_deviant",
        action="store_true",
        help="Also save plot for most-deviant point subject to p>=pmin (even if not saving all)",
    )

    ap.add_argument(
        "--dip_anywhere",
        type=float,
        default=0.0,
        help="If >0, require a fractional dip anywhere: min_E (F_dm/F_smooth) <= 1 - dip_anywhere.",
    )
    ap.add_argument(
        "--plot_first_dip",
        action="store_true",
        help="If a dip-anywhere point is found, also save a plot for the first such scan point.",
    )
    ap.add_argument(
        "--stop_at_first_dip",
        action="store_true",
        help="Stop scanning as soon as the first dip-anywhere point is found (still writes outputs for points scanned so far).",
    )

    ap.add_argument(
        "--pmin",
        type=float,
        default=0.01,
        help="Goodness-of-fit requirement: require p-value >= pmin (default 0.01 corresponds to 99% CL compatibility).",
    )
    ap.add_argument(
        "--pmin_model",
        choices=["dm", "forced", "both"],
        default="dm",
        help="Which attenuated model must satisfy p >= pmin: 'dm' uses A_def, 'forced' uses A_forced, 'both' requires both.",
    )

    args = ap.parse_args()

    # file name compat
    filename = args.filename
    if filename is None:
        raise SystemExit("Provide --filename spectrum_data.txt")

    # L in cm
    if args.L_unit == "Gpc":
        L_cm = args.L * CM_PER_GPC
    elif args.L_unit == "kpc":
        L_cm = args.L * CM_PER_KPC
    else:
        L_cm = args.L

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data (your method)
    data = np.loadtxt(filename)
    E_data = data[:, 0]
    F_data = data[:, 1]
    F_err  = data[:, 2]

    # Reduced plotting/fit points
    E_plot, F_plot, F_err_plot = data_format(data, args.spacing)

    # Smooth fit on reduced points (your method)
    p0 = [float(x) for x in args.p0.split(",")]
    popt, pcov = curve_fit(
        smooth_flux_model,
        E_plot,
        F_plot,
        p0=p0,
        sigma=F_err_plot,
        absolute_sigma=True
    )
    N0_fit, E0_fit, gamma_fit = popt
    print("==============================================")
    print("Smooth fit (no DM):")
    print(f"N0_fit    = {N0_fit:.6e}")
    print(f"E0_fit    = {E0_fit:.6e}")
    print(f"gamma_fit = {gamma_fit:.6e}")
    print("==============================================")

    # Physical amplitude A_def
    A_def = A_from_mchi(args.mchi, args.rho, L_cm)
    print(f"Scattering A_def = rho*L/mchi = {A_def:.6e} (1/cm^2)")
    print("==============================================")

    # n_params in smooth model = 3 (N0,E0,gamma)
    n_params = 3
    F_smooth_at_Eplot = smooth_flux_model(E_plot, *popt)
    chi2_smooth, ndof, chi2red, pval = compute_chi2(F_plot, F_smooth_at_Eplot, F_err_plot, n_params)
    print(f"[Smooth@plotpoints] chi2={chi2_smooth:.2f} dof={ndof} (chi2red={chi2red:.2f}, p={pval:.3g})")

    # Build scan grid
    scan_dict = {}
    for kv in args.scan:
        k, arr = parse_scan_kv(kv)
        scan_dict[k] = arr

    # Defaults for non-scanned params
    defaults = dict(mmed=args.mmed, gchi=args.gchi, cgamma=args.cgamma, ctilde=args.ctilde, qeff=args.qeff, Lambda=args.Lambda)

    required = {
        "scalar": ["mmed", "gchi", "cgamma"],
        "pseudoscalar": ["mmed", "gchi", "ctilde"],
        "thomson": ["qeff"],
        "rayleigh_even": ["Lambda"],
        "rayleigh_odd": ["Lambda"],
        "graviton": [],
    }[args.mediator]

    grid_axes = {}
    for k in required:
        if k in scan_dict:
            grid_axes[k] = scan_dict[k]
        else:
            if defaults[k] is None:
                raise SystemExit(f"Need --{k} or --scan {k}=... for mediator {args.mediator}")
            grid_axes[k] = np.array([defaults[k]], dtype=float)

    keys = list(grid_axes.keys())
    grids = [grid_axes[k] for k in keys]
    n_points = int(np.prod([len(g) for g in grids])) if grids else 1
    print(f"[info] scanning {n_points} points for mediator={args.mediator}")

    # Fine grid for plotting
    E_fit = np.linspace(E_data.min(), E_data.max(), 400)

    # Precompute smooth curves (your method)
    F_smooth_fit = smooth_flux_model(E_fit, *popt)

    # Results rows
    rows = []
    best = None  # track best by chi2 at plotted points (DM)
    best_payload = None

    best_deviant = None  # track max deviation from smooth, subject to p>=pmin
    best_deviant_score = None
    best_deviant_payload = None

    first_dip = None
    first_dip_payload = None

    for values in (itertools.product(*grids) if grids else [()]):
        params = {}
        for k, v in zip(keys, values):
            params[k] = float(v)

        # cuts
        if "gchi" in params and args.gchi_max is not None and params["gchi"] > args.gchi_max:
            continue
        if "Lambda" in params and args.Lambda_min is not None and params["Lambda"] < args.Lambda_min:
            continue

        # dsigma + sigma spline using YOUR E_array choice (use E_data for spline)
        dsigma_func = make_dsigma_func(args.mediator, args.mchi, params, args.norm, args.regulator)
        sigma_spline, sigma_max_cm2 = get_sigma_spline(E_data, dsigma_func, args.mchi, args.ncos)

        # DM curves using your flux_dm
        F_dm_fit = flux_dm(E_fit, A_def, *popt, sigma_spline, sigma_max_cm2)

        # Dip-anywhere metric (compare to smooth)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_dm_fit = np.where(F_smooth_fit > 0, F_dm_fit / F_smooth_fit, np.nan)
        ratio_dm_fit = np.nan_to_num(ratio_dm_fit, nan=np.inf, posinf=np.inf, neginf=np.inf)
        imin = int(np.argmin(ratio_dm_fit))
        dip_min_ratio = float(ratio_dm_fit[imin])
        dip_min_E = float(E_fit[imin])
        dip_depth_found = float(max(0.0, 1.0 - dip_min_ratio))
        meets_dip_anywhere = bool(args.dip_anywhere > 0.0 and dip_min_ratio <= (1.0 - float(args.dip_anywhere)))

        # Forced A
        A_forced = compute_A_forced(args.E0_forced, E_data, F_data, N0_fit, E0_fit, gamma_fit, sigma_spline, sigma_max_cm2)
        F_dm_forced_fit = flux_dm(E_fit, A_forced, *popt, sigma_spline, sigma_max_cm2)

        # Evaluate models at E_plot (for chi2 and residuals)
        F_smooth_plot = smooth_flux_model(E_plot, *popt)
        F_dm_plot = flux_dm(E_plot, A_def, *popt, sigma_spline, sigma_max_cm2)
        F_dm_forced_plot = flux_dm(E_plot, A_forced, *popt, sigma_spline, sigma_max_cm2)

        chi2_dm, ndof_dm, chi2red_dm, p_dm = compute_chi2(F_plot, F_dm_plot, F_err_plot, n_params)
        chi2_forced, ndof_f, chi2red_f, p_f = compute_chi2(F_plot, F_dm_forced_plot, F_err_plot, n_params)

        # GOF requirement: p-value threshold ("compatible at 99% CL" => p >= 0.01)
        if args.pmin_model == "dm":
            meets_pmin = bool(np.isfinite(p_dm) and (float(p_dm) >= float(args.pmin)))
        elif args.pmin_model == "forced":
            meets_pmin = bool(np.isfinite(p_f) and (float(p_f) >= float(args.pmin)))
        else:
            meets_pmin = bool(
                np.isfinite(p_dm)
                and np.isfinite(p_f)
                and (float(p_dm) >= float(args.pmin))
                and (float(p_f) >= float(args.pmin))
            )

        # tau extrema (at E_data)
        sigma_cm2_data = sigma_spline(E_data) * sigma_max_cm2
        tau_data = A_def * sigma_cm2_data

        # y_eff text and title
        y_eff_tex = compute_yeff_tex(args.mediator, params)
        med_name = mediator_title(args.mediator)

        row = {
            "mediator": args.mediator,
            "mchi_GeV": args.mchi,
            "rho_GeV_cm3": args.rho,
            "L_cm": L_cm,
            "A_def_cm-2": A_def,
            "A_forced_cm-2": A_forced,
            "chi2_smooth": chi2_smooth,
            "chi2_dm": chi2_dm,
            "chi2_forced": chi2_forced,
            "p_dm": float(p_dm) if np.isfinite(p_dm) else np.nan,
            "p_forced": float(p_f) if np.isfinite(p_f) else np.nan,
            "pmin_req": float(args.pmin),
            "pmin_model": str(args.pmin_model),
            "pmin_meets": int(meets_pmin),
            "delta_chi2_smooth_minus_dm": chi2_smooth - chi2_dm,
            "delta_chi2_smooth_minus_forced": chi2_smooth - chi2_forced,
            "tau_max": float(np.max(tau_data)),
            "tau_min": float(np.min(tau_data)),
            "sigma_max_cm2": float(sigma_max_cm2),
            "dip_anywhere_req": float(args.dip_anywhere),
            "dip_anywhere_found": dip_depth_found,
            "dip_anywhere_min_ratio": dip_min_ratio,
            "dip_anywhere_min_E": dip_min_E,
            "dip_anywhere_meets": int(meets_dip_anywhere),
            "dip_and_pmin_meets": int(meets_dip_anywhere and meets_pmin),
            "y_eff_tex": y_eff_tex.replace(",", ";"),  # keep CSV safe-ish
        }
        for k in required:
            row[k] = params.get(k, np.nan)
        rows.append(row)

        # save plot for this point (YOU requested "save all the plots")
        if args.save_all_plots:
            # Residuals relative to smooth at E_data for bottom panel:
            # Use E_data (full) for residual curves; but plot points are E_plot
            F_smooth_data_full = smooth_flux_model(E_data, *popt)
            F_dm_data_full = flux_dm(E_data, A_def, *popt, sigma_spline, sigma_max_cm2)
            F_dm_forced_data_full = flux_dm(E_data, A_forced, *popt, sigma_spline, sigma_max_cm2)

            res_data_plot = (F_plot - smooth_flux_model(E_plot, *popt))
            res_smooth = np.zeros_like(E_data)
            res_dm = (F_dm_data_full - F_smooth_data_full)
            res_dm_forced = (F_dm_forced_data_full - F_smooth_data_full)

            # build a unique filename
            tag_parts = [args.mediator]
            for k in required:
                tag_parts.append(f"{k}{params[k]:.3g}")
            tag = "_".join(tag_parts).replace(".", "p").replace("-", "m")

            outfile = os.path.join(plots_dir, f"{tag}.png")
            plot_spectrum(
                E_data=E_data,
                E_plot=E_plot, F_plot=F_plot, F_err_plot=F_err_plot,
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
                y_eff_tex=y_eff_tex,
                mediator_name=med_name,
                outfile=outfile
            )

        # track best (min chi2_dm)
        if best is None or chi2_dm < best["chi2_dm"]:
            best = row
            best_payload = (params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced)

        # track most-deviant point among those that satisfy GOF (p>=pmin)
        # Deviation metric: maximum dip depth anywhere in the band, i.e. max(0, 1 - min_E(F_dm/F_smooth)).
        # This is monotonic in overall attenuation strength, and matches the user's request to be as far
        # from the smooth curve as possible while still compatible with the data.
        if meets_pmin:
            score = float(dip_depth_found)
            if best_deviant is None:
                best_deviant = row
                best_deviant_score = score
                best_deviant_payload = (params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced)
            else:
                assert best_deviant_score is not None
                if score > float(best_deviant_score) + 1e-15:
                    best_deviant = row
                    best_deviant_score = score
                    best_deviant_payload = (params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced)
                elif abs(score - float(best_deviant_score)) <= 1e-15:
                    # tie-breaker: prefer smaller chi2_dm
                    if chi2_dm < float(best_deviant.get("chi2_dm", np.inf)):
                        best_deviant = row
                        best_deviant_score = score
                        best_deviant_payload = (params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced)

        # track first dip-anywhere + GOF point (in scan order)
        if first_dip is None and (meets_dip_anywhere and meets_pmin):
            first_dip = row
            first_dip_payload = (params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced)
            if args.stop_at_first_dip:
                break

    if len(rows) == 0:
        raise SystemExit("No scan points survived cuts; loosen gchi_max/Lambda_min or adjust grid.")

    # Save CSV
    out_csv = os.path.join(args.outdir, f"scan_{args.mediator}_mchi{args.mchi:g}.csv")
    cols = list(rows[0].keys())
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[saved] {out_csv}")

    # Save best summary + plot
    out_best = os.path.join(args.outdir, f"best_{args.mediator}_mchi{args.mchi:g}.txt")
    with open(out_best, "w") as f:
        f.write("=== Best point (min chi2_dm at E_plot) ===\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")
    print(f"[saved] {out_best}")

    # Save best-deviant (subject to p>=pmin) summary + (optional) plot
    if best_deviant is not None:
        out_bestdev = os.path.join(args.outdir, f"bestdeviant_{args.mediator}_mchi{args.mchi:g}.txt")
        with open(out_bestdev, "w") as f:
            f.write("=== Most-deviant point subject to GOF(p>=pmin) ===\n")
            for k, v in best_deviant.items():
                f.write(f"{k}: {v}\n")
        print(f"[saved] {out_bestdev}")

        if args.plot_best_deviant and best_deviant_payload is not None:
            params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced = best_deviant_payload

            F_dm_fit = flux_dm(E_fit, A_def, *popt, sigma_spline, sigma_max_cm2)
            F_dm_forced_fit = flux_dm(E_fit, A_forced, *popt, sigma_spline, sigma_max_cm2)

            F_smooth_data_full = smooth_flux_model(E_data, *popt)
            F_dm_data_full = flux_dm(E_data, A_def, *popt, sigma_spline, sigma_max_cm2)
            F_dm_forced_data_full = flux_dm(E_data, A_forced, *popt, sigma_spline, sigma_max_cm2)

            res_data_plot = (F_plot - smooth_flux_model(E_plot, *popt))
            res_smooth = np.zeros_like(E_data)
            res_dm = (F_dm_data_full - F_smooth_data_full)
            res_dm_forced = (F_dm_forced_data_full - F_smooth_data_full)

            out_plot = os.path.join(args.outdir, f"bestdeviantplot_{args.mediator}_mchi{args.mchi:g}.png")
            plot_spectrum(
                E_data=E_data,
                E_plot=E_plot, F_plot=F_plot, F_err_plot=F_err_plot,
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
                y_eff_tex=y_eff_tex,
                mediator_name=med_name,
                outfile=out_plot
            )
            print(f"[saved] {out_plot}")
    else:
        print(f"[warn] No scan point satisfied GOF requirement p>=pmin (pmin={args.pmin}, model={args.pmin_model}).")

    # Save first-dip-anywhere + GOF summary + (optional) plot
    if first_dip is not None:
        out_first = os.path.join(args.outdir, f"firstdip_{args.mediator}_mchi{args.mchi:g}.txt")
        with open(out_first, "w") as f:
            f.write("=== First dip-anywhere + GOF(p>=pmin) point (in scan order) ===\n")
            for k, v in first_dip.items():
                f.write(f"{k}: {v}\n")
        print(f"[saved] {out_first}")

        if args.plot_first_dip and first_dip_payload is not None:
            params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced = first_dip_payload

            F_dm_fit = flux_dm(E_fit, A_def, *popt, sigma_spline, sigma_max_cm2)
            F_dm_forced_fit = flux_dm(E_fit, A_forced, *popt, sigma_spline, sigma_max_cm2)

            F_smooth_data_full = smooth_flux_model(E_data, *popt)
            F_dm_data_full = flux_dm(E_data, A_def, *popt, sigma_spline, sigma_max_cm2)
            F_dm_forced_data_full = flux_dm(E_data, A_forced, *popt, sigma_spline, sigma_max_cm2)

            res_data_plot = (F_plot - smooth_flux_model(E_plot, *popt))
            res_smooth = np.zeros_like(E_data)
            res_dm = (F_dm_data_full - F_smooth_data_full)
            res_dm_forced = (F_dm_forced_data_full - F_smooth_data_full)

            out_plot = os.path.join(args.outdir, f"firstdipplot_{args.mediator}_mchi{args.mchi:g}.png")
            plot_spectrum(
                E_data=E_data,
                E_plot=E_plot, F_plot=F_plot, F_err_plot=F_err_plot,
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
                y_eff_tex=y_eff_tex,
                mediator_name=med_name,
                outfile=out_plot
            )
            print(f"[saved] {out_plot}")

    if args.plot_best and best_payload is not None:
        params, y_eff_tex, med_name, sigma_spline, sigma_max_cm2, A_forced = best_payload

        F_dm_fit = flux_dm(E_fit, A_def, *popt, sigma_spline, sigma_max_cm2)
        F_dm_forced_fit = flux_dm(E_fit, A_forced, *popt, sigma_spline, sigma_max_cm2)

        F_smooth_data_full = smooth_flux_model(E_data, *popt)
        F_dm_data_full = flux_dm(E_data, A_def, *popt, sigma_spline, sigma_max_cm2)
        F_dm_forced_data_full = flux_dm(E_data, A_forced, *popt, sigma_spline, sigma_max_cm2)

        res_data_plot = (F_plot - smooth_flux_model(E_plot, *popt))
        res_smooth = np.zeros_like(E_data)
        res_dm = (F_dm_data_full - F_smooth_data_full)
        res_dm_forced = (F_dm_forced_data_full - F_smooth_data_full)

        out_plot = os.path.join(args.outdir, f"bestplot_{args.mediator}_mchi{args.mchi:g}.png")
        plot_spectrum(
            E_data=E_data,
            E_plot=E_plot, F_plot=F_plot, F_err_plot=F_err_plot,
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
            y_eff_tex=y_eff_tex,
            mediator_name=med_name,
            outfile=out_plot
        )
        print(f"[saved] {out_plot}")


if __name__ == "__main__":
    main()
