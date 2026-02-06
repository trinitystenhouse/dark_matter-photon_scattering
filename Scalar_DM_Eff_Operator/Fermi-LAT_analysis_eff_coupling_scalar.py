import matplotlib.pyplot as plt
import numpy as np
from helpers.trinity_plotting import set_plot_style
from scipy.optimize import curve_fit
import argparse
import os

set_plot_style(style="dark")

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


LAMBDA = None
C_PHI = None

# Astrophysical params
rho_chi_gc = 0.4   # GeV / cm^3
rho_chi_cosmic = 1.2e-6
L_gc       = 8.5e3 * 3.086e18  # 8.5 kpc in cm
L_cosmic   = 12e9 * 0.0003066014 * 3.086e18 

# Unit conversions
HC2_GEV2_TO_M2 = 3.89379e-32   # 1 GeV^-2 = 3.89379e-32 m^2
GEV2_TO_FB     = 3.89379e11    # 1 GeV^-2 = 3.89379e11 fb


VERBOSE_DSIGMA = False


# ---------- Kinematics ----------

def get_s_lab_DMrest(mchi, omega):
    """Lab frame (DM at rest): s = mchi^2 + 2 mchi * omega."""
    return mchi**2 + 2*mchi*omega

def get_t_lab_DMrest(mchi, omega, theta):
    """
    Lab frame (DM at rest): Compton-like kinematics with m_e -> mchi.
    omega' = omega / (1 + (omega/mchi)*(1-cosθ))
    t = -2 * omega * omega' * (1 - cosθ)
    """
    denom = 1.0 + (omega/mchi)*(1.0 - np.cos(theta))
    omega_out = omega / denom
    return -2.0 * omega * omega_out * (1.0 - np.cos(theta))


def get_s_max_lab_DMrest(mchi, omega_max):
    return mchi**2 + 2.0 * mchi * omega_max


def get_t_abs_max_lab_DMrest(mchi, omega_max):
    denom = 1.0 + (2.0 * omega_max / mchi)
    return 4.0 * omega_max**2 / denom


# ---------- dσ/dΩ ----------

def get_dsigma_dOmega(mchi, theta, E_gamma, *,
                      frame="lab",      # kept for API similarity; scalar analysis uses lab (DM at rest)
                      in_SI=False,
                      c_phi=None, Lambda=None,
                      ):
    """
    mchi    : GeV
    theta   : radians
    E_gamma : photon energy (GeV). Interpreted as:
              - CM photon energy if frame="cm"
              - Lab incoming photon energy (DM at rest) if frame="lab"
    returns : fb/sr (default) or m^2/sr if in_SI=True
    """
    if c_phi is None:
        c_phi = C_PHI
    if Lambda is None:
        Lambda = LAMBDA

    s = get_s_lab_DMrest(mchi, E_gamma)
    t = get_t_lab_DMrest(mchi, E_gamma, theta)

    if VERBOSE_DSIGMA:
        print("c_phi:", c_phi)
        print("Lambda:", Lambda)

    val = (c_phi**2 * t**2) / (Lambda**4 * 256 * np.pi**2 * s)  # GeV^-2 / sr
    return val * (HC2_GEV2_TO_M2 if in_SI else GEV2_TO_FB)


def sigma_tot_params(E_gamma, mchi, c_phi, Lambda, n_theta=300):
    theta = np.linspace(0.0, np.pi, n_theta)
    dtheta = theta[1] - theta[0]

    dsdo = get_dsigma_dOmega(mchi, theta, E_gamma, frame="lab", in_SI=True, c_phi=c_phi, Lambda=Lambda)
    dsdo = np.nan_to_num(dsdo, nan=0.0, posinf=0.0, neginf=0.0)

    integral = 2.0 * np.pi * np.sum(np.sin(theta) * dsdo) * dtheta
    return integral


def sigma_tot_params_cm2(E_gamma, mchi, c_phi, Lambda, n_theta=300):
    return float(sigma_tot_params(E_gamma, mchi, c_phi, Lambda, n_theta=n_theta)) * 1e4


def sigma_array_cm2(E_array, mchi, c_phi, Lambda, n_theta=300):
    E_array = np.asarray(E_array, dtype=float)
    out = np.empty_like(E_array, dtype=float)
    for i, E in enumerate(E_array):
        out[i] = sigma_tot_params_cm2(float(E), float(mchi), float(c_phi), float(Lambda), n_theta=n_theta)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def eft_valid_kinematics_lab(mchi, Lambda, omega_max, eft_kinematic_factor):
    s_max = get_s_max_lab_DMrest(float(mchi), float(omega_max))
    t_abs_max = get_t_abs_max_lab_DMrest(float(mchi), float(omega_max))
    q2_max = float(max(s_max, t_abs_max))
    return (Lambda**2) >= (float(eft_kinematic_factor) * q2_max)


def find_visible_params_fixed_lambda(
    *,
    Lambda,
    E_target,
    dip_depth,
    rho_chi,
    L_cm,
    omega_max_for_validity,
    eft_kinematic_factor,
    log10_mchi_min,
    log10_mchi_max,
    log10_cphi_min,
    log10_cphi_max,
    n_samples,
    seed=0,
):
    rng = np.random.default_rng(int(seed))

    dip_depth = float(dip_depth)
    dip_depth = min(max(dip_depth, 0.0), 0.999999)
    tau_needed = -np.log(1.0 - dip_depth) if dip_depth > 0 else 0.0

    best_visible = None
    best_any = None

    for _ in range(int(n_samples)):
        log10_mchi = rng.uniform(float(log10_mchi_min), float(log10_mchi_max))
        log10_cphi = rng.uniform(float(log10_cphi_min), float(log10_cphi_max))

        mchi = 10.0**log10_mchi
        cphi = 10.0**log10_cphi

        if not eft_valid_kinematics_lab(mchi, Lambda, omega_max_for_validity, eft_kinematic_factor):
            continue

        sigma_cm2 = sigma_tot_params_cm2(float(E_target), float(mchi), float(cphi), float(Lambda))
        if not np.isfinite(sigma_cm2) or sigma_cm2 <= 0:
            continue

        A_def = float(rho_chi) * float(L_cm) / float(mchi)
        tau = A_def * float(sigma_cm2)

        margin = float(tau) - float(tau_needed)
        score = float(margin) - 0.05 * float(log10_cphi**2)

        cand = {
            "mchi": float(mchi),
            "c_phi": float(cphi),
            "Lambda": float(Lambda),
            "sigma_cm2": float(sigma_cm2),
            "A_def": float(A_def),
            "tau": float(tau),
            "tau_needed": float(tau_needed),
            "margin": float(margin),
            "score": float(score),
        }
        cand_any = dict(cand)
        cand_any["meets_visibility"] = bool(tau_needed <= 0 or margin >= 0.0)

        if best_any is None or cand_any["score"] > best_any["score"]:
            best_any = cand_any

        if cand_any["meets_visibility"]:
            if best_visible is None or cand_any["score"] > best_visible["score"]:
                best_visible = cand_any

    return best_visible if best_visible is not None else best_any


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

    # Build reduced arrays
    E_plot = E_data[selected_indices]
    F_plot = F_data[selected_indices]
    F_err_plot = F_err[selected_indices]

    print("Selected E values:", E_plot)

    return E_plot, F_plot, F_err_plot


def smooth_flux_model(E, N0, E0, gamma):
    """Power-law model: F(E) = N0 * (E/E0)^(-gamma)."""
    return N0 * (E / E0) ** (-gamma)


def flux_dm(E, A, N0_fit, E0_fit, gamma_fit, sigma_cm2_E):
    E = np.asarray(E, dtype=float)
    sigma_cm2_E = np.asarray(sigma_cm2_E, dtype=float)
    tau = A * sigma_cm2_E
    return smooth_flux_model(E, N0_fit, E0_fit, gamma_fit) * np.exp(-tau)


def main(): 
    parser = argparse.ArgumentParser(
        description="Fermi-LAT spectrum smooth fit + fixed-Lambda EFT scan for visible scattering attenuation"
    )
    parser.add_argument(
        "--is-cosmic",
        action="store_true",
        help="Use cosmic mean DM density and long baseline (default: use GC-like values).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=os.path.join(REPO_DIR, "fermi_data", "york", "processed", "spectrum_data.txt"),
        help="Input spectrum file [E, F, Ferr columns].",
    )
    parser.add_argument(
        "--Lambda",
        type=float,
        default=2e7,
        help="EFT scale.",
    )
    parser.add_argument(
        "--c_phi",
        type=float,
        default=4e-2,
        help="Scalar DM coupling constant (used if --find-visible not set).",
    )
 
    parser.add_argument(
        "--mchi",
        type=float,
        default=1.0,
        help="Dark matter mass in GeV (used if --find-visible not set).",
    )
    parser.add_argument(
        "--dip-energy",
        type=float,
        default=175.0,
        help="Energy (GeV) where you want a visible attenuation (used by --find-visible).",
    )
    parser.add_argument(
        "--dip-depth",
        type=float,
        default=0.01,
        help="Target fractional dip depth at dip-energy (e.g. 0.01 for 1%).",
    )
    parser.add_argument(
        "--eft-kinematic-factor",
        type=float,
        default=10.0,
        help="EFT validity: require Lambda^2 >= factor*max(s_max,|t|_max) over E<=E_max.",
    )
    parser.add_argument(
        "--verbose-dsigma",
        action="store_true",
        help="Print EFT params each time dσ/dΩ is evaluated (very noisy).",
    )
    parser.add_argument(
        "--find-visible",
        action="store_true",
        help="Fix Lambda and search over (mchi, c_phi) for parameters that produce a visible attenuation (tau at dip-energy >= tau_needed).",
    )
    parser.add_argument(
        "--find-visible-samples",
        type=int,
        default=200,
        help="Number of random samples for --find-visible.",
    )
    parser.add_argument(
        "--find-visible-seed",
        type=int,
        default=0,
        help="RNG seed for --find-visible.",
    )
    parser.add_argument(
        "--log10-mchi-min",
        type=float,
        default=-12.0,
        help="log10(mchi/GeV) minimum for --find-visible.",
    )
    parser.add_argument(
        "--log10-mchi-max",
        type=float,
        default=6.0,
        help="log10(mchi/GeV) maximum for --find-visible.",
    )
    parser.add_argument(
        "--log10-cphi-min",
        type=float,
        default=-6.0,
        help="log10(c_phi) minimum for --find-visible.",
    )
    parser.add_argument(
        "--log10-cphi-max",
        type=float,
        default=0.0,
        help="log10(c_phi) maximum for --find-visible.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    global VERBOSE_DSIGMA
    VERBOSE_DSIGMA = bool(args.verbose_dsigma)

    global LAMBDA, C_PHI

    LAMBDA = float(args.Lambda)
    C_PHI = float(args.c_phi)
    mchi = float(args.mchi)

    if args.is_cosmic:
        RHO_CHI_SETTING = rho_chi_cosmic
        L_SETTING = L_cosmic
    else:
        RHO_CHI_SETTING = rho_chi_gc
        L_SETTING = L_gc

    print("==============================================")
    print(f"mχ fixed    = {mchi:.4e} GeV")
    print(f"c_phi    = {C_PHI:.4e}")
    print(f"Lambda    = {LAMBDA:.4e}")
    print(f"ρχ          = {RHO_CHI_SETTING:.4e} GeV/cm^3")
    print(f"L           = {L_SETTING:.4e} cm")
    print("==============================================")

    # Load spectral data
    filename = args.filename
    data = np.loadtxt(filename)
    E_data = data[:, 0]
    F_data = data[:, 1]
    F_err  = data[:, 2]

    # Reduced data for fitting
    E_plot, F_plot, F_err_plot = data_format(data, 50)

    # -------- Smooth Flux Model (No Scattering) --------
    # Initial guess
    p0 = [9.6e-7, 8.4, 0.42]
    # Weighted fit to get a smooth baseline (no DM) line
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

    if args.find_visible:
        best = find_visible_params_fixed_lambda(
            Lambda=float(LAMBDA),
            E_target=float(args.dip_energy),
            dip_depth=float(args.dip_depth),
            rho_chi=float(RHO_CHI_SETTING),
            L_cm=float(L_SETTING),
            omega_max_for_validity=float(np.max(E_data)),
            eft_kinematic_factor=float(args.eft_kinematic_factor),
            log10_mchi_min=float(args.log10_mchi_min),
            log10_mchi_max=float(args.log10_mchi_max),
            log10_cphi_min=float(args.log10_cphi_min),
            log10_cphi_max=float(args.log10_cphi_max),
            n_samples=int(args.find_visible_samples),
            seed=int(args.find_visible_seed),
        )

        if best is None:
            raise RuntimeError("Scan produced no finite parameter points. Expand ranges or relax EFT factor.")

        meets_visibility = bool(best.get("meets_visibility", False))
        if not meets_visibility and float(best.get("tau_needed", 0.0)) > 0:
            print("==============================================")
            print("Best scan attempt (no visible-effect point found):")
            print(f"mchi   = {best['mchi']:.4e} GeV")
            print(f"c_phi  = {best['c_phi']:.4e}")
            print(f"Lambda = {LAMBDA:.4e} GeV")
            print(f"sigma(E_target) = {best['sigma_cm2']:.4e} cm^2")
            print(f"A_def = {best['A_def']:.4e} cm^-2")
            print(f"tau(E_target) = {best['tau']:.4e} (needed {best['tau_needed']:.4e})")
            print("==============================================")
            raise RuntimeError(
                "No visible-effect parameter point found in the scan range (tau < tau_needed). "
                "Expand coupling/mchi ranges, increase samples, lower dip-depth, or relax EFT factor."
            )

        mchi = best["mchi"]
        C_PHI = best["c_phi"]

        print("==============================================")
        print("Found visible-effect parameters (Lambda fixed):")
        print(f"mchi   = {mchi:.4e} GeV")
        print(f"c_phi  = {C_PHI:.4e}")
        print(f"Lambda = {LAMBDA:.4e} GeV")
        print(f"sigma(E_target) = {best['sigma_cm2']:.4e} cm^2")
        print(f"A_def = {best['A_def']:.4e} cm^-2")
        print(f"tau(E_target) = {best['tau']:.4e} (needed {best['tau_needed']:.4e})")
        print("==============================================")

    os.makedirs(args.outdir, exist_ok=True)

    # Fine Grid for plotting
    E_fit = np.linspace(E_data.min(), E_data.max(), 400)
    F_smooth_fit = smooth_flux_model(E_fit, *popt)

    A_use = (RHO_CHI_SETTING * L_SETTING / mchi)
    sigma_cm2_fit = sigma_array_cm2(E_fit, mchi, C_PHI, LAMBDA)
    F_dm = flux_dm(E_fit, A_use, *popt, sigma_cm2_fit)

    # Make plot (residuals relative to smooth on top)

    F_smooth_plot = smooth_flux_model(E_plot, *popt)
    sigma_cm2_plot = sigma_array_cm2(E_plot, mchi, C_PHI, LAMBDA)
    F_dm_plot = flux_dm(E_plot, A_use, *popt, sigma_cm2_plot)
    res_data_plot = (F_plot - F_smooth_plot)
    res_data_err_plot = F_err_plot
    res_smooth = F_smooth_fit - F_smooth_fit
    res_dm_fit = F_dm - F_smooth_fit

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(7.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    ax_top.axhline(0.0, color="gray", lw=1, alpha=0.6)


    ax_top.errorbar(
        E_plot,
        res_data_plot,
        yerr=res_data_err_plot,
        fmt="o",
        ms=4,
        lw=1,
        capsize=2,
        label="Data - Smooth",
    )
    
    ax_top.plot(E_fit, res_smooth, label="Smooth")
    ax_top.plot(E_fit, res_dm_fit, label="(Smooth×Atten) - Smooth")

    ax_top.set_ylabel(r"Residual")
    ax_top.grid(alpha=0.3)
    ax_top.legend()

    ax_bot.errorbar(
        E_plot,
        F_plot * 1e6,
        yerr=F_err_plot * 1e6,
        fmt="o",
        ms=4,
        lw=1,
        capsize=2,
        label="FERMI-LAT Data",
    )
    ax_bot.plot(E_fit, F_smooth_fit * 1e6, linestyle="--", label="Smooth")
    ax_bot.plot(E_fit, F_dm * 1e6, label="Smooth × Attenuation")
    ax_bot.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_bot.set_ylabel(r"$E^2\,dN/dE$ [GeV cm$^{-2}$ s$^{-1}$] × 10$^{-6}$")
    ax_bot.grid(alpha=0.3)
    ax_bot.legend()

    plt.tight_layout()
    outfile = os.path.join(
        args.outdir,
        f"spectrum_with_attenuation_Lambda_{LAMBDA:.2e}_cphi_{C_PHI:.2e}_mchi_{mchi:.2e}.png",
    )
    plt.savefig(outfile)
    plt.close(fig)

   

if __name__ == "__main__":
    main()
