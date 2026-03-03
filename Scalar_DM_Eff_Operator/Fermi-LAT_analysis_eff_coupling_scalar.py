import matplotlib.pyplot as plt
import numpy as np
from helpers.trinity_plotting import set_plot_style
from scipy.optimize import curve_fit
from scipy.stats import qmc
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
L_cosmic   =  12e9 * 0.0003066014 * 3.086e21 

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


def compute_max_tau_grid(
    *,
    E_target,
    rho_chi,
    L_cm,
    omega_max_for_validity,
    eft_kinematic_factor,
    log10_Lambda_min,
    log10_Lambda_max,
    log10_mchi_min,
    log10_mchi_max,
    n_Lambda=40,
    n_mchi=40,
    operator=None,
    fermion_type=None,
):
    """
    Compute tau_max(Lambda, mchi) on a 2D grid at c=1 (maximum coupling).

    Returns a dict with keys:
        'Lambda_grid'   : 1D array, shape (n_Lambda,)
        'mchi_grid'     : 1D array, shape (n_mchi,)
        'tau_grid'      : 2D array, shape (n_Lambda, n_mchi)
        'eft_valid_grid': 2D bool array, shape (n_Lambda, n_mchi)
    """

    log10_Lambda_min = float(log10_Lambda_min)
    log10_Lambda_max = float(log10_Lambda_max)
    log10_mchi_min = float(log10_mchi_min)
    log10_mchi_max = float(log10_mchi_max)

    Lambda_grid = np.logspace(log10_Lambda_min, log10_Lambda_max, int(n_Lambda))
    mchi_grid = np.logspace(log10_mchi_min, log10_mchi_max, int(n_mchi))

    tau_grid = np.zeros((int(n_Lambda), int(n_mchi)), dtype=float)
    eft_valid_grid = np.zeros((int(n_Lambda), int(n_mchi)), dtype=bool)

    for iL, Lam in enumerate(Lambda_grid):
        Lam = float(Lam)
        for im, mchi in enumerate(mchi_grid):
            mchi = float(mchi)
            is_valid = eft_valid_kinematics_lab(
                mchi,
                Lam,
                float(omega_max_for_validity),
                float(eft_kinematic_factor),
            )
            eft_valid_grid[iL, im] = bool(is_valid)
            if not is_valid:
                tau_grid[iL, im] = 0.0
                continue

            sigma_cm2 = sigma_tot_params_cm2(
                float(E_target),
                float(mchi),
                1.0,
                float(Lam),
            )
            if not np.isfinite(sigma_cm2) or float(sigma_cm2) <= 0.0:
                tau_grid[iL, im] = 0.0
                continue

            A = float(rho_chi) * float(L_cm) / float(max(mchi, 1e-30))
            tau_grid[iL, im] = float(A) * float(sigma_cm2)

    return {
        "Lambda_grid": Lambda_grid,
        "mchi_grid": mchi_grid,
        "tau_grid": tau_grid,
        "eft_valid_grid": eft_valid_grid,
    }


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
    sampler = qmc.LatinHypercube(d=2, seed=int(seed))

    dip_depth = float(dip_depth)
    dip_depth = min(max(dip_depth, 0.0), 0.999999)
    tau_needed = -np.log(1.0 - dip_depth) if dip_depth > 0 else 0.0

    best_visible = None
    best_any = None

    u = sampler.random(n=int(n_samples))

    for i in range(int(n_samples)):
        log10_mchi = float(log10_mchi_min) + float(u[i, 0]) * (float(log10_mchi_max) - float(log10_mchi_min))
        log10_cphi = float(log10_cphi_min) + float(u[i, 1]) * (float(log10_cphi_max) - float(log10_cphi_min))

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
        "--tau-grid",
        action="store_true",
        help="If set, compute and plot tau_max(Lambda,mchi) grid at c_phi=1.",
    )
    parser.add_argument(
        "--log10-Lambda-min",
        type=float,
        default=-1.0,
        help="log10(Lambda/GeV) minimum for --tau-grid.",
    )
    parser.add_argument(
        "--log10-Lambda-max",
        type=float,
        default=7.0,
        help="log10(Lambda/GeV) maximum for --tau-grid.",
    )
    parser.add_argument(
        "--tau-grid-n-lambda",
        type=int,
        default=40,
        help="Number of Lambda grid points for --tau-grid.",
    )
    parser.add_argument(
        "--tau-grid-n-mchi",
        type=int,
        default=40,
        help="Number of mchi grid points for --tau-grid.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["gc", "cosmic", "custom"],
        help="Baseline choice. If not provided, falls back to legacy --is-cosmic behavior.",
    )
    parser.add_argument(
        "--rho-chi",
        type=float,
        default=None,
        help="Custom baseline: rho_chi in GeV/cm^3 (used if --baseline custom).",
    )
    parser.add_argument(
        "--L-cm",
        type=float,
        default=None,
        help="Custom baseline: path length in cm (used if --baseline custom).",
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

    baseline = args.baseline
    if baseline is None:
        baseline = "cosmic" if bool(args.is_cosmic) else "gc"

    if baseline == "gc":
        RHO_CHI_SETTING = rho_chi_gc
        L_SETTING = L_gc
    elif baseline == "cosmic":
        RHO_CHI_SETTING = rho_chi_cosmic
        L_SETTING = L_cosmic
    elif baseline == "custom":
        if args.rho_chi is None or args.L_cm is None:
            raise ValueError("For --baseline custom you must provide both --rho-chi and --L-cm.")
        RHO_CHI_SETTING = float(args.rho_chi)
        L_SETTING = float(args.L_cm)
    else:
        raise ValueError(f"Unknown baseline={baseline!r}")

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

    dip_depth = float(args.dip_depth)
    dip_depth = min(max(dip_depth, 0.0), 0.999999)
    tau_needed = -np.log(1.0 - dip_depth) if dip_depth > 0 else 0.0

    if bool(args.tau_grid):
        os.makedirs(args.outdir, exist_ok=True)

        grid = compute_max_tau_grid(
            E_target=float(args.dip_energy),
            rho_chi=float(RHO_CHI_SETTING),
            L_cm=float(L_SETTING),
            omega_max_for_validity=float(np.max(E_data)),
            eft_kinematic_factor=float(args.eft_kinematic_factor),
            log10_Lambda_min=float(getattr(args, "log10_Lambda_min")),
            log10_Lambda_max=float(getattr(args, "log10_Lambda_max")),
            log10_mchi_min=float(args.log10_mchi_min),
            log10_mchi_max=float(args.log10_mchi_max),
            n_Lambda=int(getattr(args, "tau_grid_n_lambda")),
            n_mchi=int(getattr(args, "tau_grid_n_mchi")),
        )

        Lambda_grid = np.asarray(grid["Lambda_grid"], dtype=float)
        mchi_grid = np.asarray(grid["mchi_grid"], dtype=float)
        tau_grid = np.asarray(grid["tau_grid"], dtype=float)
        eft_valid_grid = np.asarray(grid["eft_valid_grid"], dtype=bool)

        best_mchi_idx = np.argmax(tau_grid, axis=1)
        tau_max_lambda = tau_grid[np.arange(tau_grid.shape[0]), best_mchi_idx]
        eft_valid_at_best = eft_valid_grid[np.arange(eft_valid_grid.shape[0]), best_mchi_idx]

        figL, axL = plt.subplots(figsize=(6.0, 4.0))
        axL.plot(Lambda_grid, np.asarray(tau_max_lambda, dtype=float), lw=2)
        axL.axhline(float(tau_needed), color="w", ls="--", lw=1)

        invalid = ~np.asarray(eft_valid_at_best, dtype=bool)
        if np.any(invalid):
            invalid_idx = np.where(invalid)[0]
            blocks = np.split(invalid_idx, np.where(np.diff(invalid_idx) != 1)[0] + 1)
            for b in blocks:
                x0 = float(Lambda_grid[int(b[0])])
                x1 = float(Lambda_grid[int(b[-1])])
                axL.axvspan(x0, x1, color="gray", alpha=0.2)

        axL.set_xscale("log")
        axL.set_yscale("log")
        axL.set_xlabel(r"$\Lambda\,\,[\mathrm{GeV}]$")
        axL.set_ylabel(r"$\tau_{\max}(E_{\rm target})$")
        out_tau = os.path.join(args.outdir, f"tau_vs_lambda_rayleigh_scalar_{baseline}.png")
        figL.tight_layout()
        plt.savefig(out_tau)
        plt.close(figL)

        tau_plot = np.where(tau_grid > 0.0, tau_grid, np.nan)
        log10_tau = np.log10(tau_plot)

        figG, axG = plt.subplots(figsize=(6.5, 5.5))
        extent = [
            float(np.log10(Lambda_grid[0])),
            float(np.log10(Lambda_grid[-1])),
            float(np.log10(mchi_grid[0])),
            float(np.log10(mchi_grid[-1])),
        ]
        im = axG.imshow(
            log10_tau.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        cbar = figG.colorbar(im, ax=axG)
        cbar.set_label(r"$\log_{10}(\tau_{\max})$")

        if float(tau_needed) > 0.0:
            axG.contour(
                np.log10(Lambda_grid),
                np.log10(mchi_grid),
                log10_tau.T,
                levels=[float(np.log10(tau_needed))],
                colors="w",
                linewidths=1.5,
            )

        axG.contour(
            np.log10(Lambda_grid),
            np.log10(mchi_grid),
            eft_valid_grid.T.astype(float),
            levels=[0.5],
            colors="k",
            linewidths=1.5,
        )

        axG.set_xlabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$")
        axG.set_ylabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$")
        out_grid = os.path.join(args.outdir, f"tau_grid_rayleigh_scalar_{baseline}.png")
        figG.tight_layout()
        plt.savefig(out_grid)
        plt.close(figG)

        return

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
