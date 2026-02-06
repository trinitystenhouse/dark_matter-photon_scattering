from webbrowser import get
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.integrate import simpson
from helpers.trinity_plotting import set_plot_style
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import argparse
import os
from scipy.stats import chi2

from helpers.path_utils import resolve_path

set_plot_style(style="dark")

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)


# --- constants ---
alpha = 1/137.035999084
v = 246.0
mW, mH, GammaH = 80.379, 125.25, 4.07e-3
mtop = 172.5
Nc_top, Q_top = 3, 2/3

# Fixed DM mass
MCHI_FIXED = None  # GeV

# Astrophysical params
rho_chi_gc = 0.4   # GeV / cm^3
rho_chi_cosmic = 1.2e-6
L_gc       = 8.5e3 * 3.086e18  # 8.5 kpc in cm
L_cosmic   = 12e9 * 0.0003066014 * 3.086e18 

# Unit conversions
HC2_GEV2_TO_M2 = 3.89379e-32   # 1 GeV^-2 = 3.89379e-32 m^2
GEV2_TO_FB     = 3.89379e11    # 1 GeV^-2 = 3.89379e11 fb


# ---------- Loop functions (H -> γγ) ----------

def get_pref(y_eff):
    gW = 2.0 * mW / v
    return np.sqrt((alpha**2) * (gW**4) * (y_eff**2) / ((4.0*np.pi)**2 * (mW**4)))

def f_scalar(tau):
    """
    f(τ) loop function.
    Works for scalar or numpy array τ (real or complex).
    """
    tau_arr = np.asarray(tau, dtype=complex)
    out = np.zeros_like(tau_arr, dtype=complex)

    mask_real = tau_arr.real >= 1.0
    if np.any(mask_real):
        out[mask_real] = np.arcsin(1.0 / np.sqrt(tau_arr[mask_real]))**2

    mask_cmplx = ~mask_real
    if np.any(mask_cmplx):
        root = np.sqrt(1.0 - tau_arr[mask_cmplx])
        out[mask_cmplx] = -0.25 * (
            np.log((1 + root) / (1 - root)) - 1j*np.pi
        )**2

    if np.ndim(tau) == 0:
        return out.item()
    return out

def A1_over2(tau):  # spin-1/2
    tau_arr = np.asarray(tau, dtype=complex)
    return -2.0 * tau_arr * (1.0 + (1.0 - tau_arr) * f_scalar(tau_arr))

def A1(tau):        # spin-1
    tau_arr = np.asarray(tau, dtype=complex)
    return 2.0 + 3.0 * tau_arr + 3.0 * (2.0 * tau_arr - tau_arr**2) * f_scalar(tau_arr)

def IW_IF_from_t(t, mW_, mferm_, eps=1e-18):
    """
    Given Mandelstam t, return (I_W, I_F) loop integrals.
    """
    t = np.asarray(t, dtype=float)
    IW = np.zeros_like(t, dtype=complex)
    IF = np.zeros_like(t, dtype=complex)
    safe = ~np.isclose(t, 0.0, atol=eps, rtol=0.0)
    if np.any(safe):
        betaW = -4.0*(mW_**2)/t[safe]
        betaf = -4.0*(mferm_**2)/t[safe]
        IW[safe] = A1(betaW)
        IF[safe] = A1_over2(betaf)
    return (IW.item(), IF.item()) if IW.ndim == 0 else (IW, IF)


# ---------- Kinematics ----------

def get_s_cm(mchi, k):
    """CoM frame: s = (sqrt(mchi^2 + k^2) + k)^2, with k = E_gamma."""
    return (np.sqrt(mchi**2 + k**2) + k)**2

def get_s_lab_DMrest(mchi, omega):
    """Lab frame (DM at rest): s = mchi^2 + 2 mchi * omega."""
    return mchi**2 + 2*mchi*omega

def get_t_cm(Eg, theta):
    """CoM frame: t = -2 Eg^2 (1 - cos(theta))."""
    return -2.0 * Eg**2 * (1.0 - np.cos(theta))

def get_t_lab_DMrest(mchi, omega, theta):
    """
    Lab frame (DM at rest): Compton-like kinematics with m_e -> mchi.
    omega' = omega / (1 + (omega/mchi)*(1-cosθ))
    t = -2 * omega * omega' * (1 - cosθ)
    """
    denom = 1.0 + (omega/mchi)*(1.0 - np.cos(theta))
    omega_out = omega / denom
    return -2.0 * omega * omega_out * (1.0 - np.cos(theta))


# ---------- dσ/dΩ ----------

def get_dsigma_dOmega(
    mchi,
    y_eff, # effective Higgs-portal coupling
    theta,
    E_gamma,
    *,         
    frame="cm",            # "cm" or "lab"
    in_SI=False,
    mW_=mW, mH_=mH, GammaH_=GammaH,
    mferm_=mtop, Nc_=Nc_top, Qf_=Q_top,
    which="full"
):
    """
    mchi      : DM mass [GeV]
    theta     : scattering angle [rad]
    E_gamma   : photon energy [GeV]
    y_eff  : effective Higgs-portal coupling (free parameter)

    returns   : fb/sr (default) or m^2/sr if in_SI=True
    """
    # --- kinematics ---
    if frame == "lab":
        s = get_s_lab_DMrest(mchi, E_gamma)
        t = get_t_lab_DMrest(mchi, E_gamma, theta)
    else:
        s = get_s_cm(mchi, E_gamma)
        t = get_t_cm(E_gamma, theta)

    # --- loops ---
    gW = 2.0 * mW_ / v
    IW, IF = IW_IF_from_t(t, mW_, mferm_)

    if which == "F":   # fermion only (top)
        amp = Nc_ * (Qf_**2) * IF
    elif which == "W": # W only
        amp = IW
    else:              # both
        amp = IW + Nc_ * (Qf_**2) * IF

    amp2 = (amp * np.conjugate(amp)).real

    # Prefactor ∝ ychi_eff^2
    pref = (y_eff**2) 

    tpart = 3.0 * (t**2) / 8.0
    prop  = (2.0*mchi**2 - 0.5*t) / (((t - mH_**2)**2) + (mH_**2)*(GammaH_**2))
    phase = amp2 / (64.0 * (np.pi**2) * s)

    val = pref * tpart * prop * phase   # GeV^-2 / sr

    return val * (HC2_GEV2_TO_M2 if in_SI else GEV2_TO_FB)


# vectorised over theta, leaving frame & in_SI as keywords
get_dsigma_dOmega_vec = np.vectorize(get_dsigma_dOmega, excluded={"frame", "in_SI"})


# σ(E) from dσ/dΩ
def sigma_tot(E_gamma, mchi, y_eff, n_theta=300):
    """
    Total cross section σ(E) = ∫ dΩ (dσ/dΩ)
    E_gamma : scalar (GeV)
    mchi    : scalar (GeV)
    returns σ (m^2)
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    dtheta = theta[1] - theta[0]

    dsdo = get_dsigma_dOmega(
        mchi, y_eff, theta, E_gamma,
        frame="lab", in_SI=True, which="full"
    )  # ychi_eff=1 for the shape
    dsdo = np.nan_to_num(dsdo, nan=0.0, posinf=0.0, neginf=0.0)

    integral = 2.0 * np.pi * np.sum(np.sin(theta) * dsdo) * dtheta
    return integral  # m^2

sigma_tot_vec = np.vectorize(sigma_tot)


def data_format(data, spacing):
    """
    Load spectral data
    Selects only some points for plotting
    assumes data has columns:
       E(GeV)   flux   flux_err
    data : array
    spacing : energy spacing between plotted points (GeV)
    returns E_plot, F_plot, F_err_plot
    """

    # Ensure sorted
    idx_sort = np.argsort(data[:, 0])
    E_data = data[:, 0][idx_sort]
    F_data = data[:, 1][idx_sort]
    F_err  = data[:, 2][idx_sort]

    # Make a linearly spaced grid (SPACING GeV spacing) 
    E_min = np.floor(E_data.min() / spacing) * spacing
    E_max = np.ceil(E_data.max()  / spacing) * spacing
    E_lin = np.arange(E_min, E_max + spacing, spacing)

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


def smooth_flux_model(E, N0, E0, gamma):
    """
    Power-law model: F(E) = N0 * (E/E0)^(-gamma)
    """
    return N0 * (E / E0)**(-gamma)



# σ(E) from dσ/dΩ
def sigma_tot(E_gamma, y_eff, mchi, n_theta=300):
    """
    Total cross section σ(E) = ∫ dΩ (dσ/dΩ)
    E_gamma : scalar (GeV)
    mchi    : scalar (GeV)
    y_eff   : effective coupling
    returns σ (m^2)
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    dtheta = theta[1] - theta[0]

    # get_dsigma_dOmega(..., in_SI=True) returns m^2/sr
    dsdo = get_dsigma_dOmega(mchi, y_eff, theta, E_gamma,
                             frame="lab", in_SI=True, which="full")
    dsdo = np.nan_to_num(dsdo, nan=0.0, posinf=0.0, neginf=0.0)

    integral = 2.0 * np.pi * np.sum(np.sin(theta) * dsdo) * dtheta
    return integral  # m^2
sigma_tot_vec = np.vectorize(sigma_tot)

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


def smooth_flux_model(E, N0, E0, gamma):
    """
    Power-law model: F(E) = N0 * (E/E0)^(-gamma)
    """
    return N0 * (E / E0)**(-gamma)


def get_sigma_spline(E_array, y_eff, mchi):
    """
    Returns:
        sigma_spline: spline over unit-normalised σ(E) shape
        sigma_max_cm2: max σ(E) in cm^2
    """
    sigma_raw = sigma_tot_vec(E_array, y_eff, mchi)   # m^2
    sigma_cm2 = sigma_raw * 1e4                # m^2 -> cm^2
    sigma_max_cm2 = sigma_cm2.max()

    sigma_shape = sigma_cm2 / sigma_max_cm2    # dimensionless, max = 1
    sigma_spline = UnivariateSpline(E_array, sigma_shape, s=0, k=3)
    return sigma_spline, sigma_max_cm2



def mchi_from_A(A, rho_chi, L):
    return rho_chi * L / A

def A_from_mchi(mchi, rho_chi, L):
    return rho_chi * L / mchi

def tau_shape(E, A, sigma_spline, sigma_max_cm2):
    """
    tau(E) = A * σ_max_cm2 * σ_shape(E),
    where A = ρ_χ L / m_χ (cm^-2) and σ_max_cm2 is in cm^2,
    so tau is dimensionless.
    """
    E = np.asarray(E)
    sigma_shape_E = sigma_spline(E)       # dimensionless, ≤1
    return A * sigma_max_cm2 * sigma_shape_E


def flux_dm(E, A, N0_fit, E0_fit, gamma_fit, sigma_spline, sigma_max_cm2):
    tau = tau_shape(E, A, sigma_spline, sigma_max_cm2)
    return smooth_flux_model(E, N0_fit, E0_fit, gamma_fit) * np.exp(-tau)

def flux_dm_forced(E, A_forced, N0_fit, E0_fit, gamma_fit, sigma_spline, sigma_max_cm2):
    return flux_dm(E, A_forced, N0_fit, E0_fit, gamma_fit, sigma_spline, sigma_max_cm2)


# Make plot
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
    y_eff,
    outfile="plots/final_plot.png"
):
    pref = y_eff

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
        label = rf"Flux with Scattering $y_\mathrm{{eff}}$ = {pref:.2g}"

    )

    ax_top.set_ylabel(r'$E^2\,dN/dE$ [GeV cm$^{-2}$ s$^{-1}$] × 10$^{-6}$')
    ax_top.set_xscale("linear")
    ax_top.grid(alpha=0.3, which="both", axis="both")
    ax_top.legend()

    # --- Bottom panel: residuals relative to smooth flux ---
    ax_bot.errorbar(
        E_plot, res_data_plot*1e9,
        yerr=F_err_plot*1e9,
        fmt="o", ms=4, lw=1, capsize=2,
        label="Smooth vs Data"
    )

    ax_bot.plot(
        E_data, res_smooth*1e9,
        linestyle="--",
        label="Smooth"
    )
    ax_bot.plot(
        E_data, res_dm_forced*1e9,
        linestyle=":",
        label="Smooth vs Forced-DM"
    )
    ax_bot.plot(
        E_data, res_dm*1e9,
        linestyle="-",
        label=rf"Smooth vs DM $y_\mathrm{{eff}}$ = {pref:.2g}"
    )

    ax_bot.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_bot.set_ylabel(r"Residuals ×10$^{-9}$")
    ax_bot.grid(alpha=0.3)
    # ax_bot.legend()

    plt.tight_layout()
    plt.savefig(outfile)
    #plt.show()


def compute_chi2(F_data, F_model, F_err, n_params):
    """
    Compute chi^2, reduced chi^2 and p-value for a given model.

    F_data  : data flux values
    F_model : model flux values evaluated at the same energies
    F_err   : 1σ uncertainties on the data
    n_params: number of free parameters in the model
    """
    F_data  = np.asarray(F_data)
    F_model = np.asarray(F_model)
    F_err   = np.asarray(F_err)

    chi2_val = np.sum(((F_data - F_model) / F_err)**2)
    ndof = len(F_data) - n_params
    chi2_red = chi2_val / ndof
    p_val = chi2.sf(chi2_val, ndof)  # tail probability

    return chi2_val, ndof, chi2_red, p_val


def print_gof(label, F_data, F_model, F_err, n_params):
    chi2_val, ndof, chi2_red, p_val = compute_chi2(F_data, F_model, F_err, n_params)
    print(f"[{label}]  χ² = {chi2_val:.2f}  for {ndof} dof  "
          f"(χ²_red = {chi2_red:.2f},  p = {p_val:.3g})")
    return chi2_val, ndof


def find_max_dip(mchi, E_data, F_data, F_err, popt, A_def, sigma_spline, sigma_max_cm2):
    # Plot residuals
    F_smooth_at_data = smooth_flux_model(E_data, *popt) #compare all to smooth

    # DM natural vs smooth
    F_dm_at_data = flux_dm(E_data, A_def, *popt, sigma_spline, sigma_max_cm2)
    res_dm = F_dm_at_data - F_smooth_at_data

    dip = res_dm / F_data
    print("DIP IS ", dip.max() )
    return dip.max()


def main(): 
    parser = argparse.ArgumentParser(
        description="GC spectrum fit with/without DM scattering (fixed mchi = 1e19 GeV)"
    )
    parser.add_argument(
        "--is-cosmic",
        action="store_true",
        help="Use cosmic mean DM density and long baseline (default: use GC-like values if available).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=os.path.join(REPO_DIR, "fermi_data", "york", "processed", "spectrum_data.txt"),
        help="Input spectrum file [E, F, Ferr columns].",
    )
    parser.add_argument(
        "--y_eff",
        type=float,
        default=2e17,
        help="Effective coupling.",
    )
    parser.add_argument(
        "--mchi",
        type=float,
        default=1.0e3,
        help="Dark matter mass.",
    )
    args = parser.parse_args()

    MCHI_FIXED = args.mchi  # GeV

    # Choose astrophysical configuration
    if args.is_cosmic:
        L_SETTING = L_cosmic
        RHO_CHI_SETTING = rho_chi_cosmic
    else:
        L_SETTING = L_gc
        RHO_CHI_SETTING = rho_chi_gc

    mchi = MCHI_FIXED
    y_eff = args.y_eff

    print("==============================================")
    print(f"mχ fixed    = {mchi:.4e} GeV")
    print(f"y_eff    = {y_eff:.4e} GeV")
    print(f"ρχ          = {RHO_CHI_SETTING:.4e} GeV/cm^3")
    print(f"L           = {L_SETTING:.4e} cm")
    print("==============================================")

    # Physical amplitude: A_phys = ρχ L / mχ_fixed
    A_phys = A_from_mchi(mchi, RHO_CHI_SETTING, L_SETTING)
    print(f"Scattering A_phys = ρχ L / mχ = {A_phys:.6e} cm^-2")
    print("==============================================")

    # Load spectral data
    filename = resolve_path(args.filename, start=__file__)
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
    # Natural amplitude from physics: A = ρχ L / mχ
    A_def = A_from_mchi(mchi, RHO_CHI_SETTING, L_SETTING)
    print(f"Scattering A_def = ρχ L / mχ = {A_def:.6e}")
    print("==============================================")

   # ----- Including Scattering -----

    # Forced fit
    E0_forced = 175.0
    idx_175 = np.argmin(np.abs(E_data - E0_forced))
    F175_data = F_data[idx_175]
    if F175_data <= 0:
        raise ValueError("Flux at 175 GeV must be positive.")

    # Value of sigma_shape at 175 GeV
    sigma_spline, sigma_max_cm2 = get_sigma_spline(E_data, y_eff, mchi)

    # Smooth model at 175 GeV
    F175_smooth = smooth_flux_model(E0_forced, N0_fit, E0_fit, gamma_fit)

    # Physical σ at 175 GeV:
    sigma175_shape = sigma_spline(E0_forced)
    sigma175_cm2   = sigma175_shape * sigma_max_cm2

    # Forcing: F_DM(175) = F_data(175)
    # F_smooth * exp(-A_forced * σ175_cm2) = F_data
    A_forced = -np.log(F175_data / F175_smooth) / sigma175_cm2
    # note this has no physical relevance

    # Fine Grid for plotting
    E_fit = np.linspace(E_data.min(), E_data.max(), 400)
    F_smooth_fit = smooth_flux_model(E_fit, *popt)
    F_dm_fit = flux_dm(E_fit, A_def, *popt, sigma_spline, sigma_max_cm2)
    F_dm_forced_fit = flux_dm(E_fit, A_forced, *popt, sigma_spline, sigma_max_cm2)

    # Plot residuals
    F_smooth_at_data = smooth_flux_model(E_data, *popt) #compare all to smooth
    F_smooth_at_data_plot = smooth_flux_model(E_plot, *popt)

    # Actual data vs smooth
    res_data_plot = F_plot - F_smooth_at_data_plot
    # Smooth vs itself (line)
    res_smooth = F_smooth_at_data - F_smooth_at_data
    # DM natural vs smooth
    F_dm_at_data = flux_dm(E_data, A_def, *popt, sigma_spline, sigma_max_cm2)
    res_dm = F_dm_at_data - F_smooth_at_data
    # Forced DM vs mooth
    F_dm_forced_at_data = flux_dm(E_data, A_forced, *popt, sigma_spline, sigma_max_cm2)
    res_dm_forced = F_dm_forced_at_data - F_smooth_at_data

    # Make plot
    pref = get_pref(y_eff)
    plot_spectrum(
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
        y_eff,
        outfile=f"plots/final_plot_y_eff_{y_eff:.2e}_mchi_{mchi:.2e}.png"
    )

    find_max_dip(mchi, E_data, F_data, F_err, popt, A_def, sigma_spline, sigma_max_cm2)
    # Goodness of fit
    # Models evaluated on the *fitted* points:
    F_smooth_plot      = smooth_flux_model(E_data, *popt)
    F_dm_plot          = flux_dm(E_data, A_def,    *popt, sigma_spline, sigma_max_cm2)

    # Number of parameters:
    # smooth model: N0, E0, gamma  -> 3
    # DM model: smooth + A_def     -> 4

    chi2_smooth, ndof_smooth = print_gof(
        "Smooth (no DM)",
        F_data,
        F_smooth_plot,
        F_err,
        n_params=3
    )

    chi2_dm, ndof_dm = print_gof(
        "DM (natural A_def)",
        F_data,
        F_dm_plot,
        F_err,
        n_params=3  
    )


    # Compare smooth vs DM (A_def or A_forced)
    delta_chi2 = chi2_smooth - chi2_dm
    delta_ndof = ndof_dm - ndof_smooth   # should be -1 if DM has 1 extra parameter

    p_improve = chi2.sf(delta_chi2, df=1)  # significance of improvement with 1 extra parameter

    print(f"Δχ² (smooth - DM natural) = {delta_chi2:.2f},  p_improve = {p_improve:.3g}")


    if delta_chi2 <= 0:
        print("DM model does not improve the fit (χ² is larger or equal).")
    elif delta_chi2 < 3.84:
        print("Improvement is marginal (< 95% CL). Not compelling.")
    elif delta_chi2 < 6.63:
        print("Improvement ~ 2σ (95–99% CL). Mild evidence at best.")
    else:
        print("Improvement > 99% CL (Δχ² > 6.63). Statistically significant.")


if __name__ == "__main__":
    main()
    # Example: python Fermi-LAT_analysis2.py --is-cosmic --y_eff 1e4 --filename spectrum_data.txt
