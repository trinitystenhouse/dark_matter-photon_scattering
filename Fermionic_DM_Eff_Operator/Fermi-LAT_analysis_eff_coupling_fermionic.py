import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from helpers.trinity_plotting import set_plot_style
from helpers.fermi_plotting import (
    latex_sci,
    add_hatched_region_from_contour,
    operator_title,
    make_combined_tau_vs_lambda_beamer,
)
from scipy.optimize import curve_fit
from scipy.stats import qmc
import argparse
import os
import sys

set_plot_style(style="dark")

REPO_DIR = os.environ.get(
    "REPO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)


LAMBDA = None
C_S = None
C_P = None
OPERATOR = "rayleigh_full"
FERMION_TYPE = "dirac"

# Astrophysical params
rho_chi_gc = 0.4   # GeV / cm^3
rho_chi_cosmic = 1.2e-6
L_gc       = 8.5e3 * 3.086e18  # 8.5 kpc in cm
L_cosmic   =  12e9 * 0.0003066014 * 3.086e21 

# Unit conversions
HC2_GEV2_TO_M2 = 3.89379e-32   # 1 GeV^-2 = 3.89379e-32 m^2
GEV2_TO_FB     = 3.89379e11    # 1 GeV^-2 = 3.89379e11 fb
ALPHA_EM = 1.0 / 137.035999084


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
                      frame="cm",      # "cm" or "lab"
                      in_SI=False,
                      c_s=None, c_p=None, Lambda=None,
                      operator=None,
                      fermion_type=None):
    """
    mchi    : GeV
    theta   : radians
    E_gamma : photon energy (GeV). Interpreted as:
              - CM photon energy if frame="cm"
              - Lab incoming photon energy (DM at rest) if frame="lab"
    returns : fb/sr (default) or m^2/sr if in_SI=True

    Operator conventions:
    - Operator names follow the photon-operator classification in Sec. 2.1 of arXiv:1810.00033.
    - For non-Rayleigh operators we use a canonical EFT normalization in the lab frame (DM at rest)
      to enforce the expected scaling with Lambda (dipoles ~ 1/Lambda^2, charge-radius/anapole ~ 1/Lambda^4).
    """

    if c_s is None:
        c_s = C_S
    if c_p is None:
        c_p = C_P
    if Lambda is None:
        Lambda = LAMBDA

    if operator is None:
        operator = OPERATOR
    if fermion_type is None:
        fermion_type = FERMION_TYPE

    if VERBOSE_DSIGMA:
        print("c_s:", c_s)
        print("c_p:", c_p)
        print("Lambda:", Lambda)

    if frame == "lab":
        s = get_s_lab_DMrest(mchi, E_gamma)
        t = get_t_lab_DMrest(mchi, E_gamma, theta)
    else:
        s = get_s_cm(mchi, E_gamma)
        t = get_t_cm(E_gamma, theta)

    operator = str(operator)
    fermion_type = str(fermion_type)

    if fermion_type not in ("dirac", "majorana"):
        raise ValueError(f"Unknown fermion_type={fermion_type!r}; expected 'dirac' or 'majorana'.")

    if operator == "rayleigh_even":
        val = (c_s**2 * (4 * mchi**2 - t) * t**2) / (Lambda**6 * 256 * np.pi**2 * s)
    elif operator == "rayleigh_odd":
        val = (c_p**2 * -t**3) / (Lambda**6 * 256 * np.pi**2 * s)
    elif operator in ("rayleigh_full", "full"):
        val = (
            (c_s**2 * (4 * mchi**2 - t) * t**2) / (Lambda**6 * 256 * np.pi**2 * s)
            + (c_p**2 * -t**3) / (Lambda**6 * 256 * np.pi**2 * s)
        )
    elif operator in ("dipole_magnetic", "dipole_electric"):
        if fermion_type == "majorana":
            return 0.0 * theta
        c = float(c_s) if operator == "dipole_magnetic" else float(c_p)
        # Correct dim-5 magnetic/electric dipole in lab frame (DM at rest).
        # dsigma/dOmega = |M|^2 / (64*pi^2 * s)
        # |M|^2 = 4 * alpha * c^2/Lambda^2 * (-t)
        # No propagator factor: this is a contact EFT operator.
        # Ref: canonical EFT normalization consistent with Sec. 2.1 of arXiv:1810.00033
        amp2 = 4.0 * ALPHA_EM * (c**2 / Lambda**2) * (-t)
        val = amp2 / (64.0 * np.pi**2 * s)

    elif operator in ("charge_radius", "anapole"):
        c = float(c_s) if operator == "charge_radius" else float(c_p)
        # Correct dim-6 charge radius / anapole.
        # |M|^2 = 4 * alpha * c^2/Lambda^4 * t^2
        amp2 = 4.0 * ALPHA_EM * (c**2 / Lambda**4) * (t**2)
        val = amp2 / (64.0 * np.pi**2 * s)
    else:
        raise ValueError(
            f"Unknown operator={operator!r}. "
            "Expected one of: rayleigh_even, rayleigh_odd, rayleigh_full, dipole_magnetic, dipole_electric, charge_radius, anapole."
        )

    #print(t)
    # print("t:", max(t))
    # print("s:", max(s))
    # print("Lambda2:", Lambda**2)

    return val * (HC2_GEV2_TO_M2 if in_SI else GEV2_TO_FB)


def compute_max_tau_grid(
    *,
    E_eval,
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
    Compute the maximum achievable optical depth tau_max(Lambda, mchi) on a
    2D grid, fixing the coupling to c=1 (maximum perturbative value).

    Used for exclusion/reach analysis: the contour tau_max = tau_needed is the
    sensitivity boundary — parameter space above it would have produced a
    visible attenuation signal and is excluded by the data.

    Returns
    -------
    dict with keys:
        'Lambda_grid'    : 1D array, shape (n_Lambda,)   [GeV]
        'mchi_grid'      : 1D array, shape (n_mchi,)     [GeV]
        'tau_grid'       : 2D array, shape (n_Lambda, n_mchi)
        'eft_valid_grid' : 2D bool array, shape (n_Lambda, n_mchi)
    """

    if operator is None:
        operator = OPERATOR
    if fermion_type is None:
        fermion_type = FERMION_TYPE

    log10_Lambda_min = float(log10_Lambda_min)
    log10_Lambda_max = float(log10_Lambda_max)
    log10_mchi_min = float(log10_mchi_min)
    log10_mchi_max = float(log10_mchi_max)

    Lambda_grid = np.logspace(log10_Lambda_min, log10_Lambda_max, int(n_Lambda))
    mchi_grid = np.logspace(log10_mchi_min, log10_mchi_max, int(n_mchi))

    tau_grid = np.zeros((int(n_Lambda), int(n_mchi)), dtype=float)
    eft_valid_grid = np.zeros((int(n_Lambda), int(n_mchi)), dtype=bool)

    operator = str(operator)
    fermion_type = str(fermion_type)

    n_Lambda = int(n_Lambda)
    progress_every = max(1, int(np.ceil(float(n_Lambda) / 5.0)))

    for iL, Lam in enumerate(Lambda_grid):
        Lam = float(Lam)
        if (iL % progress_every) == 0 or iL == (len(Lambda_grid) - 1):
            pct = int(round(100.0 * float(iL) / float(max(len(Lambda_grid) - 1, 1))))
            print(f"compute_max_tau_grid: {pct}%")

        for im, mchi in enumerate(mchi_grid):
            mchi = float(mchi)
            is_valid = eft_valid_kinematics_lab(
                mchi,
                Lam,
                float(omega_max_for_validity),
                float(eft_kinematic_factor),
            )
            eft_valid_grid[iL, im] = bool(is_valid)

            sigma_cm2_arr = sigma_array_cm2(
                E_eval,
                float(mchi),
                1.0,
                1.0,
                float(Lam),
                operator=operator,
                fermion_type=fermion_type,
            )
            sigma_cm2_arr = np.nan_to_num(sigma_cm2_arr, nan=0.0, posinf=0.0, neginf=0.0)
            sigma_cm2_arr = np.where(sigma_cm2_arr > 0.0, sigma_cm2_arr, 0.0)

            A = float(rho_chi) * float(L_cm) / float(max(mchi, 1e-30))
            tau_arr = float(A) * np.asarray(sigma_cm2_arr, dtype=float)
            tau_grid[iL, im] = float(np.max(tau_arr)) if tau_arr.size else 0.0

    return {
        "Lambda_grid": Lambda_grid,
        "mchi_grid": mchi_grid,
        "tau_grid": tau_grid,
        "eft_valid_grid": eft_valid_grid,
    }


def plot_tau_grid(
    *,
    Lambda_grid,
    mchi_grid,
    tau_grid,
    eft_valid_grid,
    tau_needed,
    tau_energy_label,
    outdir,
    operator,
    fermion_type=None,
    baseline,
    meta_text=None,
):
    """Plot exclusion/reach figures from a tau grid.

    Produces:
    - Figure A: 1D slice of tau_max(E_target) vs Lambda with EFT-invalid shading
    - Figure B: 2D map of log10(tau_max) in (log10 Lambda, log10 mchi) with
      sensitivity and EFT-validity contours.

    Parameters
    ----------
    Lambda_grid, mchi_grid, tau_grid, eft_valid_grid
        Outputs of compute_max_tau_grid.
    tau_needed : float
        Target optical depth corresponding to the chosen dip depth.
    tau_energy_label : str
        Label describing how tau_max was defined (single energy or band-based).
    outdir : str
        Output directory.
    operator : str
        Operator label for filenames.
    baseline : str
        Baseline label for filenames.
    meta_text : str or None
        Optional multi-line string to draw onto the plots listing run constants.
    """

    if fermion_type is None:
        fermion_type = FERMION_TYPE
    Lambda_grid = np.asarray(Lambda_grid, dtype=float)
    mchi_grid = np.asarray(mchi_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    eft_valid_grid = np.asarray(eft_valid_grid, dtype=bool)

    os.makedirs(outdir, exist_ok=True)

    best_mchi_idx = np.argmax(tau_grid, axis=1)
    tau_max_lambda = tau_grid[np.arange(tau_grid.shape[0]), best_mchi_idx]
    eft_valid_at_best = eft_valid_grid[np.arange(eft_valid_grid.shape[0]), best_mchi_idx]
    mchi_best_lambda = mchi_grid[np.asarray(best_mchi_idx, dtype=int)]
    mchi_best_lambda = np.asarray(mchi_best_lambda, dtype=float)
    mchi_best_lambda[~np.isfinite(mchi_best_lambda)] = np.nan
    mchi_best_lambda[mchi_best_lambda <= 0.0] = np.nan
    finite_m = mchi_best_lambda[np.isfinite(mchi_best_lambda)]
    mchi_best_is_constant = False
    mchi_best_const_val = None
    if finite_m.size > 0:
        mmin = float(np.nanmin(finite_m))
        mmax = float(np.nanmax(finite_m))
        if mmin > 0.0 and (mmax / mmin) < 1.02:
            mchi_best_is_constant = True
            mchi_best_const_val = float(np.nanmedian(finite_m))

    plot_text_fs = 9

    figL = plt.figure(figsize=(9.5, 4.0), constrained_layout=True)
    gsL = figL.add_gridspec(1, 2, width_ratios=[4.2, 1.8], wspace=0.10)
    axL = figL.add_subplot(gsL[0, 0])
    axLtxt = figL.add_subplot(gsL[0, 1])
    axLtxt.axis("off")
    axL.plot(Lambda_grid, np.asarray(tau_max_lambda, dtype=float), lw=2)
    axL.axhline(float(tau_needed), color="w", ls="--", lw=1, label="Target")

    axL2 = None
    if not mchi_best_is_constant:
        axL2 = axL.twinx()
        axL2.plot(
            Lambda_grid,
            np.asarray(mchi_best_lambda, dtype=float),
            color="c",
            lw=1.5,
            label=r"$m_{\chi,\,\mathrm{best}}(\Lambda)$",
        )

    side_text = str(meta_text) if (meta_text is not None) else ""
    if mchi_best_is_constant and (mchi_best_const_val is not None):
        extra = rf"$m_{{\chi,\,\mathrm{{best}}}}\approx {latex_sci(mchi_best_const_val)}\ \mathrm{{GeV}}$"
        side_text = (side_text + "\n" + extra) if side_text.strip() else extra

    if side_text.strip():
        axLtxt.text(
            0.0,
            0.5,
            side_text,
            transform=axLtxt.transAxes,
            ha="left",
            va="center",
            color="w",
            fontsize=plot_text_fs,
        )

    invalid = ~np.asarray(eft_valid_at_best, dtype=bool)
    if np.any(invalid):
        invalid_idx = np.where(invalid)[0]
        blocks = np.split(invalid_idx, np.where(np.diff(invalid_idx) != 1)[0] + 1)
        first = True
        for b in blocks:
            x0 = float(Lambda_grid[int(b[0])])
            x1 = float(Lambda_grid[int(b[-1])])
            axL.axvspan(
                x0,
                x1,
                color="tab:orange",
                alpha=0.15,
                label="EFT invalid" if first else None,
            )
            first = False

    axL.set_xscale("log")
    axL.set_yscale("log")
    if axL2 is not None:
        axL2.set_yscale("log")
    axL.set_xlabel(r"$\Lambda\,\,[\mathrm{GeV}]$")
    axL.set_ylabel(rf"$\tau_\max$ ({str(tau_energy_label)})")
    h1, l1 = axL.get_legend_handles_labels()
    if axL2 is not None:
        axL2.set_ylabel(r"$m_{\chi,\,\mathrm{best}}\,[\mathrm{GeV}]$")
        h2, l2 = axL2.get_legend_handles_labels()
        axL.legend(handles=(h1 + h2), labels=(l1 + l2), frameon=False)
    else:
        axL.legend(handles=h1, labels=l1, frameon=False)
    out_tau = os.path.join(
        outdir,
        f"tau_vs_lambda_{str(operator)}_{str(fermion_type)}_{str(baseline)}.png",
    )
    plt.savefig(out_tau)
    plt.close(figL)

    log10_tau = np.log10(np.asarray(tau_grid, dtype=float) + 1e-30)

    figG = plt.figure(figsize=(9.5, 5.5))
    gsG = figG.add_gridspec(1, 2, width_ratios=[4.9, 1.6], wspace=0.12)
    axG = figG.add_subplot(gsG[0, 0])
    axGtxt = figG.add_subplot(gsG[0, 1])
    axGtxt.axis("off")
    x = np.log10(Lambda_grid)
    y = np.log10(mchi_grid)
    X, Y = np.meshgrid(x, y, indexing="xy")

    levels = 40
    im = axG.contourf(
        X,
        Y,
        log10_tau.T,
        levels=levels,
        cmap="plasma",
    )
    cbar = figG.colorbar(im, ax=axG)
    cbar.set_label(r"$\log_{10}(\tau_{\max})$")

    if meta_text is not None and str(meta_text).strip():
        axGtxt.text(
            0.0,
            0.5,
            str(meta_text),
            transform=axGtxt.transAxes,
            ha="left",
            va="center",
            color="w",
            fontsize=plot_text_fs,
        )

    legend_handles = []

    has_overlap = False

    if float(tau_needed) > 0.0:
        tau_field = np.asarray(tau_grid, dtype=float).T
        eft_mask = np.asarray(eft_valid_grid, dtype=bool).T

        tau_field_valid = np.ma.masked_where(
            (~eft_mask) | (~np.isfinite(np.asarray(tau_field, dtype=float))),
            tau_field,
        )

        overlap = (
            np.asarray(eft_mask, dtype=bool)
            & np.isfinite(np.asarray(tau_field, dtype=float))
            & (np.asarray(tau_field, dtype=float) >= float(tau_needed))
        )

        has_overlap = bool(
            np.any(
                (~tau_field_valid.mask)
                & (np.asarray(tau_field_valid) >= float(tau_needed))
            )
        )
        if has_overlap:
            tau_max_valid = float(np.max(tau_field_valid))
            add_hatched_region_from_contour(
                ax=axG,
                X=X,
                Y=Y,
                Z=overlap.astype(float),
                level=0.5,
                upper_level=1.5,
                hatch="////",
                edgecolor="c",
                zorder=3,
                outline_lw=1.5,
            )

            legend_handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="c",
                    hatch="////",
                    label=r"EFT-valid + testable ($\tau\geq\tau_{\rm needed}$)",
                )
            )
        else:
            axG.text(
                0.02,
                0.98,
                "No EFT-valid & testable region",
                transform=axG.transAxes,
                ha="left",
                va="top",
                color="w",
                fontsize=plot_text_fs,
                bbox={"facecolor": "k", "alpha": 0.35, "edgecolor": "none"},
                zorder=6,
            )

        tau_ge_needed = (
            np.isfinite(np.asarray(tau_field, dtype=float))
            & (np.asarray(tau_field, dtype=float) >= float(tau_needed))
        )
        axG.contour(
            X,
            Y,
            tau_ge_needed.astype(float),
            levels=[0.5],
            colors="w",
            linewidths=2.0,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="w",
                lw=2.0,
                label=r"Sensitivity boundary ($\tau=\tau_{\rm needed}$)",
            )
        )

    axG.contour(
        X,
        Y,
        eft_valid_grid.T.astype(float),
        levels=[0.5],
        colors="r",
        linewidths=1.5,
        linestyles="--",
        zorder=5,
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="r",
            lw=1.5,
            ls="--",
            label="EFT validity limit",
        )
    )

    axG.set_xlabel(r"$\log_{10}(\Lambda/\mathrm{GeV})$")
    axG.set_ylabel(r"$\log_{10}(m_\chi/\mathrm{GeV})$")

    cdm_bound = 1e-3
    if float(np.min(mchi_grid)) <= float(cdm_bound) <= float(np.max(mchi_grid)):
        axG.axhline(
            float(np.log10(cdm_bound)),
            color="w",
            lw=1.5,
            ls=":",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="w",
                lw=1.5,
                ls=":",
                label=r"CDM bound ($m_\chi\geq 10^{-3}\,\mathrm{GeV}$)",
            )
        )

    if len(legend_handles) > 0:
        legend_anchor_y = 0.98 if has_overlap else 0.86
        axG.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, legend_anchor_y),
            borderaxespad=0.0,
        )

    out_grid = os.path.join(outdir, f"tau_grid_{str(operator)}_{str(baseline)}.png")
    figG.tight_layout()
    plt.savefig(out_grid)
    plt.close(figG)


def sigma_tot_params(E_gamma, mchi, c_s, c_p, Lambda, n_theta=300, *, operator=None, fermion_type=None):
    theta = np.linspace(0.0, np.pi, n_theta)
    dtheta = theta[1] - theta[0]

    dsdo = get_dsigma_dOmega(
        mchi,
        theta,
        E_gamma,
        frame="lab",
        in_SI=True,
        c_s=c_s,
        c_p=c_p,
        Lambda=Lambda,
        operator=operator,
        fermion_type=fermion_type,
    )
    dsdo = np.nan_to_num(dsdo, nan=0.0, posinf=0.0, neginf=0.0)

    integral = 2.0 * np.pi * np.sum(np.sin(theta) * dsdo) * dtheta
    return integral


def sigma_tot_params_cm2(E_gamma, mchi, c_s, c_p, Lambda, n_theta=300, *, operator=None, fermion_type=None):
    return float(
        sigma_tot_params(
            E_gamma,
            mchi,
            c_s,
            c_p,
            Lambda,
            n_theta=n_theta,
            operator=operator,
            fermion_type=fermion_type,
        )
    ) * 1e4


def sigma_array_cm2(E_array, mchi, c_s, c_p, Lambda, n_theta=300, *, operator=None, fermion_type=None):
    E_array = np.asarray(E_array, dtype=float)
    out = np.empty_like(E_array, dtype=float)
    for i, E in enumerate(E_array):
        out[i] = sigma_tot_params_cm2(
            float(E),
            float(mchi),
            float(c_s),
            float(c_p),
            float(Lambda),
            n_theta=n_theta,
            operator=operator,
            fermion_type=fermion_type,
        )
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
    mchi_min_gev,
    mchi_max_gev,
    cs_max,
    cp_max,
    require_cold_dm,
    log10_mchi_min,
    log10_mchi_max,
    log10_cs_min,
    log10_cs_max,
    log10_cp_min,
    log10_cp_max,
    n_samples,
    seed=0,
):
    rng = np.random.default_rng(int(seed))

    dip_depth = float(dip_depth)
    dip_depth = min(max(dip_depth, 0.0), 0.999999)
    tau_needed = -np.log(1.0 - dip_depth) if dip_depth > 0 else 0.0

    best_visible = None
    best_any = None

    mchi_min_gev = float(mchi_min_gev)
    mchi_max_gev = float(mchi_max_gev)
    cs_max = float(cs_max)
    cp_max = float(cp_max)
    require_cold_dm = bool(require_cold_dm)

    log10_mchi_min = max(float(log10_mchi_min), float(np.log10(max(mchi_min_gev, 1e-30))))
    log10_mchi_max = min(float(log10_mchi_max), float(np.log10(max(mchi_max_gev, 1e-30))))
    log10_cs_max = min(float(log10_cs_max), float(np.log10(max(cs_max, 1e-30))))
    log10_cp_max = min(float(log10_cp_max), float(np.log10(max(cp_max, 1e-30))))

    if float(log10_mchi_min) > float(log10_mchi_max):
        return None
    if float(log10_cs_min) > float(log10_cs_max):
        return None
    if float(log10_cp_min) > float(log10_cp_max):
        return None

    E_low_default = 10.0
    E_high = float(omega_max_for_validity)
    E_data_min = None
    try:
        E_data_global = globals().get("E_data", None)
        if E_data_global is not None:
            E_data_min = float(np.min(np.asarray(E_data_global, dtype=float)))
    except Exception:
        E_data_min = None
    E_low = max(float(E_data_min) if E_data_min is not None else float(E_low_default), 10.0)

    sampler = qmc.LatinHypercube(d=3, seed=int(seed))
    u = sampler.random(n=int(n_samples))

    for i in range(int(n_samples)):
        log10_mchi = float(log10_mchi_min) + float(u[i, 0]) * (float(log10_mchi_max) - float(log10_mchi_min))
        log10_cs = float(log10_cs_min) + float(u[i, 1]) * (float(log10_cs_max) - float(log10_cs_min))
        log10_cp = float(log10_cp_min) + float(u[i, 2]) * (float(log10_cp_max) - float(log10_cp_min))

        mchi = 10.0**log10_mchi
        cs = 10.0**log10_cs
        cp = 10.0**log10_cp

        if float(mchi) < float(mchi_min_gev) or float(mchi) > float(mchi_max_gev):
            continue
        if float(cs) > float(cs_max) or float(cp) > float(cp_max):
            continue
        if require_cold_dm and float(mchi) < 1e-3:
            continue

        if not eft_valid_kinematics_lab(mchi, Lambda, omega_max_for_validity, eft_kinematic_factor):
            continue

        sigma_cm2 = sigma_tot_params_cm2(
            float(E_target),
            float(mchi),
            float(cs),
            float(cp),
            float(Lambda),
            operator=OPERATOR,
            fermion_type=FERMION_TYPE,
        )
        if not np.isfinite(sigma_cm2) or sigma_cm2 <= 0:
            continue

        sigma_cm2_low = sigma_tot_params_cm2(
            float(E_low),
            float(mchi),
            float(cs),
            float(cp),
            float(Lambda),
            operator=OPERATOR,
            fermion_type=FERMION_TYPE,
        )
        if not np.isfinite(sigma_cm2_low) or sigma_cm2_low <= 0:
            continue

        sigma_cm2_high = sigma_tot_params_cm2(
            float(E_high),
            float(mchi),
            float(cs),
            float(cp),
            float(Lambda),
            operator=OPERATOR,
            fermion_type=FERMION_TYPE,
        )
        if not np.isfinite(sigma_cm2_high) or sigma_cm2_high <= 0:
            continue

        A_def = float(rho_chi) * float(L_cm) / float(mchi)
        tau_target = float(A_def) * float(sigma_cm2)
        tau_low = float(A_def) * float(sigma_cm2_low)
        tau_high = float(A_def) * float(sigma_cm2_high)

        tau_target_floor = max(float(tau_target), 1e-30)
        tau_needed_floor = max(float(tau_needed), 1e-30)

        target_term = -float(np.log(tau_target_floor / tau_needed_floor) ** 2)
        low_pen = -50.0 * float(max(0.0, float(tau_low) - 1e-3) ** 2)
        high_pen = -0.2 * float(max(0.0, float(tau_high) - 5.0) ** 2)

        complexity = float(max(0.0, float(np.log10(cs))) ** 2 + max(0.0, float(np.log10(cp))) ** 2)
        comp_pen = -0.1 * float(complexity)

        score = float(target_term + low_pen + high_pen + comp_pen)

        cand = {
            "mchi": float(mchi),
            "c_s": float(cs),
            "c_p": float(cp),
            "Lambda": float(Lambda),
            "sigma_cm2": float(sigma_cm2),
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            "A_def": float(A_def),
            "tau": float(tau_target),
            "tau_needed": float(tau_needed),
            "score": float(score),
        }
        cand_any = dict(cand)
        cand_any["meets_visibility"] = bool(float(tau_needed) <= 0.0 or float(tau_target) >= float(tau_needed))

        if best_any is None or cand_any["score"] > best_any["score"]:
            best_any = cand_any

        if cand_any["meets_visibility"]:
            if best_visible is None or cand_any["score"] > best_visible["score"]:
                best_visible = cand_any

    if best_visible is None:
        return best_any

    log10_mchi_center = float(np.log10(max(float(best_visible["mchi"]), 1e-30)))
    log10_cs_center = float(np.log10(max(float(best_visible["c_s"]), 1e-30)))
    log10_cp_center = float(np.log10(max(float(best_visible["c_p"]), 1e-30)))

    ref_log10_mchi_min = max(float(log10_mchi_min), log10_mchi_center - 1.5)
    ref_log10_mchi_max = min(float(log10_mchi_max), log10_mchi_center + 1.5)
    ref_log10_cs_min = max(float(log10_cs_min), log10_cs_center - 1.5)
    ref_log10_cs_max = min(float(log10_cs_max), log10_cs_center + 1.5)
    ref_log10_cp_min = max(float(log10_cp_min), log10_cp_center - 1.5)
    ref_log10_cp_max = min(float(log10_cp_max), log10_cp_center + 1.5)

    ref_best = find_visible_params_fixed_lambda(
        Lambda=float(Lambda),
        E_target=float(E_target),
        dip_depth=float(dip_depth),
        rho_chi=float(rho_chi),
        L_cm=float(L_cm),
        omega_max_for_validity=float(omega_max_for_validity),
        eft_kinematic_factor=float(eft_kinematic_factor) * 1.5,
        mchi_min_gev=float(mchi_min_gev),
        mchi_max_gev=float(mchi_max_gev),
        cs_max=float(cs_max),
        cp_max=float(cp_max),
        require_cold_dm=bool(require_cold_dm),
        log10_mchi_min=float(ref_log10_mchi_min),
        log10_mchi_max=float(ref_log10_mchi_max),
        log10_cs_min=float(ref_log10_cs_min),
        log10_cs_max=float(ref_log10_cs_max),
        log10_cp_min=float(ref_log10_cp_min),
        log10_cp_max=float(ref_log10_cp_max),
        n_samples=int(n_samples) * 2,
        seed=int(seed) + 1000003,
    )

    if ref_best is not None and bool(ref_best.get("meets_visibility", False)):
        if float(ref_best.get("score", -np.inf)) > float(best_visible.get("score", -np.inf)):
            return ref_best

    return best_visible


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
        default=1000.0,
        help="EFT scale.",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="rayleigh_full",
        choices=[
            "rayleigh_even",
            "rayleigh_odd",
            "rayleigh_full",
            "dipole_magnetic",
            "dipole_electric",
            "charge_radius",
            "anapole",
        ],
        help="Photon operator choice (see Sec. 2.1 of arXiv:1810.00033).",
    )
    parser.add_argument(
        "--fermion-type",
        type=str,
        default="dirac",
        choices=["dirac", "majorana"],
        help="Fermion type. Majorana forbids dipole operators; anapole/Rayleigh remain.",
    )
    parser.add_argument(
        "--c_s",
        type=float,
        default=4e-2,
        help="Even parity coupling constant (used if --find-visible not set).",
    )
    parser.add_argument(
        "--c_p",
        type=float,
        default=4e-2,
        help="Odd parity coupling constant (used if --find-visible not set).",
    )
    parser.add_argument(
        "--mchi",
        type=float,
        default=1.0,
        help="Dark matter mass in GeV (used if --find-visible not set).",
    )
    parser.add_argument(
        "--mchi-min-gev",
        type=float,
        default=1e-6,
        help="Physics prior: minimum DM mass (GeV) allowed in scan.",
    )
    parser.add_argument(
        "--mchi-max-gev",
        type=float,
        default=1e6,
        help="Physics prior: maximum DM mass (GeV) allowed in scan.",
    )
    parser.add_argument(
        "--cs-max",
        type=float,
        default=1.0,
        help="Physics prior: maximum c_s allowed in scan.",
    )
    parser.add_argument(
        "--cp-max",
        type=float,
        default=1.0,
        help="Physics prior: maximum c_p allowed in scan.",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=10.0,
        help="Physics prior: minimum EFT scale Lambda (GeV) allowed.",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1e5,
        help="Maximum Lambda (GeV) for --scan-lambda grid.",
    )
    parser.add_argument(
        "--scan-lambda",
        action="store_true",
        help="If set, scan over Lambda on a log-spaced grid between --lambda-min and --lambda-max.",
    )
    parser.add_argument(
        "--scan-lambda-n",
        type=int,
        default=20,
        help="Number of Lambda points for --scan-lambda.",
    )
    parser.add_argument(
        "--require-cold-dm",
        action="store_true",
        help="If set, require mchi >= 1e-3 GeV (MeV) in scan.",
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
        help="Fix Lambda and search over (mchi, c_s, c_p) for parameters that produce a visible attenuation (tau at dip-energy >= tau_needed).",
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
        help="If set, compute and plot tau_max(Lambda,mchi) grid at c=1.",
    )
    parser.add_argument(
        "--combined-limits",
        action="store_true",
        help="If set, make beamer-ready combined multi-panel tau_vs_lambda plots over operators.",
    )
    parser.add_argument(
        "--tau-energy-mode",
        type=str,
        default="band",
        choices=["band", "dip"],
        help="For --tau-grid: use 'band' (max over an energy band) or 'dip' (single energy at --dip-energy).",
    )
    parser.add_argument(
        "--tau-energy-min",
        type=float,
        default=None,
        help="For --tau-grid with --tau-energy-mode band: minimum energy in GeV (default: min energy in the input data).",
    )
    parser.add_argument(
        "--tau-energy-max",
        type=float,
        default=None,
        help="For --tau-grid with --tau-energy-mode band: maximum energy in GeV (default: max energy in the input data).",
    )
    parser.add_argument(
        "--tau-energy-n",
        type=int,
        default=60,
        help="For --tau-grid with --tau-energy-mode band: number of log-spaced energies in the band.",
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
        "--log10-cs-min",
        type=float,
        default=-4.0,
        help="log10(c_s) minimum for --find-visible.",
    )
    parser.add_argument(
        "--log10-cs-max",
        type=float,
        default=0.0,
        help="log10(c_s) maximum for --find-visible.",
    )
    parser.add_argument(
        "--log10-cp-min",
        type=float,
        default=-4.0,
        help="log10(c_p) minimum for --find-visible.",
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
        "--log10-cp-max",
        type=float,
        default=0.0,
        help="log10(c_p) maximum for --find-visible.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    argv = set(sys.argv[1:])
    legacy_mode = (
        ("--Lambda" not in argv)
        and ("--operator" not in argv)
        and ("--baseline" not in argv)
        and ("--scan-lambda" not in argv)
        and (not bool(args.is_cosmic))
    )

    if legacy_mode:
        args.Lambda = 2e7
        args.lambda_min = 1.0
        args.log10_cs_min = -6.0
        args.log10_cp_min = -12.0

    global VERBOSE_DSIGMA
    VERBOSE_DSIGMA = bool(args.verbose_dsigma)

    global LAMBDA, C_S, C_P

    global OPERATOR, FERMION_TYPE

    OPERATOR = str(args.operator)
    FERMION_TYPE = str(args.fermion_type)

    LAMBDA = float(args.Lambda)
    if float(LAMBDA) < float(args.lambda_min):
        raise ValueError(
            f"Lambda={float(LAMBDA):.6e} GeV is below --lambda-min={float(args.lambda_min):.6e} GeV. "
            "Increase --Lambda or lower --lambda-min."
        )
    C_S = float(args.c_s)
    C_P = float(args.c_p)
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
    print(f"c_s    = {C_S:.4e}")
    print(f"c_p    = {C_P:.4e}")
    print(f"Lambda    = {LAMBDA:.4e}")
    print(f"operator    = {OPERATOR}")
    print(f"fermion_type = {FERMION_TYPE}")
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
    A_def_check = float(RHO_CHI_SETTING) * float(L_SETTING) / float(max(mchi, 1e-30))
    sigma_required = float(tau_needed) / float(max(A_def_check, 1e-30))

    if bool(args.combined_limits):
        tau_mode = str(getattr(args, "tau_energy_mode", "band"))
        if tau_mode == "dip":
            E_eval = np.asarray([float(args.dip_energy)], dtype=float)
            tau_energy_label = rf"$E={float(args.dip_energy):g}\,\mathrm{{GeV}}$"
        else:
            Emin = float(args.tau_energy_min) if args.tau_energy_min is not None else float(np.min(E_data))
            Emax = float(args.tau_energy_max) if args.tau_energy_max is not None else float(np.max(E_data))
            if Emin <= 0.0 or Emax <= 0.0 or Emin >= Emax:
                raise ValueError(f"Invalid tau energy band: Emin={Emin}, Emax={Emax} (need 0 < Emin < Emax).")
            nE = int(max(2, getattr(args, "tau_energy_n", 60)))
            E_eval = np.logspace(np.log10(Emin), np.log10(Emax), nE)
            tau_energy_label = rf"$\max_{{E\in[{Emin:g},{Emax:g}]\,\mathrm{{GeV}}}}$"

        header_text = "  ".join(
            [
                rf"baseline={str(baseline)}",
                rf"$\rho_\chi={latex_sci(RHO_CHI_SETTING)}\,\mathrm{{GeV/cm^3}}$",
                rf"$L={latex_sci(L_SETTING)}\,\mathrm{{cm}}$",
                rf"$f_\mathrm{{EFT}}={float(args.eft_kinematic_factor):g}$",
                rf"$\tau_\mathrm{{needed}}={latex_sci(tau_needed)}$",
                rf"dip\ depth={float(dip_depth):g}",
                rf"{str(tau_energy_label)}",
            ]
        )

        operators = [
            "rayleigh_even",
            "rayleigh_odd",
            "rayleigh_full",
            "dipole_magnetic",
            "dipole_electric",
            "charge_radius",
            "anapole",
        ]

        for ft in ["dirac", "majorana"]:
            ops_ft = list(operators)
            if str(ft) == "majorana":
                ops_ft = [o for o in ops_ft if o not in ["dipole_magnetic", "dipole_electric"]]

            make_combined_tau_vs_lambda_beamer(
                operators=ops_ft,
                fermion_type=str(ft),
                baseline=str(baseline),
                E_eval=np.asarray(E_eval, dtype=float),
                tau_energy_label=str(tau_energy_label),
                tau_needed=float(tau_needed),
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
                outdir=str(args.outdir),
                header_text=str(header_text),
            )

        return

    if bool(args.tau_grid):
        tau_mode = str(getattr(args, "tau_energy_mode", "band"))
        if tau_mode == "dip":
            E_eval = np.asarray([float(args.dip_energy)], dtype=float)
            tau_energy_label = rf"$E={float(args.dip_energy):g}\,\mathrm{{GeV}}$"
        else:
            Emin = float(args.tau_energy_min) if args.tau_energy_min is not None else float(np.min(E_data))
            Emax = float(args.tau_energy_max) if args.tau_energy_max is not None else float(np.max(E_data))
            if Emin <= 0.0 or Emax <= 0.0 or Emin >= Emax:
                raise ValueError(f"Invalid tau energy band: Emin={Emin}, Emax={Emax} (need 0 < Emin < Emax).")
            nE = int(max(2, getattr(args, "tau_energy_n", 60)))
            E_eval = np.logspace(np.log10(Emin), np.log10(Emax), nE)
            tau_energy_label = rf"$\max_{{E\in[{Emin:g},{Emax:g}]\,\mathrm{{GeV}}}}$"

        meta_text = "\n".join(
            [
                f"operator={str(OPERATOR)}",
                f"fermion={str(FERMION_TYPE)}",
                f"baseline={str(baseline)}",
                rf"$m_\chi\in[10^{{{float(args.log10_mchi_min):g}}},10^{{{float(args.log10_mchi_max):g}}}]\ \mathrm{{GeV}}$",
                r"$\tau_{\max}(\Lambda)=\max_{m_\chi}\,\tau_{\max}(\Lambda,m_\chi)$",
                rf"$\rho_\chi={latex_sci(RHO_CHI_SETTING)}\ \mathrm{{GeV/cm^3}}$",
                rf"$L={latex_sci(L_SETTING)}\ \mathrm{{cm}}$",
                rf"$f_\mathrm{{EFT}}={float(args.eft_kinematic_factor):g}$",
                rf"$\tau_\mathrm{{needed}}={latex_sci(tau_needed)}$",
                rf"dip\ depth={float(dip_depth):g}",
                f"tau_mode={tau_mode}",
                rf"{str(tau_energy_label)}",
            ]
        )

        grid = compute_max_tau_grid(
            E_eval=E_eval,
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
            operator=str(OPERATOR),
            fermion_type=str(FERMION_TYPE),
        )

        plot_tau_grid(
            Lambda_grid=grid["Lambda_grid"],
            mchi_grid=grid["mchi_grid"],
            tau_grid=grid["tau_grid"],
            eft_valid_grid=grid["eft_valid_grid"],
            tau_needed=float(tau_needed),
            tau_energy_label=str(tau_energy_label),
            outdir=str(args.outdir),
            operator=str(OPERATOR),
            fermion_type=str(FERMION_TYPE),
            baseline=str(baseline),
            meta_text=str(meta_text),
        )

        Lambda_grid = np.asarray(grid["Lambda_grid"], dtype=float)
        tau_grid = np.asarray(grid["tau_grid"], dtype=float)
        best_mchi_idx = np.argmax(tau_grid, axis=1)
        tau_max_lambda = tau_grid[np.arange(tau_grid.shape[0]), best_mchi_idx]
        min_idx = np.where(np.asarray(tau_max_lambda, dtype=float) >= float(tau_needed))[0]
        if len(min_idx) > 0:
            i0 = int(min_idx[0])
            Lam0 = float(Lambda_grid[i0])
            mchi0 = float(np.asarray(grid["mchi_grid"], dtype=float)[int(best_mchi_idx[i0])])
            print("==============================================")
            print(f"Minimum Lambda with tau_max >= tau_needed in grid: {Lam0:.4e} GeV")
            print(f"Occurs at mchi ~ {mchi0:.4e} GeV")
            print("==============================================")
        else:
            print("==============================================")
            print("No Lambda in the scanned grid reaches tau_needed at c=1.")
            print("==============================================")

        if not bool(args.find_visible):
            return

    if OPERATOR == "dipole_magnetic":
        cs_test, cp_test = 1.0, 0.0
    elif OPERATOR == "dipole_electric":
        cs_test, cp_test = 0.0, 1.0
    elif OPERATOR == "charge_radius":
        cs_test, cp_test = 1.0, 0.0
    elif OPERATOR == "anapole":
        cs_test, cp_test = 0.0, 1.0
    else:
        cs_test, cp_test = 1.0, 1.0

    sigma_max = sigma_tot_params_cm2(
        float(args.dip_energy),
        float(mchi),
        float(cs_test),
        float(cp_test),
        float(args.lambda_min),
        operator=OPERATOR,
        fermion_type=FERMION_TYPE,
    )
    ratio = float(sigma_max) / float(max(sigma_required, 1e-30))
    if float(sigma_max) < 0.1 * float(sigma_required):
        print(
            "WARNING: Maximum achievable sigma ({:.2e} cm^2) is {:.1e}x below\n"
            "required ({:.2e} cm^2) even at c=1, Lambda={:.2e} GeV.\n"
            "Scan is unlikely to find visible parameters. Consider:\n"
            "  - Reducing --Lambda (currently {:.2e} GeV)\n"
            "  - Using --baseline cosmic for longer baseline\n"
            "  - Switching to a lower-dimension operator (--operator dipole_magnetic)"
        .format(float(sigma_max), float(ratio), float(sigma_required), float(args.lambda_min), float(LAMBDA))
        )

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
        if args.scan_lambda:
            lambdas = np.logspace(
                np.log10(float(args.lambda_min)),
                np.log10(float(args.lambda_max)),
                int(args.scan_lambda_n),
            )
            tau_vs_lambda = []
            best_visible_by_lambda = None
            min_lambda_visible = None

            for Lam in lambdas:
                best_L = find_visible_params_fixed_lambda(
                    Lambda=float(Lam),
                    E_target=float(args.dip_energy),
                    dip_depth=float(args.dip_depth),
                    rho_chi=float(RHO_CHI_SETTING),
                    L_cm=float(L_SETTING),
                    omega_max_for_validity=float(np.max(E_data)),
                    eft_kinematic_factor=float(args.eft_kinematic_factor),
                    mchi_min_gev=float(args.mchi_min_gev),
                    mchi_max_gev=float(args.mchi_max_gev),
                    cs_max=float(args.cs_max),
                    cp_max=float(args.cp_max),
                    require_cold_dm=bool(args.require_cold_dm),
                    log10_mchi_min=float(args.log10_mchi_min),
                    log10_mchi_max=float(args.log10_mchi_max),
                    log10_cs_min=float(args.log10_cs_min),
                    log10_cs_max=float(args.log10_cs_max),
                    log10_cp_min=float(args.log10_cp_min),
                    log10_cp_max=float(args.log10_cp_max),
                    n_samples=int(args.find_visible_samples),
                    seed=int(args.find_visible_seed),
                )
                tau_val = float(best_L.get("tau", 0.0)) if best_L is not None else 0.0
                tau_vs_lambda.append(tau_val)

                if best_L is not None and bool(best_L.get("meets_visibility", False)):
                    if min_lambda_visible is None:
                        min_lambda_visible = float(Lam)
                        best_visible_by_lambda = best_L

            if min_lambda_visible is not None:
                print("==============================================")
                print(f"Minimum Lambda with visible-effect point found: {float(min_lambda_visible):.4e} GeV")
                print("==============================================")
                best = best_visible_by_lambda
            else:
                best = None

            figL, axL = plt.subplots(figsize=(6.0, 4.0))
            axL.plot(lambdas, np.asarray(tau_vs_lambda, dtype=float), lw=2)
            axL.axhline(float(tau_needed), color="w", ls="--", lw=1)
            axL.set_xscale("log")
            axL.set_yscale("log")
            axL.set_xlabel(r"$\Lambda\,\,[\mathrm{GeV}]$")
            axL.set_ylabel(r"$\tau(E_{\rm target})$")
            out_tau = os.path.join(args.outdir, f"tau_vs_lambda_{OPERATOR}_baseline_{baseline}.png")
            figL.tight_layout()
            plt.savefig(out_tau)
            plt.close(figL)
        else:
            best = find_visible_params_fixed_lambda(
                Lambda=float(LAMBDA),
                E_target=float(args.dip_energy),
                dip_depth=float(args.dip_depth),
                rho_chi=float(RHO_CHI_SETTING),
                L_cm=float(L_SETTING),
                omega_max_for_validity=float(np.max(E_data)),
                eft_kinematic_factor=float(args.eft_kinematic_factor),
                mchi_min_gev=float(args.mchi_min_gev),
                mchi_max_gev=float(args.mchi_max_gev),
                cs_max=float(args.cs_max),
                cp_max=float(args.cp_max),
                require_cold_dm=bool(args.require_cold_dm),
                log10_mchi_min=float(args.log10_mchi_min),
                log10_mchi_max=float(args.log10_mchi_max),
                log10_cs_min=float(args.log10_cs_min),
                log10_cs_max=float(args.log10_cs_max),
                log10_cp_min=float(args.log10_cp_min),
                log10_cp_max=float(args.log10_cp_max),
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
            print(f"c_s    = {best['c_s']:.4e}")
            print(f"c_p    = {best['c_p']:.4e}")
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
        C_S = best["c_s"]
        C_P = best["c_p"]

        print("==============================================")
        print("Found visible-effect parameters (Lambda fixed):")
        print(f"mchi   = {mchi:.4e} GeV")
        print(f"c_s    = {C_S:.4e}")
        print(f"c_p    = {C_P:.4e}")
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
    sigma_cm2_fit = sigma_array_cm2(E_fit, mchi, C_S, C_P, LAMBDA, operator=OPERATOR, fermion_type=FERMION_TYPE)
    F_dm = flux_dm(E_fit, A_use, *popt, sigma_cm2_fit)

    # Make plot (residuals relative to smooth on top)

    F_smooth_plot = smooth_flux_model(E_plot, *popt)
    sigma_cm2_plot = sigma_array_cm2(E_plot, mchi, C_S, C_P, LAMBDA, operator=OPERATOR, fermion_type=FERMION_TYPE)
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
        f"spectrum_with_attenuation_Lambda_{LAMBDA:.2e}_cs_{C_S:.2e}_cp_{C_P:.2e}_mchi_{mchi:.2e}.png",
    )
    plt.savefig(outfile)
    plt.close(fig)

   

if __name__ == "__main__":
    main()

#INTERESTING PLOT
# ==============================================
# Found visible-effect parameters (Lambda fixed):
# mchi   = 5.1189e-09 GeV
# c_s    = 4.7599e-03
# c_p    = 9.7353e+00
# Lambda = 1.0000e-02 GeV
# sigma(E_target) = 5.8915e-28 cm^2
# A_def = 2.6617e+27 cm^-2
# tau(E_target) = 1.5681e+00 (needed 1.0050e-02)
# ==============================================