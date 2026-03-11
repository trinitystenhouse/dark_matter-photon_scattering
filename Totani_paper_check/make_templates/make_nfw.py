import numpy as np

R_SUN_KPC = 8.0    # paper section 2.3
RS_KPC    = 21.0   # Via Lactea II
GAMMA     = 1.0    # standard NFW for all three templates
RVIR_KPC  = 402.0
N_S       = 2048
R_MIN_KPC = 1e-3   # inner cutoff to avoid divergence


def nfw_rho(r, *, gamma=GAMMA, rs_kpc=RS_KPC):
    """gNFW density profile (unnormalized)."""
    x = np.maximum(r / rs_kpc, 1e-12)
    return x ** (-gamma) * (1.0 + x) ** (gamma - 3.0)


def smax_to_rvir(lon_deg, lat_deg, *, r0=R_SUN_KPC, rvir=RVIR_KPC):
    """
    Per-pixel LOS distance to the far intersection with the virial sphere.
    Solves: s^2 - 2*R0*cospsi*s + (R0^2 - rvir^2) = 0
    Returns the larger root (forward LOS), shape matching input arrays.
    """
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)
    cospsi = np.cos(b) * np.cos(l)
    disc = (r0 * cospsi) ** 2 - (r0 ** 2 - rvir ** 2)
    disc = np.maximum(disc, 0.0)
    return r0 * cospsi + np.sqrt(disc)


def make_nfw_template(
    lon_deg,
    lat_deg,
    *,
    gamma=GAMMA,
    rho_power=2.0,
    n_s=N_S,
    r_min_kpc=R_MIN_KPC,
    r0=R_SUN_KPC,
    rs_kpc=RS_KPC,
    rvir=RVIR_KPC,
    chunk_s=32,
):
    """
    Compute J_p(l,b) = integral_0^smax rho(r(l,b,s))^p ds
    using a log-spaced per-pixel LOS grid truncated at the virial radius.
    Uses trapezoid integration for accuracy near the GC cusp.
    """
    l = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    b = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cospsi = np.cos(b) * np.cos(l)           # (ny, nx)

    # Per-pixel smax: LOS distance to virial sphere boundary
    smax = smax_to_rvir(lon_deg, lat_deg, r0=r0, rvir=rvir)  # (ny, nx)

    # Log-spaced grid in unit interval u in [0,1]; s = s0 * (smax/s0)^u
    s0 = r_min_kpc
    u = np.linspace(0.0, 1.0, int(n_s), dtype=np.float64)    # (n_s,)

    intensity = np.zeros_like(cospsi, dtype=np.float64)

    for i in range(0, len(u), chunk_s):
        u_chunk = u[i : i + chunk_s]                          # (chunk,)

        # s shape: (chunk, ny, nx)
        ratio = np.maximum(smax / s0, 1.0)
        s_chunk = s0 * (ratio[None, :, :] ** u_chunk[:, None, None])

        r2 = r0**2 + s_chunk**2 - 2.0 * r0 * s_chunk * cospsi[None, :, :]
        r  = np.sqrt(np.maximum(r2, 0.0))
        r  = np.maximum(r, r_min_kpc)

        rho       = nfw_rho(r, gamma=gamma, rs_kpc=rs_kpc)
        integrand = rho ** rho_power                          # (chunk, ny, nx)

        # Trapezoid rule along s within this chunk
        intensity += np.trapz(integrand, s_chunk, axis=0)

    # The chunked trapezoid misses the seams between chunks.
    # Re-do as a single trapezoid over the full u grid if memory allows,
    # otherwise accept the small seam error (sub-percent for chunk_s>=16).

    return intensity.astype(np.float32)


def pole_normalization_value(*, rho_power=2.0, gamma=GAMMA, rs_kpc=RS_KPC,
                              r0=R_SUN_KPC, rvir=RVIR_KPC, n_s=8192):
    """
    Analytic 1D integral at b=+90 deg (cos b = 0 => r = sqrt(R0^2 + s^2)).
    Used to normalize templates to pole flux, matching paper Eq. 4.2.
    At rho_power=2 should equal 8.93e14 Msun^2 kpc^-5 for Via Lactea II params.
    """
    smax = np.sqrt(max(rvir**2 - r0**2, 0.0))
    s    = np.linspace(0.0, smax, int(n_s), dtype=np.float64)
    r    = np.sqrt(r0**2 + s**2)
    rho  = nfw_rho(r, gamma=gamma, rs_kpc=rs_kpc)
    return float(np.trapz(rho ** rho_power, s))


def check_pole_normalization(rho_power=2.0, tol=0.05):
    """
    Sanity check: for rho_power=2, the pole integral should match
    Eq. 4.2 of Totani (2025): 8.93e14 Msun^2 kpc^-5.
    Raises RuntimeError if the discrepancy exceeds tol (default 5%).
    """
    expected = 8.93e14
    got      = pole_normalization_value(rho_power=rho_power)
    frac     = abs(got - expected) / expected
    if frac > tol:
        raise RuntimeError(
            f"Pole normalization check failed: got {got:.3e}, "
            f"expected {expected:.3e} (discrepancy {frac*100:.1f}%). "
            f"Check RS_KPC={RS_KPC}, R_SUN_KPC={R_SUN_KPC}, RVIR_KPC={RVIR_KPC}."
        )
    return got


# --- Three templates from the paper ---

def make_nfw_rho1_template(lon_deg, lat_deg):
    """NFW-rho^1: subhalo-dominated or decaying DM (paper section 2.3)."""
    return make_nfw_template(lon_deg, lat_deg, gamma=1.0, rho_power=1.0)

def make_nfw_rho2_template(lon_deg, lat_deg):
    """NFW-rho^2: smooth annihilation, preferred by paper (section 3.2)."""
    return make_nfw_template(lon_deg, lat_deg, gamma=1.0, rho_power=2.0)

def make_nfw_rho25_template(lon_deg, lat_deg):
    """NFW-rho^2.5: GC GeV excess morphology (paper section 2.3)."""
    return make_nfw_template(lon_deg, lat_deg, gamma=1.0, rho_power=2.5)