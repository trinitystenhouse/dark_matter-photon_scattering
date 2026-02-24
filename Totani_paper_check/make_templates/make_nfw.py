import numpy as np

R_SUN_KPC = 8.25
RS_KPC = 20.0
GAMMA = 1.25

S_MAX_KPC = 410.0   # closer to R0+rvir
N_S = 2048
R_MIN_KPC = 1e-3    # inner cutoff to avoid divergence


def nfw_rho(r, *, gamma=GAMMA, rs_kpc=RS_KPC):
    """gNFW density profile (unnormalized)."""
    x = np.maximum(r / rs_kpc, 1e-12)
    return x ** (-gamma) * (1.0 + x) ** (gamma - 3.0)


def make_nfw_template(lon_deg, lat_deg, *, gamma=GAMMA, rho_power=2.5, n_s=N_S, s_max_kpc=S_MAX_KPC, r_min_kpc=R_MIN_KPC, chunk_s=32):
    l = np.deg2rad(lon_deg).astype(np.float64)
    b = np.deg2rad(lat_deg).astype(np.float64)

    s = np.linspace(0.0, float(s_max_kpc), int(n_s), dtype=np.float64)
    ds = float(s[1] - s[0])

    cospsi = np.cos(b) * np.cos(l)
    intensity = np.zeros_like(cospsi, dtype=np.float64)

    for i in range(0, len(s), chunk_s):
        s_chunk = s[i:i+chunk_s][:, None, None]  # (chunk, ny, nx)
        r2 = (R_SUN_KPC**2 + s_chunk**2 - 2.0 * R_SUN_KPC * s_chunk * cospsi[None, :, :])
        r = np.sqrt(np.maximum(r2, 0.0))
        r = np.maximum(r, r_min_kpc)

        rho = nfw_rho(r, gamma=gamma)
        intensity += np.sum(rho ** rho_power, axis=0) * ds

    return intensity.astype(np.float32)


def make_nfw_rho25_template(lon_deg, lat_deg):
    return make_nfw_template(lon_deg, lat_deg, gamma=GAMMA, rho_power=2.5)