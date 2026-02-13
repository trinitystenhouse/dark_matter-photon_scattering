import numpy as np

# -------------------------
# Physical parameters
# -------------------------
R_SUN_KPC = 8.25
RS_KPC = 20.0
GAMMA = 1.25

# Line-of-sight sampling
S_MAX_KPC = 50.0
N_S = 512


def nfw_rho(r, *, gamma=GAMMA):
    """gNFW density profile (unnormalized)."""
    x = r / RS_KPC
    return 1.0 / (np.power(x, gamma) * np.power(1.0 + x, 3.0 - gamma))


def make_nfw_template(lon_deg, lat_deg, *, gamma=GAMMA, rho_power=2.5, n_s=N_S, chunk=8):
    """
    Compute NFW rho^2.5 line-of-sight template.

    Parameters
    ----------
    lon_deg, lat_deg : 2D arrays
        Galactic longitude and latitude in degrees.

    Returns
    -------
    template : 2D array
        Unnormalized NFW rho^2.5 intensity map.
    """

    # Convert to radians
    l = np.deg2rad(lon_deg)
    b = np.deg2rad(lat_deg)

    # LOS grid
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
        rho = nfw_rho(r, gamma=gamma)
        intensity += np.sum(np.power(rho, rho_power), axis=0, dtype=np.float32) * ds

    return intensity


def make_nfw_rho25_template(lon_deg, lat_deg):
    return make_nfw_template(lon_deg, lat_deg, gamma=GAMMA, rho_power=2.5, n_s=N_S, chunk=8)
