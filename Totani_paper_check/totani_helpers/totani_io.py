import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter


def read_counts_and_ebounds(counts_path):
    with fits.open(counts_path) as h:
        counts = h[0].data.astype(float)
        hdr = h[0].header
        eb = h["EBOUNDS"].data

    Emin = eb["E_MIN"].astype(float) / 1000.0
    Emax = eb["E_MAX"].astype(float) / 1000.0
    Ectr = np.sqrt(Emin * Emax)
    dE = (Emax - Emin)
    return counts, hdr, Emin, Emax, Ectr, dE

def set_lon_ticks_wrapped(ax, wcs, ny, nx, lons_deg=(-60, -30, 0, 30, 60)):
    xs = []
    for L in lons_deg:
        L360 = L % 360
        x, _ = wcs.world_to_pixel_values(L360, 0.0)  # place ticks at b=0
        xs.append(x)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{L:d}" for L in lons_deg])

def read_exposure(expo_path):
    with fits.open(expo_path) as h:
        expo = h[0].data.astype(float)
        E_expo = None
        if "ENERGIES" in h:
            col0 = h["ENERGIES"].columns.names[0]
            E_expo = np.array(h["ENERGIES"].data[col0], dtype=float)
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E_expo = np.sqrt(Emin * Emax)
    return expo, E_expo

def read_expcube_energies_mev(hdul):
    """
    Read exposure cube energies and return a 1D array in MeV.

    Requires an ENERGIES extension (preferred). If you ever encounter an expcube without it,
    fail loudly rather than guessing.
    """
    if "ENERGIES" not in hdul:
        raise RuntimeError("Exposure cube missing ENERGIES extension; can't resample exposure vs energy.")
    col0 = hdul["ENERGIES"].columns.names[0]
    E = np.array(hdul["ENERGIES"].data[col0], dtype=float)

    # Sanity: for your pipeline, MeV scale is ~1e3..1e6.
    # If clearly keV-scale, convert.
    if np.nanmax(E) > 1e8 and np.nanmin(E) > 1e5:
        E = E / 1000.0

    if not np.all(np.isfinite(E)) or np.any(E <= 0):
        raise RuntimeError("Invalid ENERGIES values in exposure cube (non-finite or non-positive).")
    return E


def resample_exposure_logE_interp(expo_raw, E_expo_mev, E_target_mev):
    """Interpolate exposure planes in log(E), clamped to endpoint energies."""
    expo_raw = np.asarray(expo_raw, dtype=np.float64)
    E_expo_mev = np.asarray(E_expo_mev, dtype=np.float64)
    E_target_mev = np.asarray(E_target_mev, dtype=np.float64)

    if expo_raw.ndim != 3:
        raise ValueError("expo_raw must be a 3D array (nE, ny, nx)")
    if expo_raw.shape[0] != E_expo_mev.size:
        raise ValueError("expo_raw first axis must match E_expo_mev size")
    if np.any(E_expo_mev <= 0) or np.any(E_target_mev <= 0):
        raise ValueError("E_expo_mev and E_target_mev must be positive")

    logE = np.log(E_expo_mev)
    logEt = np.log(E_target_mev)

    # Ensure increasing energy
    order = np.argsort(logE)
    logE = logE[order]
    expo_raw = expo_raw[order]

    # Clamp targets (no extrapolation)
    logEt = np.clip(logEt, logE[0], logE[-1])

    # Bracket indices
    idx_hi = np.searchsorted(logE, logEt, side="left")
    idx_hi = np.clip(idx_hi, 1, len(logE) - 1)
    idx_lo = idx_hi - 1

    x0 = logE[idx_lo]
    x1 = logE[idx_hi]
    w = (logEt - x0) / (x1 - x0 + 1e-300)

    return (1.0 - w)[:, None, None] * expo_raw[idx_lo] + w[:, None, None] * expo_raw[idx_hi]


def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    """
    Backward-compatible wrapper.
    - If expo already has the right number of energy planes, return as-is.
    - Otherwise require E_expo_mev and use logE interpolation.
    """
    expo_raw = np.asarray(expo_raw)
    if expo_raw.shape[0] == len(E_tgt_mev):
        return expo_raw
    if E_expo_mev is None:
        raise RuntimeError("Exposure planes != target planes and exposure cube has no ENERGIES table.")
    return resample_exposure_logE_interp(expo_raw, E_expo_mev, E_tgt_mev)


def pixel_solid_angle_map(wcs, ny, nx, binsz_deg):
    dl = np.deg2rad(binsz_deg)
    db = np.deg2rad(binsz_deg)
    y = np.arange(ny)
    x_mid = np.full(ny, (nx - 1) / 2.0)
    _, b_deg = wcs.pixel_to_world_values(x_mid, y)
    omega_row = dl * db * np.cos(np.deg2rad(b_deg))
    return omega_row[:, None] * np.ones((1, nx), float)


def lonlat_grids(wcs, ny, nx, *, wrap_lon_deg=True):
    yy, xx = np.mgrid[:ny, :nx]
    lon, lat = wcs.pixel_to_world_values(xx, yy)
    if wrap_lon_deg:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def wcs_from_header(hdr):
    return WCS(hdr).celestial

def resample_exposure(expo_raw, E_expo, E_cnt):
    """
    Interpolate exposure(E) onto counts bin centers using log(E).
    expo_raw: (Ne, ny, nx)
    E_expo, E_cnt in GeV
    """
    if expo_raw.shape[0] == len(E_cnt):
        return expo_raw

    order = np.argsort(E_expo)
    expo_raw = expo_raw[order]
    E_expo = E_expo[order]

    logE_src = np.log(E_expo)
    logE_tgt = np.log(E_cnt)
    print(logE_src)
    print(logE_tgt)

    ne, ny, nx = expo_raw.shape
    flat = expo_raw.reshape(ne, ny * nx)

    out = np.empty((len(E_cnt), ny * nx))
    for i, le in enumerate(logE_tgt):
        j = np.searchsorted(logE_src, le)
        j = np.clip(j, 1, ne - 1)
        w = (le - logE_src[j-1]) / (logE_src[j] - logE_src[j-1])
        out[i] = (1 - w) * flat[j-1] + w * flat[j]

    return out.reshape(len(E_cnt), ny, nx)

def smooth_nan_2d(x2d, sigma_pix):
    m = np.isfinite(x2d)
    if not np.any(m):
        return np.full_like(x2d, np.nan)
    w = m.astype(float)
    xs = gaussian_filter(np.where(m, x2d, 0.0), sigma_pix)
    ws = gaussian_filter(w, sigma_pix)
    y = np.full_like(x2d, np.nan, dtype=float)
    ok = ws > 0
    y[ok] = xs[ok] / ws[ok]
    return y

def load_mask_any_shape(mask_path, counts_shape):
    m = fits.getdata(mask_path).astype(bool)
    nE, ny, nx = counts_shape
    if m.shape == (nE, ny, nx):
        return m
    if m.shape == (ny, nx):
        return np.broadcast_to(m[None, :, :], (nE, ny, nx)).copy()
    raise RuntimeError(f"Mask shape {m.shape} not compatible with counts shape {(nE, ny, nx)}")
