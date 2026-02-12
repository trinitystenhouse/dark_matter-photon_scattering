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


def resample_exposure_logE(expo_raw, E_expo_mev, E_tgt_mev):
    if expo_raw.shape[0] == len(E_tgt_mev):
        return expo_raw
    if E_expo_mev is None:
        raise RuntimeError("Exposure planes != counts planes and EXPO has no ENERGIES/EBOUNDS table.")

    order = np.argsort(E_expo_mev)
    E_expo_mev = np.asarray(E_expo_mev, dtype=float)[order]
    expo_raw = np.asarray(expo_raw, dtype=float)[order]

    logEs = np.log(E_expo_mev)
    logEt = np.log(np.asarray(E_tgt_mev, dtype=float))

    ne, ny, nx = expo_raw.shape
    flat = expo_raw.reshape(ne, ny * nx)

    idx = np.searchsorted(logEs, logEt)
    idx = np.clip(idx, 1, ne - 1)
    i0 = idx - 1
    i1 = idx
    w = (logEt - logEs[i0]) / (logEs[i1] - logEs[i0])

    out = np.empty((len(E_tgt_mev), ny * nx), float)
    for j in range(len(E_tgt_mev)):
        out[j] = (1 - w[j]) * flat[i0[j]] + w[j] * flat[i1[j]]
    return out.reshape(len(E_tgt_mev), ny, nx)


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
