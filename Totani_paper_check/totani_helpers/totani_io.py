import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
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


def read_mapcube_primary(path):
    """Read a 3D mapcube from primary HDU.

    Returns
    -------
    cube : np.ndarray
        (nE, ny, nx) if we can infer/reshape it; otherwise the raw array.
    wcs : astropy.wcs.WCS
        Celestial WCS from the primary header.
    E : np.ndarray | None
        Energy grid (MeV) if an ENERGIES or EBOUNDS extension exists.
    hdr : fits.Header
        Primary header.
    """
    with fits.open(path) as h:
        data = h[0].data.astype(float)
        hdr = h[0].header
        w = WCS(hdr).celestial

        E = None
        if "ENERGIES" in h:
            tab = h["ENERGIES"].data
            col = h["ENERGIES"].columns.names[0]
            E = np.array(tab[col], float)
        elif "EBOUNDS" in h:
            eb = h["EBOUNDS"].data
            Emin = np.array(eb["E_MIN"], float)
            Emax = np.array(eb["E_MAX"], float)
            E = np.sqrt(Emin * Emax)

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D mapcube in primary HDU; got shape {data.shape}")

    # Try to ensure (nE, ny, nx)
    cube = data
    if E is not None:
        if cube.shape[0] == len(E):
            pass
        elif cube.shape[-1] == len(E):
            cube = np.moveaxis(cube, -1, 0)

    return cube, w, E, hdr


def reproject_plane_to_target(*, src_plane, w_src, w_tgt, ny_tgt, nx_tgt, wrap_lon=True):
    """Reproject one 2D plane (ny_src,nx_src) onto target WCS grid.

    If wrap_lon is True, it attempts +/-360 deg longitude shifts to avoid seam NaNs.
    """
    yy, xx = np.mgrid[0:ny_tgt, 0:nx_tgt]
    lon, lat = w_tgt.pixel_to_world_values(xx, yy)

    xs0, ys0 = w_src.world_to_pixel_values(lon, lat)
    if wrap_lon:
        xs_p, ys_p = w_src.world_to_pixel_values(lon + 360.0, lat)
        xs_m, ys_m = w_src.world_to_pixel_values(lon - 360.0, lat)
    else:
        xs_p = ys_p = xs_m = ys_m = None

    ny, nx = src_plane.shape
    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)

    interp = RegularGridInterpolator(
        (y, x),
        np.asarray(src_plane, dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    def _eval(xs, ys):
        pts = np.vstack([ys.ravel(), xs.ravel()]).T
        return interp(pts).reshape(ny_tgt, nx_tgt)

    out = _eval(xs0, ys0)
    bad = ~np.isfinite(out)

    if wrap_lon and np.any(bad):
        out_p = _eval(xs_p, ys_p)
        fill = bad & np.isfinite(out_p)
        out[fill] = out_p[fill]
        bad = ~np.isfinite(out)

    if wrap_lon and np.any(bad):
        out_m = _eval(xs_m, ys_m)
        fill = bad & np.isfinite(out_m)
        out[fill] = out_m[fill]
        bad = ~np.isfinite(out)

    if np.any(bad):
        out2 = np.array(out, copy=True)
        for j in range(nx_tgt):
            col = out2[:, j]
            if np.all(~np.isfinite(col)):
                jl = (j - 1) % nx_tgt
                jr = (j + 1) % nx_tgt
                col_l = out2[:, jl]
                col_r = out2[:, jr]
                if np.any(np.isfinite(col_l)):
                    out2[:, j] = col_l
                elif np.any(np.isfinite(col_r)):
                    out2[:, j] = col_r
        out = out2

    return out


def reproject_cube_to_target(*, src_cube, w_src, w_tgt, ny_tgt, nx_tgt, wrap_lon=True):
    src_cube = np.asarray(src_cube)
    out = np.empty((src_cube.shape[0], ny_tgt, nx_tgt), float)
    for k in range(src_cube.shape[0]):
        out[k] = reproject_plane_to_target(
            src_plane=src_cube[k],
            w_src=w_src,
            w_tgt=w_tgt,
            ny_tgt=ny_tgt,
            nx_tgt=nx_tgt,
            wrap_lon=wrap_lon,
        )
    return out

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

def write_cube(path, data, hdr_like, bunit=None):
    hdu = fits.PrimaryHDU(data.astype("f4"), header=hdr_like)
    if bunit is not None:
        hdu.header["BUNIT"] = bunit
    hdu.writeto(path, overwrite=True)

def read_counts_bins(counts_ccube_path):
    with fits.open(counts_ccube_path) as h:
        hdr = h[0].header
        eb = h["EBOUNDS"].data
        emin = np.array(eb["E_MIN"], float)  # keV (for this dataset)
        emax = np.array(eb["E_MAX"], float)  # keV (for this dataset)
    ectr = np.sqrt(emin * emax)  # keV or MeV depending on file
    return hdr, emin, emax, ectr