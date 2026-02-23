import numpy as np
from typing import Dict, Tuple, Union
from scipy.optimize import nnls
import os
from astropy.io import fits

def data_coverage_mask3d(*, counts, expo):
    """Return True where data coverage exists: finite counts/expo and expo>0."""
    counts = np.asarray(counts)
    expo = np.asarray(expo)
    return np.isfinite(counts) & np.isfinite(expo) & (expo > 0)


def build_fit_mask3d(
    *,
    roi2d,
    srcmask3d,
    counts,
    expo,
    extra2d=None,
):
    """Build fit mask enforcing ROI, source masks, optional extra 2D mask, and data coverage."""
    roi2d = np.asarray(roi2d, dtype=bool)
    srcmask3d = np.asarray(srcmask3d, dtype=bool)

    fit_mask3d = srcmask3d & roi2d[None, :, :]
    if extra2d is not None:
        extra2d = np.asarray(extra2d, dtype=bool)
        fit_mask3d &= extra2d[None, :, :]

    fit_mask3d &= data_coverage_mask3d(counts=counts, expo=expo)
    return fit_mask3d


def component_counts_from_cellwise_fit(*, templates_counts, res_fit, mask3d=None):
    """Accumulate fitted counts cubes per component from cellwise multipliers.

    templates_counts: dict[str, (nE,ny,nx)] counts templates
    res_fit: output dict from fit_cellwise_poisson_mle_counts
    mask3d: optional boolean (nE,ny,nx) to restrict accumulation region

    Returns:
      comp_counts: dict[name -> (nE,ny,nx)]
      total_counts: (nE,ny,nx)
    """
    labels = list(res_fit["labels"])
    cells = res_fit["cells"]
    coeff_cells = np.asarray(res_fit["coeff_cells"], float)

    first = next(iter(templates_counts.values()))
    nE, ny, nx = first.shape

    if mask3d is None:
        mask3d = np.ones((nE, ny, nx), dtype=bool)
    else:
        mask3d = np.asarray(mask3d, dtype=bool)

    comp_counts = {lab: np.zeros((nE, ny, nx), dtype=float) for lab in labels}
    total = np.zeros((nE, ny, nx), dtype=float)

    for ci, cell2d in enumerate(cells):
        cell2d = np.asarray(cell2d, dtype=bool)
        for k in range(nE):
            m = mask3d[k] & cell2d
            if not np.any(m):
                continue
            a = coeff_cells[ci, k, :]
            if not np.all(np.isfinite(a)):
                continue
            for j, lab in enumerate(labels):
                mu = np.asarray(templates_counts[lab][k], float)
                if a[j] != 0.0:
                    comp_counts[lab][k][m] += float(a[j]) * mu[m]
                    total[k][m] += float(a[j]) * mu[m]

    return comp_counts, total

def normalise_templates_counts(templates_counts, fit_mask_2d):
    """
    templates_counts: dict name -> (nE,ny,nx) counts template
    fit_mask_2d: (ny,nx) boolean mask used for the fit

    Returns:
      templates_norm: dict name -> (nE,ny,nx) counts template, where for each k:
                       sum_k(template[k][fit_mask]) = 1 (if original sum > 0)
      norms: dict name -> (nE,) original sums in the fit mask
    """
    templates_norm = {}
    norms = {}

    for name, T in templates_counts.items():
        T = np.asarray(T, dtype=float)
        if T.ndim != 3:
            raise ValueError(f"{name} must be 3D (nE,ny,nx), got shape {T.shape}")

        nE = T.shape[0]
        Tn = T.copy()
        s = np.zeros(nE, dtype=float)

        for k in range(nE):
            sk = float(np.nansum(Tn[k][fit_mask_2d]))
            s[k] = sk
            if sk > 0:
                Tn[k] /= sk

        templates_norm[name] = Tn
        norms[name] = s

    return templates_norm, norms

def fit_nnls_counts_3d(
    data_counts: np.ndarray,                      # (nE,ny,nx)
    templates_counts: Dict[str, np.ndarray],      # each (nE,ny,nx) OR (ny,nx)
    fit_mask_2d: np.ndarray,                      # (ny,nx) bool
    *,
    mode: str = "per_energy",                     # "per_energy" or "global"
    min_pix: int = 10,
) -> Tuple[Dict[str, Union[np.ndarray, float]], np.ndarray]:
    """
    NNLS fit in counts space for 3D data cubes.

    model(E,x) ≈ Σ_i a_i(E) * T_i(E,x)    (mode="per_energy")
    or
    model(E,x) ≈ Σ_i a_i     * T_i(E,x)   (mode="global")

    Templates can be provided as:
      - (nE,ny,nx): energy-dependent counts templates
      - (ny,nx): broadcast to all energies (useful for purely spatial masks only IF already in counts;
                 in most cases you should convert spatial templates to counts per energy first)
    """
    data_counts = np.asarray(data_counts)
    if data_counts.ndim != 3:
        raise ValueError(f"data_counts must be (nE,ny,nx); got {data_counts.shape}")
    nE, ny, nx = data_counts.shape

    fit_mask_2d = np.asarray(fit_mask_2d, dtype=bool)
    if fit_mask_2d.shape != (ny, nx):
        raise ValueError(f"fit_mask_2d must be (ny,nx)={ny,nx}; got {fit_mask_2d.shape}")

    # Prepare template stack (nComp, nE, ny, nx)
    names = list(templates_counts.keys())
    Ts = []
    for name in names:
        T = np.asarray(templates_counts[name], dtype=float)
        if T.ndim == 2:
            if T.shape != (ny, nx):
                raise ValueError(f"Template '{name}' 2D shape must be (ny,nx); got {T.shape}")
            T = np.broadcast_to(T[None, :, :], (nE, ny, nx)).copy()
        elif T.ndim == 3:
            if T.shape != (nE, ny, nx):
                raise ValueError(f"Template '{name}' 3D shape must be (nE,ny,nx); got {T.shape}")
        else:
            raise ValueError(f"Template '{name}' must be 2D or 3D; got {T.shape}")
        Ts.append(T)
    Tstack = np.stack(Ts, axis=0)  # (nComp,nE,ny,nx)

    # Mask: broadcast over energy, and require finite data
    m3 = fit_mask_2d[None, :, :] & np.isfinite(data_counts)

    if mode == "per_energy":
        coeffs = np.zeros((nE, len(names)), dtype=float)
        model = np.zeros_like(data_counts, dtype=float)

        for k in range(nE):
            mk = m3[k]
            if mk.sum() < min_pix:
                # leave coeffs[k] = 0 and model[k]=0; or raise if you prefer
                continue

            y = data_counts[k][mk].astype(float)  # (Npix,)
            A = np.vstack([Tstack[j, k][mk] for j in range(len(names))]).T  # (Npix, nComp)

            c, _ = nnls(A, y)
            coeffs[k, :] = c

            # build model plane
            for j, name in enumerate(names):
                model[k] += c[j] * Tstack[j, k]

        coeff_dict = {name: coeffs[:, j].copy() for j, name in enumerate(names)}
        return coeff_dict, model

    elif mode == "global":
        mflat = m3.reshape(-1)
        if mflat.sum() < min_pix:
            raise RuntimeError("fit mask too small")

        y = data_counts.reshape(-1)[mflat].astype(float)  # (Npts,)
        A = np.vstack([Tstack[j].reshape(-1)[mflat] for j in range(len(names))]).T  # (Npts,nComp)

        c, _ = nnls(A, y)

        model = np.zeros_like(data_counts, dtype=float)
        for j, name in enumerate(names):
            model += c[j] * Tstack[j]

        coeff_dict = {name: float(c[j]) for j, name in enumerate(names)}
        return coeff_dict, model

    else:
        raise ValueError("mode must be 'per_energy' or 'global'")


def load_mu_templates_from_fits(
    template_dir,
    labels,
    filename_pattern="{label}.fits",   # or "{label}_template.fits"
    hdu=0,
    dtype=np.float32,
    memmap=True,
    require_same_shape=True,
):
    """
    Load TRUE-counts template cubes (mu) from FITS files into mu_list.

    Assumes each template FITS contains a data cube shaped like:
        (nE, ny, nx)   OR   (nE, npix)
    matching your counts cube.

    Parameters
    ----------
    template_dir : str
        Directory containing FITS templates.
    labels : list[str]
        Component labels, used to build filenames.
    filename_pattern : str
        How to build a filename from a label. Example:
            "{label}.fits"
            "mu_{label}.fits"
            "{label}_mapcube.fits"
    hdu : int
        FITS HDU index holding the data.
    dtype : numpy dtype
        Cast output arrays to this dtype (float32 is usually plenty).
    memmap : bool
        Use FITS memmap to avoid loading everything at once.
    require_same_shape : bool
        If True, raises if templates don't all share the same shape.

    Returns
    -------
    mu_list : list[np.ndarray]
        List of template arrays in counts units.
    headers : list[fits.Header]
        Corresponding FITS headers (for WCS / energy metadata if needed).
    """
    mu_list = []
    headers = []
    shapes = []

    for lab in labels:
        path = os.path.join(template_dir, filename_pattern.format(label=lab))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found for '{lab}': {path}")

        with fits.open(path, memmap=memmap) as hdul:
            data = hdul[hdu].data
            hdr = hdul[hdu].header

        if data is None:
            raise ValueError(f"No data in {path} (HDU {hdu})")

        arr = np.asarray(data, dtype=dtype)

        # Basic sanity: must be at least 2D with energy axis first
        if arr.ndim < 2:
            raise ValueError(f"Template '{lab}' has ndim={arr.ndim}, expected (nE, ...spatial...)")

        mu_list.append(arr)
        headers.append(hdr)
        shapes.append(arr.shape)

    if require_same_shape:
        s0 = shapes[0]
        for lab, s in zip(labels, shapes):
            if s != s0:
                raise ValueError(f"Shape mismatch: '{labels[0]}' {s0} vs '{lab}' {s}")

    return mu_list, headers

from scipy.optimize import minimize
from scipy.optimize import nnls
import numpy as np

def fit_global_poisson_mle_counts(*, counts, templates, mask3d, nonneg=True, tiny=1e-30, maxiter=200, init="nnls",
                                  ridge=None):
    # templates: dict name->(nE,ny,nx) or list
    if isinstance(templates, dict):
        labels = list(templates.keys())
        tpl_list = [np.asarray(templates[k], float) for k in labels]
    else:
        tpl_list = [np.asarray(t, float) for t in templates]
        labels = [f"comp{j}" for j in range(len(tpl_list))]

    nE = counts.shape[0]
    nComp = len(tpl_list)
    coeff = np.zeros((nE, nComp), float)
    success = np.zeros(nE, bool)

    bounds = [(0.0, None)] * nComp if nonneg else [(None, None)] * nComp

    # optional ridge dict: {j: lam}
    ridge = ridge or {}

    for k in range(nE):
        m = mask3d[k]
        if not np.any(m):
            continue
        y = counts[k][m].ravel().astype(float)
        X = np.stack([tpl_list[j][k][m].ravel().astype(float) for j in range(nComp)], axis=1)

        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        y, X = y[good], X[good]
        if y.size == 0:
            continue

        # L2 column scaling for numerical stability (matches cellwise fit)
        s = np.sqrt(np.sum(X * X, axis=0))
        s = np.maximum(s, tiny)
        X_scaled = X / s[None, :]

        # init in scaled space
        if init == "nnls" and nonneg:
            a0_scaled, _ = nnls(X_scaled, y)
        else:
            a0_scaled = np.full(nComp, 1e-3, float)

        # Optimize in scaled parameter space: a = a_scaled / s
        def fun(a_scaled):
            mu = np.clip(X_scaled @ a_scaled, tiny, None)
            nll = np.sum(mu - y*np.log(mu))
            # ridge on scaled params
            for j, lam in ridge.items():
                nll += 0.5 * lam * a_scaled[j] * a_scaled[j]
            return nll

        def jac(a_scaled):
            mu = np.clip(X_scaled @ a_scaled, tiny, None)
            r = 1.0 - (y/mu)
            g = X_scaled.T @ r
            for j, lam in ridge.items():
                g[j] += lam * a_scaled[j]
            return g

        opt = minimize(fun, a0_scaled, jac=jac, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": int(maxiter)})
        
        # Unscale back to original template units
        a_scaled = opt.x
        coeff[k] = a_scaled / s
        success[k] = opt.success

    return {"coeff": coeff, "labels": labels, "success": success}

import numpy as np

def predict_counts_maps_global(*, aG, mu_list, mask3d):
    """
    aG: (nE, nComp) global coeffs
    mu_list: list of (nE,ny,nx) counts templates
    mask3d: bool (nE,ny,nx) keep mask defining region+coverage

    Returns:
      pred_comp: (nComp,nE) predicted total counts in region
      pred_model:(nE,) total predicted counts in region
    """
    nE = mask3d.shape[0]
    nComp = len(mu_list)
    pred_comp = np.zeros((nComp, nE), float)
    pred_model = np.zeros(nE, float)

    for k in range(nE):
        m = mask3d[k]
        if not np.any(m):
            continue
        tot = 0.0
        for j in range(nComp):
            mu = np.asarray(mu_list[j][k], float)
            s = float(np.nansum(mu[m]))
            c = float(aG[k, j]) * s
            pred_comp[j, k] = c
            tot += c
        pred_model[k] = tot
    return pred_comp, pred_model

def E2_from_pred_counts_maps(*, pred_counts_map, expo, omega, dE_mev, Ectr_mev, mask2d, tiny=1e-30):
    """
    pred_counts_map: (nE,ny,nx) predicted counts map for ONE component (or model sum)
    mask2d: (ny,nx) region mask (geometric), but we also require finite expo per bin.
    Returns: (E2, ) array length nE in MeV cm^-2 s^-1 sr^-1
    """
    nE = pred_counts_map.shape[0]
    E2 = np.full(nE, np.nan, float)

    for k in range(nE):
        m = mask2d & np.isfinite(expo[k]) & (expo[k] > 0)
        if not np.any(m):
            continue

        Omega_reg = float(np.nansum(omega[m]))
        if not np.isfinite(Omega_reg) or Omega_reg <= 0:
            continue

        denom = expo[k][m] * dE_mev[k]
        good = np.isfinite(denom) & (denom > tiny) & np.isfinite(pred_counts_map[k][m])
        if not np.any(good):
            continue

        N = pred_counts_map[k][m][good]
        denom = denom[good]
        I_mean = float(np.nansum(N / denom) / Omega_reg)   # ph cm^-2 s^-1 sr^-1 MeV^-1
        E2[k] = I_mean * (Ectr_mev[k] ** 2)
    return E2
