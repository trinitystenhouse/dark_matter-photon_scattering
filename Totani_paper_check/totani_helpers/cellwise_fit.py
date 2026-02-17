import numpy as np
from scipy.optimize import minimize, nnls


def build_lonlat_cells(*, lon, lat, roi_lon, roi_lat, cell_deg):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape")

    l_edges = np.arange(-float(roi_lon), float(roi_lon) + 1e-9, float(cell_deg))
    b_edges = np.arange(-float(roi_lat), float(roi_lat) + 1e-9, float(cell_deg))

    cells = []
    for l0 in l_edges[:-1]:
        l1 = l0 + float(cell_deg)
        in_l = (lon >= l0) & (lon < l1)
        for b0 in b_edges[:-1]:
            b1 = b0 + float(cell_deg)
            cell = in_l & (lat >= b0) & (lat < b1)
            if np.any(cell):
                cells.append(cell)
    return cells


def fit_cellwise_poisson_mle_counts(
    *,
    counts,
    templates,
    mask3d,
    lon,
    lat,
    roi_lon,
    roi_lat,
    cell_deg=10.0,
    component_order=None,
    nonneg=True,
    tiny=1e-30,
    maxiter=200,
    init="nnls",
):
    """Cell-wise, per-energy-bin Poisson MLE in counts-space.

    Parameters
    ----------
    counts: array (nE, ny, nx)
    templates: dict[str, array] (each (nE, ny, nx)) or list/tuple of arrays
        Expected counts templates.
    mask3d: bool array (nE, ny, nx) True=use pixel in fit

    Returns
    -------
    result: dict
        - cells: list[bool2d]
        - coeff_cells: (nCells, nE, nComp)
        - labels: list[str]
        - info: dict(success, nll, message)
    """

    counts = np.asarray(counts, dtype=float)
    mask3d = np.asarray(mask3d, dtype=bool)
    if counts.shape != mask3d.shape:
        raise ValueError("mask3d must match counts shape")

    nE = counts.shape[0]
    spatial_shape = counts.shape[1:]
    if np.asarray(lon).shape != spatial_shape or np.asarray(lat).shape != spatial_shape:
        raise ValueError("lon/lat must match counts spatial shape")

    if isinstance(templates, dict):
        if component_order is None:
            labels = list(templates.keys())
        else:
            labels = list(component_order)
        tpl_list = [np.asarray(templates[k], dtype=float) for k in labels]
    elif isinstance(templates, (list, tuple)):
        tpl_list = [np.asarray(t, dtype=float) for t in templates]
        labels = [f"comp{j}" for j in range(len(tpl_list))]
    else:
        raise TypeError("templates must be a dict or a list/tuple")

    if len(tpl_list) == 0:
        raise ValueError("templates must be non-empty")

    for j, T in enumerate(tpl_list):
        if T.shape != counts.shape:
            raise ValueError(f"template '{labels[j]}' shape {T.shape} != counts shape {counts.shape}")

    cells = build_lonlat_cells(lon=lon, lat=lat, roi_lon=roi_lon, roi_lat=roi_lat, cell_deg=cell_deg)
    nCells = len(cells)
    nComp = len(tpl_list)

    coeff_cells = np.zeros((nCells, nE, nComp), dtype=float)
    success = np.zeros((nCells, nE), dtype=bool)
    nll_vals = np.full((nCells, nE), np.nan, dtype=float)
    messages = np.empty((nCells, nE), dtype=object)

    def _initial_guess(y, X):
        if init == "ones":
            return np.full(nComp, max(np.sum(y) / max(nComp, 1), 1.0), dtype=float)
        if init == "lsq":
            a, *_ = np.linalg.lstsq(X, y, rcond=None)
            return np.clip(a, 0.0, None) if nonneg else a
        try:
            a0, _ = nnls(X, y)
            if (not np.isfinite(a0).all()) or np.all(a0 == 0):
                a0 = np.full(nComp, 1e-3, dtype=float)
            return a0
        except Exception:
            return np.full(nComp, 1e-3, dtype=float)

    bounds = [(0.0, None)] * nComp if nonneg else [(None, None)] * nComp

    for ci, cell2d in enumerate(cells):
        for k in range(nE):
            m = mask3d[k] & cell2d
            if not np.any(m):
                messages[ci, k] = "empty mask"
                continue

            y = counts[k][m].astype(float).ravel()
            X = np.stack([tpl_list[j][k][m].astype(float).ravel() for j in range(nComp)], axis=1)

            good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            y = y[good]
            X = X[good]
            if y.size == 0:
                messages[ci, k] = "no finite pixels"
                continue

            def nll_and_grad(a):
                mu = X @ a
                mu = np.clip(mu, tiny, None)
                nll = np.sum(mu - y * np.log(mu))
                r = 1.0 - (y / mu)
                grad = X.T @ r
                return nll, grad

            def fun(a):
                v, _ = nll_and_grad(a)
                return v

            def jac(a):
                _, g = nll_and_grad(a)
                return g

            a0 = _initial_guess(y, X)
            opt = minimize(
                fun,
                x0=a0,
                jac=jac,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": int(maxiter)},
            )

            coeff_cells[ci, k] = opt.x
            success[ci, k] = bool(opt.success)
            nll_vals[ci, k] = float(opt.fun)
            messages[ci, k] = str(opt.message)

    info = {"success": success, "nll": nll_vals, "message": messages}
    return {"cells": cells, "coeff_cells": coeff_cells, "labels": labels, "info": info}


def per_bin_total_counts_from_cellwise_coeffs(*, cells, coeff_cells, templates, mask3d):
    """Convert cellwise multipliers into per-bin total counts per component.

    templates: list of arrays (nE, ny, nx) matching coeff_cells last axis.
    mask3d: boolean keep mask applied to the summation.

    Returns array totals (nE, nComp).
    """
    mask3d = np.asarray(mask3d, dtype=bool)
    nCells, nE, nComp = coeff_cells.shape
    if len(templates) != nComp:
        raise ValueError("templates length must match coeff_cells nComp")

    out = np.zeros((nE, nComp), dtype=float)
    for k in range(nE):
        for ci, cell2d in enumerate(cells):
            cm = mask3d[k] & cell2d
            if not np.any(cm):
                continue
            a = coeff_cells[ci, k, :]
            if not np.all(np.isfinite(a)):
                continue
            for j in range(nComp):
                s = float(np.nansum(np.asarray(templates[j])[k][cm]))
                if np.isfinite(s) and s != 0.0:
                    out[k, j] += float(a[j]) * s
    return out
