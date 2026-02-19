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


import numpy as np
from scipy.optimize import minimize, nnls


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
    column_scale="l2",     # NEW: "l2", "l1", "max", or None
    drop_tol=0.0,          # NEW: drop templates with column scale <= drop_tol in that cell/bin
    ridge=0.0,             # NEW: small >=0 regularisation on scaled params (e.g. 1e-6)
):
    """Cell-wise, per-energy-bin Poisson MLE in counts-space, robust to template scaling.

    counts: (nE, ny, nx) observed counts
    templates: dict[str,(nE,ny,nx)] or list of arrays (expected counts templates; can be unnormalised)
    mask3d: bool (nE, ny, nx) True = use pixel in fit

    Returns
    -------
    dict with:
      - cells: list[bool2d]
      - coeff_cells: (nCells, nE, nComp) coefficients in ORIGINAL template units
      - labels: list[str]
      - info: dict(success, nll, message, dropped, scales)
    """
    counts = np.asarray(counts, dtype=float)
    mask3d = np.asarray(mask3d, dtype=bool)
    if counts.shape != mask3d.shape:
        raise ValueError("mask3d must match counts shape")

    nE = counts.shape[0]
    spatial_shape = counts.shape[1:]
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if lon.shape != spatial_shape or lat.shape != spatial_shape:
        raise ValueError("lon/lat must match counts spatial shape")

    # --- templates -> list + labels ---
    if isinstance(templates, dict):
        labels = list(templates.keys()) if component_order is None else list(component_order)
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

    # User-provided helper
    cells = build_lonlat_cells(lon=lon, lat=lat, roi_lon=roi_lon, roi_lat=roi_lat, cell_deg=cell_deg)
    nCells = len(cells)
    nComp = len(tpl_list)

    coeff_cells = np.zeros((nCells, nE, nComp), dtype=float)
    success = np.zeros((nCells, nE), dtype=bool)
    nll_vals = np.full((nCells, nE), np.nan, dtype=float)
    messages = np.empty((nCells, nE), dtype=object)

    # diagnostics
    dropped = np.zeros((nCells, nE, nComp), dtype=bool)
    scales_store = np.full((nCells, nE, nComp), np.nan, dtype=float)

    bounds = [(0.0, None)] * nComp if nonneg else [(None, None)] * nComp

    def _col_scale(X):
        if column_scale is None:
            s = np.ones(X.shape[1], dtype=float)
        elif column_scale == "l2":
            s = np.sqrt(np.sum(X * X, axis=0))
        elif column_scale == "l1":
            s = np.sum(np.abs(X), axis=0)
        elif column_scale == "max":
            s = np.max(np.abs(X), axis=0)
        else:
            raise ValueError("column_scale must be one of: None, 'l2', 'l1', 'max'")
        return np.maximum(s, 0.0)

    def _initial_guess(y, X_scaled, active_idx):
        """Return a0 for the active (non-dropped) parameters in SCALED space."""
        nA = len(active_idx)
        if nA == 0:
            return np.zeros(0, dtype=float)

        if init == "ones":
            # roughly match total counts
            return np.full(nA, max(np.sum(y) / max(nA, 1), 1.0), dtype=float)

        if init == "lsq":
            a, *_ = np.linalg.lstsq(X_scaled, y, rcond=None)
            return np.clip(a, 0.0, None) if nonneg else a

        # nnls is good in scaled space
        try:
            a0, _ = nnls(X_scaled, y)
            if (not np.isfinite(a0).all()) or np.all(a0 == 0):
                a0 = np.full(nA, 1e-3, dtype=float)
            return a0
        except Exception:
            return np.full(nA, 1e-3, dtype=float)

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

            # Column scaling (per cell+bin)
            s = _col_scale(X)
            scales_store[ci, k, :] = s

            active = s > drop_tol
            if not np.any(active):
                messages[ci, k] = "all templates dropped by scale"
                continue

            dropped[ci, k, :] = ~active

            X_a = X[:, active]
            s_a = s[active]
            X_scaled = X_a / s_a[None, :]

            # Work in scaled parameter space: a = a_scaled / s
            # mu = X_a @ a = (X_a/s) @ a_scaled = X_scaled @ a_scaled
            def nll_and_grad_scaled(a_scaled):
                mu = X_scaled @ a_scaled
                mu = np.clip(mu, tiny, None)
                nll = np.sum(mu - y * np.log(mu))

                # optional ridge in scaled params (helps collinearity)
                if ridge > 0:
                    nll = nll + 0.5 * ridge * float(np.dot(a_scaled, a_scaled))

                r = 1.0 - (y / mu)
                grad = X_scaled.T @ r
                if ridge > 0:
                    grad = grad + ridge * a_scaled
                return nll, grad

            def fun(a_scaled):
                v, _ = nll_and_grad_scaled(a_scaled)
                return v

            def jac(a_scaled):
                _, g = nll_and_grad_scaled(a_scaled)
                return g

            a0_scaled = _initial_guess(y, X_scaled, np.where(active)[0])

            # bounds in scaled space map directly if nonneg
            if nonneg:
                bounds_a = [(0.0, None)] * X_scaled.shape[1]
            else:
                bounds_a = [(None, None)] * X_scaled.shape[1]

            opt = minimize(
                fun,
                x0=a0_scaled,
                jac=jac,
                bounds=bounds_a,
                method="L-BFGS-B",
                options={"maxiter": int(maxiter)},
            )

            # Unscale back to original parameterisation
            a_scaled = opt.x
            a_active = a_scaled / s_a

            a_full = np.zeros(nComp, dtype=float)
            a_full[active] = a_active
            # dropped templates remain 0

            coeff_cells[ci, k, :] = a_full
            success[ci, k] = bool(opt.success)
            nll_vals[ci, k] = float(opt.fun)
            messages[ci, k] = str(opt.message)

    info = {
        "success": success,
        "nll": nll_vals,
        "message": messages,
        "dropped": dropped,
        "scales": scales_store,
        "column_scale": column_scale,
        "drop_tol": drop_tol,
        "ridge": ridge,
    }
    return {"cells": cells, "coeff_cells": coeff_cells, "labels": labels, "info": info}

def per_bin_total_counts_from_cellwise_coeffs(*, cells, coeff_cells, templates, mask3d):
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
                T = np.asarray(templates[j])[k]
                s = float(np.nansum(T[cm]))
                if np.isfinite(s):
                    out[k, j] += a[j] * s
    return out
