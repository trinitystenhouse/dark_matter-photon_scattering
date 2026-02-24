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
    init="nnls",               # "nnls", "lsq", "ones", "totani"
    column_scale="l2",         # "l2", "l1", "max", or None
    drop_tol=0.0,
    ridge=0.0,
    # NEW: free-sign components (no lower bound)
    free_sign_labels=("fb_neg",),
    free_sign_substrings=("nfw", "halo"),
):
    """
    Cell-wise, per-energy-bin Poisson MLE in counts-space, robust to scaling.

    Totani-like sign constraints:
      - By default, all components are constrained >=0 when nonneg=True
      - Components whose label is in free_sign_labels OR contains any substring in free_sign_substrings
        are allowed free sign (lower bound None).
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

    # cells
    cells = build_lonlat_cells(lon=lon, lat=lat, roi_lon=roi_lon, roi_lat=roi_lat, cell_deg=cell_deg)
    nCells = len(cells)
    nComp = len(tpl_list)

    coeff_cells = np.zeros((nCells, nE, nComp), dtype=float)
    success = np.zeros((nCells, nE), dtype=bool)
    nll_vals = np.full((nCells, nE), np.nan, dtype=float)
    messages = np.empty((nCells, nE), dtype=object)

    dropped = np.zeros((nCells, nE, nComp), dtype=bool)
    scales_store = np.full((nCells, nE, nComp), np.nan, dtype=float)

    # --- determine which components are free-sign ---
    free_sign_labels = set(free_sign_labels or ())
    free_sign_substrings = tuple(s.lower() for s in (free_sign_substrings or ()))
    free_sign_idx = set()
    for j, lab in enumerate(labels):
        lab_l = str(lab).lower()
        if lab in free_sign_labels or any(ss in lab_l for ss in free_sign_substrings):
            free_sign_idx.add(j)

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

    def _initial_guess_scaled(y, X_scaled, active_idx, s_a, labels_active):
        """
        Return a0_scaled for the active parameters.
        Note: model uses a_scaled, with a = a_scaled / s_a.
        """
        nA = len(active_idx)
        if nA == 0:
            return np.zeros(0, dtype=float)

        if init == "totani":
            # Totani-style start: known comps ~1, unknown/halo start 0.
            # We treat: ps/gas/ics/iso/loop as "known".
            a0 = np.zeros(nA, dtype=float)
            for ii, lab in enumerate(labels_active):
                lab_l = lab.lower()
                if ("ps" == lab_l) or ("point" in lab_l and "source" in lab_l):
                    a0[ii] = 1.0
                elif ("gas" in lab_l) or ("pi0" in lab_l) or ("brem" in lab_l) or ("ics" in lab_l) or ("iem" in lab_l) or ("galprop" in lab_l):
                    a0[ii] = 1.0
                elif lab_l in ("iso", "isotropic"):
                    a0[ii] = 1.0
                elif ("loop" in lab_l):
                    a0[ii] = 1.0
                else:
                    a0[ii] = 0.0  # halo/bubbles/etc start 0
            return a0 * s_a  # convert to scaled space

        if init == "ones":
            a0 = np.full(nA, max(np.sum(y) / max(nA, 1), 1.0), dtype=float)
            return a0

        if init == "lsq":
            a, *_ = np.linalg.lstsq(X_scaled, y, rcond=None)
            # if nonneg, clip only those that are constrained
            if nonneg:
                a = a.copy()
                for ii, j_full in enumerate(active_idx):
                    if j_full not in free_sign_idx:
                        a[ii] = max(a[ii], 0.0)
            return a

        # init == "nnls" default:
        # Only use nnls if ALL active params are constrained nonneg.
        if nonneg and all(j_full not in free_sign_idx for j_full in active_idx):
            try:
                a0, _ = nnls(X_scaled, y)
                if (not np.isfinite(a0).all()) or np.all(a0 == 0):
                    a0 = np.full(nA, 1e-3, dtype=float)
                return a0
            except Exception:
                return np.full(nA, 1e-3, dtype=float)

        # fallback
        return np.zeros(nA, dtype=float)

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

            s = _col_scale(X)
            scales_store[ci, k, :] = s

            active = s > drop_tol
            if not np.any(active):
                messages[ci, k] = "all templates dropped by scale"
                continue
            dropped[ci, k, :] = ~active

            active_idx = np.where(active)[0]
            X_a = X[:, active]
            s_a = s[active]
            X_scaled = X_a / s_a[None, :]

            labels_active = [labels[j] for j in active_idx]

            def nll_and_grad_scaled(a_scaled):
                mu = X_scaled @ a_scaled
                mu = np.clip(mu, tiny, None)
                nll = np.sum(mu - y * np.log(mu))

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

            a0_scaled = _initial_guess_scaled(y, X_scaled, active_idx, s_a, labels_active)

            # --- bounds per active parameter (Totani-like) ---
            bounds_a = []
            for j_full in active_idx:
                if nonneg and (j_full not in free_sign_idx):
                    bounds_a.append((0.0, None))     # known components
                else:
                    bounds_a.append((None, None))    # halo (nfw/halo) and fb_neg

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
        "free_sign_labels": list(free_sign_labels),
        "free_sign_substrings": list(free_sign_substrings),
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
