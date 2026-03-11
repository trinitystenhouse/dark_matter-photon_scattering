"""
Fermi bubble geometry and template construction utilities.

This module provides functions for defining Fermi bubble spatial masks and
constructing emission templates based on iterative thresholding of residual maps.

Key Classes
-----------
BubblesIterationResult : Container for bubble mask iteration results

Key Functions
-------------
build_flat_counts_template : Create flat emission template from mask
iterate_bubbles_masks : Iteratively refine bubble masks from residuals
cleanup_binary_mask : Morphological cleanup of binary masks

The Fermi bubbles are modeled as regions with uniform emissivity, defined by
spatial masks derived from residual gamma-ray emission after subtracting
known backgrounds.

Notes
-----
Bubble vertices can be provided as text files with (l, b) coordinates in degrees.
The iterative masking approach follows the methodology of Totani (2025).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, gaussian_filter, label


@dataclass(frozen=True)
class BubblesIterationResult:
    pos_mask: np.ndarray  # (ny,nx) bool
    neg_mask: np.ndarray  # (ny,nx) bool
    smoothed: np.ndarray  # (ny,nx) float
    pos_thresh_used: float
    neg_thresh_used: float
    frac_change_pos: float
    frac_change_neg: float


def _disk_structure(radius_pix: float) -> np.ndarray:
    r = int(max(1, round(float(radius_pix))))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= (r * r)


def cleanup_binary_mask(mask2d: np.ndarray, *, r_open_pix: float, r_close_pix: float) -> np.ndarray:
    m = np.asarray(mask2d, dtype=bool)
    if not np.any(m):
        return m
    st_open = _disk_structure(r_open_pix)
    st_close = _disk_structure(r_close_pix)
    m = binary_opening(m, structure=st_open)
    m = binary_closing(m, structure=st_close)
    m = binary_fill_holes(m)
    return m.astype(bool)


def keep_largest_cc(mask2d: np.ndarray) -> np.ndarray:
    m = np.asarray(mask2d, dtype=bool)
    if not np.any(m):
        return m
    lab, nlab = label(m)
    if nlab <= 1:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return (lab == keep)


def keep_largest_cc_by_hemisphere(mask2d: np.ndarray, *, lat_deg_2d: np.ndarray) -> np.ndarray:
    m = np.asarray(mask2d, dtype=bool)
    lat = np.asarray(lat_deg_2d, dtype=float)
    out = np.zeros_like(m, dtype=bool)

    north = m & np.isfinite(lat) & (lat > 0)
    south = m & np.isfinite(lat) & (lat < 0)

    out |= keep_largest_cc(north)
    out |= keep_largest_cc(south)
    return out


def resolve_overlap(
    *,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    smoothed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos_mask, dtype=bool).copy()
    neg = np.asarray(neg_mask, dtype=bool).copy()
    ov = pos & neg
    if not np.any(ov):
        return pos, neg

    s = np.asarray(smoothed, dtype=float)
    assign_pos = ov & (s >= 0)
    assign_neg = ov & (s < 0)

    pos[ov] = False
    neg[ov] = False
    pos[assign_pos] = True
    neg[assign_neg] = True
    return pos, neg


def build_flat_counts_template(
    *,
    mask2d: np.ndarray,
    roi2d: np.ndarray | None = None,
    expo: np.ndarray,
    omega: np.ndarray,
    dE_mev: np.ndarray,
    Ectr_mev: np.ndarray,
    iso_target_E2: float = 1e-4,
) -> np.ndarray:
    m = np.asarray(mask2d, dtype=bool)
    roi = None if roi2d is None else np.asarray(roi2d, dtype=bool)
    expo = np.asarray(expo, dtype=float)
    omega = np.asarray(omega, dtype=float)
    dE = np.asarray(dE_mev, dtype=float).reshape(-1)
    Ectr = np.asarray(Ectr_mev, dtype=float).reshape(-1)

    nE, ny, nx = expo.shape
    if expo.ndim != 3:
        raise ValueError(f"expo must be 3D (nE,ny,nx); got shape {expo.shape}")
    if omega.shape != (ny, nx):
        raise ValueError(f"omega shape {omega.shape} != {(ny, nx)}")
    if dE.shape[0] != nE:
        raise ValueError(f"dE shape {dE.shape} != ({nE},)")
    if Ectr.shape[0] != nE:
        raise ValueError(f"Ectr_mev shape {Ectr.shape} != ({nE},)")

    # Build a spatial-only template and normalize it over the ROI for fitter stability.
    # Convention: mean(template) = 1 within ROI (including zeros).
    T = m.astype(float)
    if roi is not None:
        vals = T[roi]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise RuntimeError("Bubbles template has no finite pixels in ROI")
        mroi = float(np.mean(vals))
        if (not np.isfinite(mroi)) or (mroi <= 0.0):
            raise RuntimeError("Bubbles ROI mean is invalid; cannot normalize")
        T = T / mroi

    Iref = float(iso_target_E2) / (Ectr**2)
    mu = Iref[:, None, None] * expo * (omega[None, :, :] * dE[:, None, None])
    mu *= T[None, :, :]
    mu[:, ~m] = 0.0
    if mu.shape != (nE, ny, nx):
        raise RuntimeError(f"mu shape {mu.shape} != {(nE, ny, nx)}")
    ok = np.isfinite(expo) & (expo > 0)
    if not np.all(np.isfinite(mu[ok])):
        raise RuntimeError("bubbles mu has non-finite values where expo>0")
    return mu


def nnls_fit_per_energy(
    *,
    counts: np.ndarray,
    templates: Sequence[np.ndarray],
    mask3d: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    try:
        from scipy.optimize import nnls

        have_nnls = True
    except Exception:
        have_nnls = False

    y = np.asarray(counts, dtype=float)
    mask3d = np.asarray(mask3d, dtype=bool)
    nE = y.shape[0]
    nComp = len(templates)
    a = np.zeros((nE, nComp), dtype=float)

    for k in range(nE):
        m = mask3d[k] & np.isfinite(y[k])
        if not np.any(m):
            continue

        yk = y[k][m].reshape(-1)
        Xk = np.vstack([np.asarray(T[k], float)[m].reshape(-1) for T in templates]).T
        good = np.isfinite(yk) & np.all(np.isfinite(Xk), axis=1)
        yk = yk[good]
        Xk = Xk[good]
        if yk.size == 0:
            continue

        if ridge and ridge > 0:
            Xk = np.vstack([Xk, ridge * np.eye(nComp)])
            yk = np.concatenate([yk, np.zeros(nComp, dtype=float)])

        if have_nnls:
            ak, _res = nnls(Xk, yk)
        else:
            ak, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
            ak = np.clip(ak, 0.0, None)

        a[k] = ak

    return a


def build_model_counts(
    *,
    coeffs: np.ndarray,
    templates: Sequence[np.ndarray],
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=float)
    nE, nComp = coeffs.shape
    first = np.asarray(templates[0], dtype=float)
    _, ny, nx = first.shape
    model = np.zeros((nE, ny, nx), dtype=float)

    for j in range(nComp):
        model += coeffs[:, j][:, None, None] * np.asarray(templates[j], dtype=float)

    return model


def residual_flux_from_counts(
    *,
    resid_counts: np.ndarray,
    expo: np.ndarray,
    omega: np.ndarray,
    dE_mev: np.ndarray,
    tiny: float = 1e-30,
) -> np.ndarray:
    resid_counts = np.asarray(resid_counts, dtype=float)
    expo = np.asarray(expo, dtype=float)
    omega = np.asarray(omega, dtype=float)
    dE = np.asarray(dE_mev, dtype=float).reshape(-1)

    denom = expo * (omega[None, :, :] * dE[:, None, None])
    out = np.full_like(resid_counts, np.nan, dtype=float)
    good = np.isfinite(denom) & (denom > tiny) & np.isfinite(resid_counts)
    out[good] = resid_counts[good] / denom[good]
    return out


def combine_residual_flux(
    *,
    resid_flux: np.ndarray,
    weights: np.ndarray,
    k_bins: Sequence[int],
) -> np.ndarray:
    resid_flux = np.asarray(resid_flux, dtype=float)
    weights = np.asarray(weights, dtype=float)

    ny, nx = resid_flux.shape[1:]
    num = np.zeros((ny, nx), dtype=float)
    den = np.zeros((ny, nx), dtype=float)

    for k in k_bins:
        wk = weights[k]
        fk = resid_flux[k]
        good = np.isfinite(wk) & (wk > 0) & np.isfinite(fk)
        num[good] += fk[good] * wk[good]
        den[good] += wk[good]

    out = np.full((ny, nx), np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _adaptive_threshold(
    *,
    smoothed: np.ndarray,
    morph_region: np.ndarray,
    pos_thresh: float,
    neg_thresh: float,
    pos_floor: float,
    neg_ceil: float,
    max_tries: int = 10,
) -> Tuple[float, float]:
    s = np.asarray(smoothed, dtype=float)
    region = np.asarray(morph_region, dtype=bool) & np.isfinite(s)

    pt = float(pos_thresh)
    nt = float(neg_thresh)

    for _ in range(int(max_tries)):
        pos_ok = np.any(region & (s > pt))
        neg_ok = np.any(region & (s < nt))
        if pos_ok and neg_ok:
            break
        if not pos_ok:
            pt = max(pos_floor, 0.8 * pt)
        if not neg_ok:
            nt = min(neg_ceil, 0.8 * nt) if nt < 0 else nt

    return float(pt), float(nt)


def iterate_bubbles_masks(
    *,
    counts: np.ndarray,
    expo: np.ndarray,
    omega: np.ndarray,
    dE_mev: np.ndarray,
    Ectr_mev: np.ndarray,
    iso_target_E2: float = 1e-4,
    templates_counts: Dict[str, np.ndarray],
    base_component_order: Sequence[str],
    roi2d: np.ndarray,
    srcmask2d: Optional[np.ndarray],
    boundary2d: Optional[np.ndarray],
    lat_deg_2d: np.ndarray,
    disk_cut_deg: float,
    k_bins_for_bubbles: Sequence[int],
    n_iter: int,
    smooth_sigma_deg: float,
    binsz_deg: float,
    pos_thresh: float,
    neg_thresh: float,
    morph_open_deg: float,
    morph_close_deg: float,
    alpha: float = 0.0,
    stop_frac: float = 0.01,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[BubblesIterationResult]]:
    counts = np.asarray(counts, dtype=float)
    expo = np.asarray(expo, dtype=float)
    omega = np.asarray(omega, dtype=float)
    dE = np.asarray(dE_mev, dtype=float).reshape(-1)
    Ectr = np.asarray(Ectr_mev, dtype=float).reshape(-1)

    nE, ny, nx = counts.shape
    if expo.shape != counts.shape:
        raise ValueError("expo shape mismatch")
    if omega.shape != (ny, nx):
        raise ValueError("omega shape mismatch")
    if Ectr.shape[0] != nE:
        raise ValueError(f"Ectr_mev shape {Ectr.shape} != ({nE},)")

    roi2d = np.asarray(roi2d, dtype=bool)
    lat = np.asarray(lat_deg_2d, dtype=float)

    if srcmask2d is None:
        src_keep = np.ones((ny, nx), dtype=bool)
    else:
        src_keep = np.asarray(srcmask2d, dtype=bool)

    if boundary2d is None:
        boundary_keep = np.ones((ny, nx), dtype=bool)
    else:
        boundary_keep = np.asarray(boundary2d, dtype=bool)

    fit_mask3d = (
        roi2d[None, :, :]
        & src_keep[None, :, :]
        & np.isfinite(expo)
        & (expo > 0)
        & np.isfinite(counts)
    )

    morph_region = (
        roi2d
        & src_keep
        & boundary_keep
        & np.isfinite(lat)
        & (np.abs(lat) >= float(disk_cut_deg))
    )

    pos_mask = np.zeros((ny, nx), dtype=bool)
    neg_mask = np.zeros((ny, nx), dtype=bool)

    sigma_pix = float(smooth_sigma_deg) / float(binsz_deg)
    r_open_pix = float(morph_open_deg) / float(binsz_deg)
    r_close_pix = float(morph_close_deg) / float(binsz_deg)

    history: List[BubblesIterationResult] = []

    weights = expo * (omega[None, :, :] * dE[:, None, None])

    for it in range(int(n_iter)):
        templates_list: List[np.ndarray] = []

        for name in base_component_order:
            if name not in templates_counts:
                raise KeyError(f"Missing template '{name}'")
            templates_list.append(np.asarray(templates_counts[name], float))

        if it > 0:
            mu_pos = build_flat_counts_template(
                mask2d=pos_mask,
                expo=expo,
                omega=omega,
                dE_mev=dE,
                Ectr_mev=Ectr,
                iso_target_E2=float(iso_target_E2),
            )
            mu_neg = build_flat_counts_template(
                mask2d=neg_mask,
                expo=expo,
                omega=omega,
                dE_mev=dE,
                Ectr_mev=Ectr,
                iso_target_E2=float(iso_target_E2),
            )
            templates_list.extend([mu_pos, mu_neg])

        coeffs = nnls_fit_per_energy(counts=counts, templates=templates_list, mask3d=fit_mask3d, ridge=ridge)
        model = build_model_counts(coeffs=coeffs, templates=templates_list)
        resid_counts = counts - model
        resid_flux = residual_flux_from_counts(resid_counts=resid_counts, expo=expo, omega=omega, dE_mev=dE)

        bubbles_image = combine_residual_flux(resid_flux=resid_flux, weights=weights, k_bins=k_bins_for_bubbles)

        signed_mask = pos_mask.astype(float) - neg_mask.astype(float)
        bubbles_image2 = bubbles_image + float(alpha) * signed_mask

        smoothed = gaussian_filter(bubbles_image2, sigma=sigma_pix, mode="constant", cval=0.0)

        pt_used, nt_used = _adaptive_threshold(
            smoothed=smoothed,
            morph_region=morph_region,
            pos_thresh=float(pos_thresh),
            neg_thresh=float(neg_thresh),
            pos_floor=0.0,
            neg_ceil=0.0,
        )

        pos_raw = morph_region & np.isfinite(smoothed) & (smoothed > pt_used)
        neg_raw = morph_region & np.isfinite(smoothed) & (smoothed < nt_used)

        pos_new = cleanup_binary_mask(pos_raw, r_open_pix=r_open_pix, r_close_pix=r_close_pix)
        neg_new = cleanup_binary_mask(neg_raw, r_open_pix=r_open_pix, r_close_pix=r_close_pix)

        pos_new = keep_largest_cc_by_hemisphere(pos_new, lat_deg_2d=lat)
        neg_new = keep_largest_cc_by_hemisphere(neg_new, lat_deg_2d=lat)

        pos_new, neg_new = resolve_overlap(pos_mask=pos_new, neg_mask=neg_new, smoothed=smoothed)

        pos_new &= morph_region
        neg_new &= morph_region

        dp = np.logical_xor(pos_new, pos_mask)
        dn = np.logical_xor(neg_new, neg_mask)
        frac_p = float(dp.sum() / max(1, pos_mask.sum())) if pos_mask.sum() > 0 else float(dp.mean())
        frac_n = float(dn.sum() / max(1, neg_mask.sum())) if neg_mask.sum() > 0 else float(dn.mean())

        history.append(
            BubblesIterationResult(
                pos_mask=pos_new,
                neg_mask=neg_new,
                smoothed=smoothed,
                pos_thresh_used=pt_used,
                neg_thresh_used=nt_used,
                frac_change_pos=frac_p,
                frac_change_neg=frac_n,
            )
        )

        pos_mask, neg_mask = pos_new, neg_new

        if (frac_p < float(stop_frac)) and (frac_n < float(stop_frac)) and it > 0:
            break

    return pos_mask, neg_mask, history
