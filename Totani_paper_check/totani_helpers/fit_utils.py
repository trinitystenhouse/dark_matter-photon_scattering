import numpy as np


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
