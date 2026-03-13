import numpy as np
import os

def compute_template_normalisations_cells(
    *,
    Cobs_cells,
    mu,
    labels,
    denom_cells=None,
    Ectr_mev: float | None = None,
    eps: float = 1e-30,
):
    Cobs_cells = np.asarray(Cobs_cells, float)
    mu = np.asarray(mu, float)

    if mu.ndim != 2:
        raise ValueError("mu must be 2D with shape (ncomp, ncells).")
    if Cobs_cells.ndim != 1:
        Cobs_cells = Cobs_cells.reshape(-1)
    if mu.shape[1] != Cobs_cells.shape[0]:
        raise ValueError("Cobs_cells length must match mu.shape[1].")
    if len(labels) != mu.shape[0]:
        raise ValueError("labels length must match mu.shape[0].")

    if denom_cells is not None:
        denom_cells = np.asarray(denom_cells, float)
        if denom_cells.ndim != 1:
            denom_cells = denom_cells.reshape(-1)
        if denom_cells.shape[0] != Cobs_cells.shape[0]:
            raise ValueError("denom_cells length must match Cobs_cells.")

    data_sum = float(np.sum(Cobs_cells))

    out = {
        "data": {
            "sum": data_sum,
            "median": float(np.median(Cobs_cells)) if Cobs_cells.size else float("nan"),
            "max": float(np.max(Cobs_cells)) if Cobs_cells.size else float("nan"),
        },
        "components": [],
    }

    for j, lab in enumerate(labels):
        m = mu[j]
        s = float(np.sum(m))
        r = (s / data_sum) if data_sum > 0 else float("nan")
        comp = {
            "label": str(lab),
            "mu_sum": s,
            "ratio_to_data": r,
            "mu_median": float(np.median(m)) if m.size else float("nan"),
            "mu_max": float(np.max(m)) if m.size else float("nan"),
        }

        if denom_cells is not None and (str(lab).lower() in ("iso", "isotropic")) and (Ectr_mev is not None):
            good = (denom_cells > 0) & np.isfinite(denom_cells) & np.isfinite(m)
            I_cells = np.full_like(m, np.nan)
            I_cells[good] = m[good] / np.maximum(denom_cells[good], eps)
            I_med = float(np.nanmedian(I_cells))
            E2I = float((Ectr_mev**2) * I_med)
            comp["iso_inferred_dNdE_median"] = I_med
            comp["iso_inferred_E2dNdE"] = E2I

        out["components"].append(comp)

    return out

def report_template_normalisations_cells(
    *,
    Cobs_cells,          # (ncells,) observed counts per cell
    mu,                  # (ncomp, ncells) expected counts per cell for f=1
    labels,              # list[str], len=ncomp
    denom_cells,         # (ncells,) = sum(expo*omega*dE) per cell [cm^2 s sr MeV]
    Ectr_mev: float,
    eps: float = 1e-30,
):
    res = compute_template_normalisations_cells(
        Cobs_cells=Cobs_cells,
        mu=mu,
        labels=labels,
        denom_cells=denom_cells,
        Ectr_mev=Ectr_mev,
        eps=eps,
    )

    d = res["data"]
    print(f"DATA: sum={d['sum']:.6e}, median={d['median']:.6e}, max={d['max']:.6e}\n")

    for comp in res["components"]:
        lab = comp["label"]
        print(
            f"{lab:12s} mu(f=1): sum={comp['mu_sum']:.6e}  ratio_to_data={comp['ratio_to_data']:.6e}  "
            f"median={comp['mu_median']:.6e}  max={comp['mu_max']:.6e}"
        )
        if "iso_inferred_dNdE_median" in comp:
            print(f"  -> inferred iso dN/dE (median over cells) = {comp['iso_inferred_dNdE_median']:.6e}")
            print(f"  -> inferred iso E^2 dN/dE                 = {comp['iso_inferred_E2dNdE']:.6e}  [MeV cm^-2 s^-1 sr^-1]")

    print("\nInterpretation:")
    print("- If mu sums are comparable to data sums, templates are in counts-space; f is dimensionless.")
    print("- For isotropic: if inferred E^2 dN/dE is already ~1e-4, then your iso template already includes Totani's typical amplitude; set f_iso_init≈1.")
    print("- If inferred E^2 dN/dE is ~1e-6, then you'd need f_iso_init≈(1e-4 / 1e-6)=100 to match that typical starting flux.")


def get_template_normalisations_cells_summary(
    *,
    Cobs_cells,          # (ncells,) observed counts per cell
    mu,                  # (ncomp, ncells) expected counts per cell for f=1
    labels,              # list[str], len=ncomp
    denom_cells,         # (ncells,) = sum(expo*omega*dE) per cell [cm^2 s sr MeV]
    Ectr_mev: float,
    eps: float = 1e-30,
):
    res = compute_template_normalisations_cells(
        Cobs_cells=Cobs_cells,
        mu=mu,
        labels=labels,
        denom_cells=denom_cells,
        Ectr_mev=Ectr_mev,
        eps=eps,
    )

    d = res["data"]
    summary = f"DATA: sum={d['sum']:.6e}, median={d['median']:.6e}, max={d['max']:.6e}\n"

    for comp in res["components"]:
        lab = comp["label"]
        summary += (
            f"{lab:12s} mu(f=1): sum={comp['mu_sum']:.6e}  ratio_to_data={comp['ratio_to_data']:.6e}  "
            f"median={comp['mu_median']:.6e}  max={comp['mu_max']:.6e}\n"
        )
        if "iso_inferred_dNdE_median" in comp:
            summary += (
                f"  -> inferred iso dN/dE (median over cells) = {comp['iso_inferred_dNdE_median']:.6e}\n"
                f"  -> inferred iso E^2 dN/dE                 = {comp['iso_inferred_E2dNdE']:.6e}  [MeV cm^-2 s^-1 sr^-1]\n"
            )

    summary += "\nInterpretation:\n"
    summary += "- If mu sums are comparable to data sums, templates are in counts-space; f is dimensionless.\n"
    summary += "- For isotropic: if inferred E^2 dN/dE is already ~1e-4, then your iso template already includes Totani's typical amplitude; set f_iso_init≈1.\n"
    summary += "- If inferred E^2 dN/dE is ~1e-6, then you'd need f_iso_init≈(1e-4 / 1e-6)=100 to match that typical starting flux.\n"

    return summary


def build_denom_cells(expo, omega, dE_mev, cell_index, ncells):
    valid = cell_index >= 0
    w = (expo * omega * dE_mev)[valid]          # [cm^2 s sr MeV]
    idx = cell_index[valid]
    return np.bincount(idx, weights=w, minlength=ncells)


def report_template_normalisations_from_mu_list(
    *,
    counts_cube,
    mu_list,
    labels,
    fit_mask3d,
    energy_bin: int,
    Ectr_mev,
    expo_cube=None,
    omega_map=None,
    dE_mev=None,
    eps: float = 1e-30,
):
    k = int(energy_bin)
    counts_cube = np.asarray(counts_cube)
    if counts_cube.ndim != 3:
        raise ValueError("counts_cube must have shape (nE, ny, nx).")
    if k < 0 or k >= counts_cube.shape[0]:
        raise ValueError("energy_bin out of range for counts_cube.")

    mask_k = np.asarray(fit_mask3d, bool)[k]
    if mask_k.shape != counts_cube.shape[1:]:
        raise ValueError("fit_mask3d[k] shape must match counts_cube[k].")

    Cobs = np.asarray(counts_cube[k][mask_k], float).ravel()

    if len(mu_list) != len(labels):
        raise ValueError("mu_list length must match labels length.")

    mu = np.zeros((len(labels), Cobs.size), dtype=float)
    for j in range(len(labels)):
        tmpl = np.asarray(mu_list[j])
        if tmpl.ndim != 3 or tmpl.shape != counts_cube.shape:
            raise ValueError("Each mu_list[j] must have the same shape as counts_cube.")
        mu[j] = np.asarray(tmpl[k][mask_k], float).ravel()

    denom = None
    if (expo_cube is not None) and (omega_map is not None) and (dE_mev is not None):
        expo_cube = np.asarray(expo_cube, float)
        omega_map = np.asarray(omega_map, float)
        if expo_cube.shape != counts_cube.shape:
            raise ValueError("expo_cube must have the same shape as counts_cube.")
        if omega_map.shape != counts_cube.shape[1:]:
            raise ValueError("omega_map shape must match (ny, nx).")
        denom = (expo_cube[k] * omega_map * float(dE_mev))[mask_k].ravel()

    report_template_normalisations_cells(
        Cobs_cells=Cobs,
        mu=mu,
        labels=labels,
        denom_cells=denom if denom is not None else np.ones_like(Cobs),
        Ectr_mev=float(np.asarray(Ectr_mev).reshape(-1)[0]),
        eps=eps,
    )


def _load_npz_array(npz, key: str, *, required: bool = True):
    if key in npz:
        return npz[key]
    if required:
        raise KeyError(f"Key '{key}' not found in npz. Available keys: {list(npz.keys())}")
    return None


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Compute template normalisation summaries for your data")
    ap.add_argument("--npz", required=True, help="Path to an .npz containing Cobs/mu/(optional) denom/labels/Ectr")
    ap.add_argument("--cobs-key", default="Cobs_cells")
    ap.add_argument("--mu-key", default="mu")
    ap.add_argument("--labels-key", default="labels")
    ap.add_argument("--denom-key", default="denom_cells")
    ap.add_argument("--ectr-key", default="Ectr_mev")
    ap.add_argument("--Ectr-mev", type=float, default=None, help="Override Ectr_mev if not in npz")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    Cobs = _load_npz_array(npz, args.cobs_key)
    mu = _load_npz_array(npz, args.mu_key)
    labels = _load_npz_array(npz, args.labels_key)
    denom = _load_npz_array(npz, args.denom_key, required=False)

    if isinstance(labels, np.ndarray):
        labels = [str(x) for x in labels.tolist()]

    Ectr_mev = args.Ectr_mev
    if Ectr_mev is None:
        Ectr = _load_npz_array(npz, args.ectr_key, required=False)
        if Ectr is not None:
            Ectr_mev = float(np.asarray(Ectr).reshape(-1)[0])

    if denom is None or Ectr_mev is None:
        denom_arg = np.asarray(denom, float) if denom is not None else np.ones_like(np.asarray(Cobs, float))
        report_template_normalisations_cells(
            Cobs_cells=Cobs,
            mu=mu,
            labels=labels,
            denom_cells=denom_arg,
            Ectr_mev=float(Ectr_mev) if Ectr_mev is not None else 1.0,
        )
        return

    report_template_normalisations_cells(
        Cobs_cells=Cobs,
        mu=mu,
        labels=labels,
        denom_cells=denom,
        Ectr_mev=float(Ectr_mev),
    )


if __name__ == "__main__":
    main()