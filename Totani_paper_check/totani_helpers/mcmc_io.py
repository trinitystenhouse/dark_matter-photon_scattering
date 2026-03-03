import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MCMCCoeffTable:
    labels: List[str]
    coeffs_by_label: Dict[str, np.ndarray]
    bins_present: np.ndarray


def add_flux_scaling_args(
    parser,
    *,
    default_expo: Optional[str] = None,
    default_coeff_file: Optional[str] = None,
    default_binsz: float = 0.125,
    include_mcmc_component: bool = True,
    include_expo: bool = True,
    include_binsz: bool = True,
):
    parser.add_argument(
        "--scale-flux",
        action="store_true",
        help="Also compute ROI-averaged E^2 dN/dE using MCMC coefficients and exposure",
    )
    if include_expo:
        parser.add_argument(
            "--expo",
            default=default_expo,
            help="Exposure cube FITS (required for --scale-flux)",
        )
    if include_binsz:
        parser.add_argument(
            "--binsz",
            type=float,
            default=float(default_binsz),
            help="Pixel size in degrees (for solid angle; required for --scale-flux)",
        )
    parser.add_argument(
        "--coeff-file",
        default=default_coeff_file,
        help="Coefficient table .txt file (optional alternative to --mcmc-dir)",
    )
    parser.add_argument(
        "--mcmc-dir",
        default=None,
        help="MCMC results directory (required for --scale-flux)",
    )
    parser.add_argument(
        "--mcmc-stat",
        choices=["f_ml", "f_p50", "f_p16", "f_p84"],
        default="f_ml",
        help="Which MCMC summary coefficient to use per bin",
    )
    if include_mcmc_component:
        parser.add_argument(
            "--mcmc-component",
            default=None,
            help="Component key in MCMC outputs (default: match --label)",
        )
    return parser


def load_coeff_table_txt(*, coeff_file: str, nE: Optional[int] = None) -> MCMCCoeffTable:
    if not os.path.exists(str(coeff_file)):
        raise FileNotFoundError(f"Coefficient file not found: {coeff_file}")

    with open(str(coeff_file), "r") as f:
        header = f.readline().strip()
        if header.startswith("#"):
            header = header.lstrip("#").strip()
        cols = header.split()
        if len(cols) < 2:
            raise ValueError(f"Malformed coefficient header: '{header}'")

        rows: List[List[float]] = []
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            try:
                rows.append([float(x) for x in parts])
            except Exception:
                continue

    if not rows:
        raise ValueError(f"No coefficient rows found in: {coeff_file}")

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Coefficient table has unexpected shape {arr.shape} in {coeff_file}")

    x = arr[:, 0]
    labels = [str(c) for c in cols[1:]]
    data = arr[:, 1:]
    if data.shape[1] != len(labels):
        raise ValueError(
            f"Header/data mismatch in {coeff_file}: header has {len(labels)} columns but data has {data.shape[1]}"
        )

    if nE is None:
        nE = int(data.shape[0])

    coeffs_by_name: Dict[str, np.ndarray] = {}
    for j, lab in enumerate(labels):
        v = np.full(int(nE), np.nan, dtype=float)
        ncopy = min(int(nE), int(data.shape[0]))
        v[:ncopy] = np.asarray(data[:ncopy, j], float)
        coeffs_by_name[str(lab)] = v

    bins_present = np.arange(min(int(nE), int(data.shape[0])), dtype=int)
    return MCMCCoeffTable(labels=[str(x) for x in labels], coeffs_by_label=coeffs_by_name, bins_present=bins_present)


def _find_mcmc_bins(*, mcmc_dir: str) -> np.ndarray:
    if not os.path.isdir(mcmc_dir):
        raise FileNotFoundError(f"MCMC directory not found: {mcmc_dir}")

    pat = re.compile(r"^mcmc_results_k(\d\d)\.npz$")
    bins: List[int] = []
    for fn in os.listdir(mcmc_dir):
        m = pat.match(fn)
        if m:
            bins.append(int(m.group(1)))

    if len(bins) == 0:
        raise FileNotFoundError(f"No mcmc_results_kXX.npz files found in: {mcmc_dir}")

    return np.array(sorted(set(bins)), dtype=int)


def load_mcmc_coeffs_by_label(*, mcmc_dir: str, stat: str = "f_ml", nE: Optional[int] = None) -> MCMCCoeffTable:
    bins_present = _find_mcmc_bins(mcmc_dir=mcmc_dir)
    if nE is None:
        nE = int(bins_present.max()) + 1

    coeffs_by_name: Dict[str, np.ndarray] = {}
    labels_ref: Optional[List[str]] = None

    for k in range(int(nE)):
        path = os.path.join(str(mcmc_dir), f"mcmc_results_k{k:02d}.npz")
        if not os.path.exists(path):
            continue

        npz = np.load(path, allow_pickle=True)
        labels = [str(x) for x in npz["labels"].tolist()]

        if labels_ref is None:
            labels_ref = labels
        else:
            if labels != labels_ref:
                raise RuntimeError(
                    "MCMC output labels differ across energy bins; cannot build a consistent table. "
                    f"First bin labels={labels_ref}, bin{k:02d} labels={labels}"
                )

        if stat not in npz:
            raise KeyError(f"MCMC file {path} missing key '{stat}'.")
        f = np.asarray(npz[stat], float).reshape(-1)
        if f.shape[0] != len(labels):
            raise RuntimeError(f"MCMC file {path} has {f.shape[0]} coeffs but {len(labels)} labels")

        for lab, val in zip(labels, f):
            if lab not in coeffs_by_name:
                coeffs_by_name[lab] = np.full(int(nE), np.nan, dtype=float)
            coeffs_by_name[lab][k] = float(val)

    if labels_ref is None:
        raise RuntimeError(f"No readable MCMC .npz files found in {mcmc_dir}")

    return MCMCCoeffTable(labels=labels_ref, coeffs_by_label=coeffs_by_name, bins_present=bins_present)


def pick_coeff(*, coeffs_by_label: Dict[str, np.ndarray], template_key: str) -> np.ndarray:
    if template_key in coeffs_by_label:
        return coeffs_by_label[template_key]

    if template_key == "nfw":
        nfw_keys = [k for k in coeffs_by_label.keys() if str(k).lower().startswith("nfw_")]
        if len(nfw_keys) == 1:
            return coeffs_by_label[nfw_keys[0]]
        if len(nfw_keys) > 1:
            raise RuntimeError(
                "Multiple MCMC labels start with 'nfw_'. Cannot map template key 'nfw' uniquely. "
                f"Candidates: {nfw_keys}"
            )
        raise KeyError("Could not find an MCMC coefficient for template key 'nfw'.")

    raise KeyError(f"Could not find an MCMC coefficient for template '{template_key}'.")


def combine_loopI(
    *,
    coeffs_by_label: Dict[str, np.ndarray],
    out_key: str = "loopI",
    drop_inputs: bool = True,
) -> Dict[str, np.ndarray]:
    out = dict(coeffs_by_label)

    if ("loopA" in out) and ("loopB" in out):
        out[out_key] = np.asarray(out["loopA"], float) + np.asarray(out["loopB"], float)
        if drop_inputs:
            del out["loopA"]
            del out["loopB"]

    return out


def save_coeff_table_txt(
    *,
    out_txt: str,
    x: np.ndarray,
    coeffs_by_label: Dict[str, np.ndarray],
    keys: Sequence[str],
    x_label: str = "k",
):
    x = np.asarray(x)
    cols = [x.reshape(-1)]
    labs = [str(x_label)]

    for key in keys:
        col = pick_coeff(coeffs_by_label=coeffs_by_label, template_key=str(key))
        col = np.asarray(col, float).reshape(-1)
        if col.shape[0] != x.shape[0]:
            raise ValueError(
                f"Column '{key}' has length {col.shape[0]} but x has length {x.shape[0]}"
            )
        cols.append(col)
        labs.append(str(key))

    arr = np.column_stack(cols)
    header = " ".join(labs)
    np.savetxt(out_txt, arr, header=header)


def E2_spectra_from_global_coeffs_counts(
    *,
    labels: Sequence[str],
    templates: Sequence[np.ndarray],
    coeffs_by_label: Dict[str, np.ndarray],
    expo: np.ndarray,
    omega: np.ndarray,
    dE_mev: np.ndarray,
    Ectr_mev: np.ndarray,
    mask3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ROI-averaged E^2 dN/dE per component from counts-space templates.

    Assumptions:
    - templates[j] has shape (nE, ny, nx) and is in expected counts per pixel.
    - coeffs_by_label[label][k] is the multiplicative coefficient for that template at bin k.
    - expo has shape (nE, ny, nx), omega has shape (ny, nx), dE_mev/Ectr_mev have shape (nE,).
    - mask3d has shape (nE, ny, nx) and True means include.

    Returns
    - E2_comp: (ncomp, nE)
    - E2_model: (nE,) sum over components
    """
    labels = [str(x) for x in labels]
    templates = [np.asarray(t, float) for t in templates]
    expo = np.asarray(expo, float)
    omega = np.asarray(omega, float)
    dE_mev = np.asarray(dE_mev, float).reshape(-1)
    Ectr_mev = np.asarray(Ectr_mev, float).reshape(-1)
    mask3d = np.asarray(mask3d, dtype=bool)

    if len(labels) != len(templates):
        raise ValueError(f"labels length {len(labels)} != templates length {len(templates)}")
    nE = int(Ectr_mev.shape[0])

    E2_comp = np.full((len(labels), nE), np.nan, dtype=float)
    E2_model = np.full(nE, np.nan, dtype=float)

    for k in range(nE):
        mk = mask3d[k]
        if not np.any(mk):
            continue

        denom_map = expo[k] * omega * float(dE_mev[k])
        denom = float(np.nansum(denom_map[mk]))
        if (not np.isfinite(denom)) or denom <= 0:
            continue

        e2 = float(Ectr_mev[k]) ** 2
        total_counts = 0.0

        for j, lab in enumerate(labels):
            a = pick_coeff(coeffs_by_label=coeffs_by_label, template_key=str(lab))
            a = np.asarray(a, float).reshape(-1)
            if k >= a.shape[0]:
                continue
            ak = float(a[k])
            if not np.isfinite(ak):
                continue

            counts_map = ak * templates[j][k]
            c = float(np.nansum(np.asarray(counts_map, float)[mk]))
            total_counts += c
            E2_comp[j, k] = e2 * (c / denom)

        E2_model[k] = e2 * (total_counts / denom)

    return E2_comp, E2_model
