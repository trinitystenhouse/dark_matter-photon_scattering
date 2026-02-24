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
