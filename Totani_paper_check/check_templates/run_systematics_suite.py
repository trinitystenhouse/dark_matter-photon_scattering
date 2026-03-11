#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS

from totani_helpers.mcmc_io import E2_spectra_from_global_coeffs_counts, combine_loopI, load_mcmc_coeffs_by_label
from totani_helpers.totani_io import (
    lonlat_grids,
    load_mask_any_shape,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)
from totani_helpers.fit_utils import build_fit_mask3d, load_mu_templates_from_fits


@dataclass
class CaseSpec:
    name: str
    mcmc_dir: str
    templates_dir: str


def _parse_case(s: str) -> CaseSpec:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--case must be 'name,mcmc_dir,templates_dir'")
    name, mcmc_dir, templates_dir = parts
    return CaseSpec(name=name, mcmc_dir=mcmc_dir, templates_dir=templates_dir)


def _roi_quadrant_mask(*, lon2d: np.ndarray, lat2d: np.ndarray, which: str) -> np.ndarray:
    which = str(which).upper()
    if which == "ALL":
        return np.ones_like(lon2d, dtype=bool)
    if which == "NE":
        return (lon2d >= 0) & (lat2d >= 0)
    if which == "NW":
        return (lon2d < 0) & (lat2d >= 0)
    if which == "SE":
        return (lon2d >= 0) & (lat2d < 0)
    if which == "SW":
        return (lon2d < 0) & (lat2d < 0)
    raise ValueError(f"Unknown quadrant '{which}'")


def _plot_overlay_all_components(
    *,
    Ectr_mev: np.ndarray,
    E2_by_case_by_label: Dict[str, Dict[str, np.ndarray]],
    labels: Sequence[str],
    out_png: str,
    title: str,
):
    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0

    n = len(labels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False)

    for i, lab in enumerate(labels):
        ax = axes[i // ncol][i % ncol]
        ax.set_xscale("log")
        ax.set_yscale("log")
        for case_name, curves in E2_by_case_by_label.items():
            y = curves.get(str(lab))
            if y is None:
                continue
            ax.plot(Ectr_gev, y, marker="o", ms=2.5, lw=1.2, label=str(case_name))
        ax.set_title(str(lab))
        ax.grid(True, which="both", alpha=0.25)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper right", frameon=False)

    fig.suptitle(str(title))
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("✓ wrote", out_png)


def _plot_overlay_component(
    *,
    Ectr_mev: np.ndarray,
    curves_by_case: Sequence[Tuple[str, np.ndarray]],
    out_png: str,
    title: str,
):
    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0
    fig = plt.figure(figsize=(7.0, 5.0))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for case_name, y in curves_by_case:
        ax.plot(Ectr_gev, y, marker="o", ms=3.0, lw=1.5, label=str(case_name))
    ax.set_title(str(title))
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel(r"$E^2 dN/dE$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("✓ wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description="Systematics suite: compare multiple MCMC cases via overlaid E^2 spectra and optional template grids.")
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Repeatable. Format: name,mcmc_dir,templates_dir",
    )
    ap.add_argument(
        "--cases-root",
        default=None,
        help="Optional root directory; if provided and --case is empty, autodiscovers immediate subdirs containing mcmc_results_k*.npz and a 'templates' directory.",
    )
    ap.add_argument(
        "--mcmc-glob",
        default="mcmc_results_k*.npz",
        help="Glob used to identify per-bin MCMC outputs within a case mcmc directory.",
    )

    ap.add_argument(
        "--counts",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "fermi_data", "totani", "processed", "counts_ccube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--expo",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "fermi_data", "totani", "processed", "expcube_1000to1000000.fits"),
    )
    ap.add_argument(
        "--ext-mask",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "fermi_data", "totani", "processed", "templates", "mask_extended_sources.fits"),
    )

    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "systematics_outputs"))
    ap.add_argument("--binsz", type=float, default=0.125)
    ap.add_argument("--roi-lon", type=float, default=60.0)
    ap.add_argument("--roi-lat", type=float, default=60.0)
    ap.add_argument("--disk-cut", type=float, default=10.0)

    ap.add_argument(
        "--labels",
        nargs="+",
        default=["gas", "iso", "ps", "loopI", "ics", "fb_flat"],
    )
    ap.add_argument("--mcmc-stat", default="f_ml", choices=["f_ml", "f_p50", "f_p16", "f_p84"])
    ap.add_argument("--combine-loopI", action="store_true")

    ap.add_argument("--make-spectra", action="store_true")
    ap.add_argument("--make-template-grids", action="store_true")
    ap.add_argument(
        "--per-component",
        action="store_true",
        help="Also write per-component overlay spectra plots (in addition to the multi-panel grid).",
    )
    ap.add_argument(
        "--quadrants",
        action="store_true",
        help="Also compute E2 spectra separately in NE/NW/SE/SW sub-ROIs using the same coefficients (no refit).",
    )
    ap.add_argument("--Egev", type=float, default=21.0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cases: List[CaseSpec] = []
    if args.case:
        cases = [_parse_case(x) for x in args.case]
    elif args.cases_root:
        root = str(args.cases_root)
        if os.path.isdir(root):
            for sub in sorted(os.listdir(root)):
                p = os.path.join(root, sub)
                if not os.path.isdir(p):
                    continue
                mdir = p
                tdir = os.path.join(p, "templates")
                if not os.path.isdir(tdir):
                    tdir = os.path.join(p, "processed", "templates")
                if os.path.isdir(mdir) and os.path.isdir(tdir):
                    cases.append(CaseSpec(name=sub, mcmc_dir=mdir, templates_dir=tdir))
        if not cases:
            raise SystemExit("No cases found. Provide --case or a valid --cases-root.")
    else:
        raise SystemExit("Provide --case or --cases-root")

    counts, hdr, _Emin, _Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(args.counts)
    nE, ny, nx = counts.shape
    expo_raw, E_expo_mev = read_exposure(args.expo)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    omega = pixel_solid_angle_map(WCS(hdr).celestial, ny, nx, float(args.binsz))
    lon2d, lat2d = lonlat_grids(WCS(hdr).celestial, ny, nx)

    roi2d = (np.abs(lon2d) <= float(args.roi_lon)) & (np.abs(lat2d) <= float(args.roi_lat))
    disk_keep = np.abs(lat2d) >= float(args.disk_cut)

    ext_keep3d = np.ones((nE, ny, nx), dtype=bool)
    if args.ext_mask and os.path.exists(args.ext_mask):
        ext_keep3d = load_mask_any_shape(args.ext_mask, counts.shape)

    fit_mask3d = build_fit_mask3d(
        roi2d=roi2d,
        srcmask3d=ext_keep3d,
        counts=counts,
        expo=expo,
        extra2d=disk_keep,
    )

    regions = ["ALL"]
    if bool(args.quadrants):
        regions = ["ALL", "NE", "NW", "SE", "SW"]

    # Pre-load templates and coefficient tables per case.
    labels_req = [str(x) for x in args.labels]

    coeffs_by_case: Dict[str, Dict[str, np.ndarray]] = {}
    mu_by_case: Dict[str, List[np.ndarray]] = {}

    for cs in cases:
        tab = load_mcmc_coeffs_by_label(mcmc_dir=str(cs.mcmc_dir), stat=str(args.mcmc_stat), nE=nE)
        coeffs = dict(tab.coeffs_by_label)
        if bool(args.combine_loopI):
            coeffs = combine_loopI(coeffs_by_label=coeffs, out_key="loopI", drop_inputs=True)
        coeffs_by_case[str(cs.name)] = coeffs

        mu_list, _hdrs = load_mu_templates_from_fits(
            template_dir=str(cs.templates_dir),
            labels=labels_req,
            filename_pattern="mu_{label}_counts.fits",
            hdu=0,
        )
        mu_by_case[str(cs.name)] = [np.asarray(m, float) for m in mu_list]

    if args.make_spectra:
        for region in regions:
            region2d = roi2d & disk_keep & _roi_quadrant_mask(lon2d=lon2d, lat2d=lat2d, which=region)
            mask3d = fit_mask3d & region2d[None, :, :]

            E2_by_case_by_label: Dict[str, Dict[str, np.ndarray]] = {}
            for cs in cases:
                coeffs = coeffs_by_case[str(cs.name)]
                mu_list = mu_by_case[str(cs.name)]
                E2_comp, _E2_model = E2_spectra_from_global_coeffs_counts(
                    labels=labels_req,
                    templates=mu_list,
                    coeffs_by_label=coeffs,
                    expo=expo,
                    omega=omega,
                    dE_mev=dE_mev,
                    Ectr_mev=Ectr_mev,
                    mask3d=mask3d,
                )
                E2_by_case_by_label[str(cs.name)] = {lab: E2_comp[i] for i, lab in enumerate(labels_req)}

            out_all = os.path.join(args.outdir, f"systematics_overlay_all_components_{region}.png")
            _plot_overlay_all_components(
                Ectr_mev=Ectr_mev,
                E2_by_case_by_label=E2_by_case_by_label,
                labels=labels_req,
                out_png=out_all,
                title=f"Systematics suite: E2 spectra (cases overlaid) [{region}]",
            )

            if bool(args.per_component):
                for lab in labels_req:
                    curves = []
                    for cs in cases:
                        y = E2_by_case_by_label.get(str(cs.name), {}).get(lab)
                        if y is not None:
                            curves.append((str(cs.name), y))
                    if not curves:
                        continue
                    out_png = os.path.join(args.outdir, f"systematics_overlay_{lab}_{region}.png")
                    _plot_overlay_component(
                        Ectr_mev=Ectr_mev,
                        curves_by_case=curves,
                        out_png=out_png,
                        title=f"Systematics: {lab} [{region}]",
                    )

    if args.make_template_grids:
        # Lightweight template grid at a single energy for each case.
        Ectr_gev = Ectr_mev / 1000.0
        k = int(np.argmin(np.abs(Ectr_gev - float(args.Egev))))
        for cs in cases:
            mu_list = mu_by_case[str(cs.name)]
            n = len(labels_req)
            ncol = 3
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False)
            for i, (lab, mu) in enumerate(zip(labels_req, mu_list)):
                ax = axes[i // ncol][i % ncol]
                img = np.asarray(mu[k], float)
                img = np.where(region2d if 'region2d' in locals() else (roi2d & disk_keep), img, np.nan)
                im = ax.imshow(img, origin="lower", cmap="viridis")
                ax.set_title(str(lab))
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(f"Templates @ {Ectr_gev[k]:.2f} GeV [{cs.name}]")
            fig.tight_layout()
            out_png = os.path.join(args.outdir, f"template_grid_{cs.name}_E{Ectr_gev[k]:.2f}GeV.png")
            fig.savefig(out_png, dpi=220)
            plt.close(fig)
            print("✓ wrote", out_png)


if __name__ == "__main__":
    main()
