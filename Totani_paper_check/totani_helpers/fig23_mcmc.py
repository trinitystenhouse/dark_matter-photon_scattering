import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fit_utils import E2_from_pred_counts_maps, build_fit_mask3d, load_mu_templates_from_fits
from .mcmc_io import combine_loopI, load_mcmc_coeffs_by_label, pick_coeff, save_coeff_table_txt
from .plotting import plot_E2_dnde_multi_totani
from .totani_io import (
    load_mask_any_shape,
    lonlat_grids,
    pixel_solid_angle_map,
    read_counts_and_ebounds,
    read_exposure,
    resample_exposure_logE,
)


def _build_counts_cubes_from_coeffs(*, templates_counts: Dict[str, np.ndarray], coeffs_by_label: Dict[str, np.ndarray]):
    names = list(templates_counts.keys())
    first = templates_counts[names[0]]
    nE, ny, nx = first.shape

    comp_counts: Dict[str, np.ndarray] = {}
    model_total = np.zeros((nE, ny, nx), dtype=float)

    for name in names:
        T = np.asarray(templates_counts[name], float)
        a = pick_coeff(coeffs_by_label=coeffs_by_label, template_key=name)
        a = np.asarray(a, float).reshape(-1)
        if a.shape[0] != nE:
            raise RuntimeError(f"Coeff for '{name}' has length {a.shape[0]} but nE={nE}")

        cube = a[:, None, None] * T
        comp_counts[name] = cube
        model_total += cube

    return comp_counts, model_total


def _save_curves_txt(*, out_txt: str, Ectr_mev: np.ndarray, curves: Sequence[Tuple[str, np.ndarray]]):
    Egev = np.asarray(Ectr_mev, float) / 1000.0
    labels = [str(x[0]) for x in curves]
    ys = [np.asarray(x[1], float).reshape(-1) for x in curves]
    nE = int(Egev.shape[0])
    for y in ys:
        if y.shape[0] != nE:
            raise ValueError(f"Curve length mismatch for {out_txt}: got {y.shape[0]} expected {nE}")

    arr = np.column_stack([Egev] + ys)
    header = "E_GeV " + " ".join(labels)
    np.savetxt(out_txt, arr, header=header)


def plot_E2_dnde_multi_diagnostic(Ectr_mev, curves, *, out_png=None, title=None):
    import matplotlib.pyplot as plt

    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0

    plt.figure(figsize=(8, 6))
    for lab, y_in in curves:
        y = np.asarray(y_in, float)
        m = np.isfinite(Ectr_gev) & np.isfinite(y) & (Ectr_gev > 0)
        if not np.any(m):
            continue
        plt.plot(Ectr_gev[m], y[m], marker="o", label=str(lab))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (GeV)")
    plt.ylabel(r"$E^2 \,\langle \mathrm{d}N/\mathrm{d}E \rangle$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    if title is not None:
        plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
        plt.close()
    else:
        plt.show()


def _plot_multi(Ectr_mev, curves, *, out_png=None, title=None, plot_style: str = "diagnostic"):
    if plot_style == "totani":
        plot_E2_dnde_multi_totani(Ectr_mev, curves, out_png=out_png, title=title)
    else:
        plot_E2_dnde_multi_diagnostic(Ectr_mev, curves, out_png=out_png, title=title)


def make_fig2_fig3_plots_from_mcmc(
    *,
    counts_path: str,
    expo_path: str,
    templates_dir: str,
    mcmc_dir: str,
    outdir: str,
    mcmc_stat: str = "f_ml",
    plot_style: str = "diagnostic",
    ext_mask_path: Optional[str] = None,
    roi_lon: float = 60.0,
    roi_lat: float = 60.0,
    disk_cut: float = 10.0,
    binsz: float = 0.125,
    labels: Optional[Sequence[str]] = None,
):
    os.makedirs(outdir, exist_ok=True)

    counts, hdr, _Emin, _Emax, Ectr_mev, dE_mev = read_counts_and_ebounds(counts_path)
    nE, ny, nx = counts.shape

    expo_raw, E_expo_mev = read_exposure(expo_path)
    expo = resample_exposure_logE(expo_raw, E_expo_mev, Ectr_mev)
    if expo.shape != counts.shape:
        raise ValueError(f"Exposure shape {expo.shape} != counts shape {counts.shape}")

    from astropy.wcs import WCS

    wcs = WCS(hdr).celestial
    omega = pixel_solid_angle_map(wcs, ny, nx, binsz)
    lon, lat = lonlat_grids(wcs, ny, nx)

    roi2d = (np.abs(lon) <= float(roi_lon)) & (np.abs(lat) <= float(roi_lat))

    if ext_mask_path is None:
        ext_mask3d = np.ones((nE, ny, nx), dtype=bool)
    else:
        ext_mask3d = load_mask_any_shape(ext_mask_path, counts.shape)

    fit_mask3d = build_fit_mask3d(roi2d=roi2d, srcmask3d=ext_mask3d, counts=counts, expo=expo)
    fit_mask_2d = np.all(fit_mask3d, axis=0)

    if labels is None:
        labels = [
            "gas",
            "iso",
            "ps",
            "nfw_NFW_g1_rho2.5_rs21_R08_rvir402_ns2048_normpole_pheno",
            "loopA",
            "loopB",
            "ics",
            "fb_flat",
        ]

    mu_list, _headers = load_mu_templates_from_fits(
        template_dir=templates_dir,
        labels=list(labels),
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )

    templates_counts: Dict[str, np.ndarray] = {}
    for lab, mu in zip(labels, mu_list):
        key = "nfw" if str(lab).startswith("nfw_") else str(lab)
        templates_counts[key] = np.asarray(mu, float)

    tab = load_mcmc_coeffs_by_label(mcmc_dir=mcmc_dir, stat=mcmc_stat, nE=nE)
    coeffs_by_label = tab.coeffs_by_label

    # Save coefficients used for plotting, mapped to template keys (and including loopI if available).
    coeffs_plot = dict(coeffs_by_label)
    if ("loopA" in coeffs_plot) and ("loopB" in coeffs_plot):
        coeffs_plot = combine_loopI(coeffs_by_label=coeffs_plot, out_key="loopI", drop_inputs=False)

    keys_for_coeff_dump: List[str] = []
    for k in templates_counts.keys():
        if k in ("loopA", "loopB") and ("loopI" in coeffs_plot):
            continue
        keys_for_coeff_dump.append(str(k))
    if "loopI" in coeffs_plot and "loopI" not in keys_for_coeff_dump:
        keys_for_coeff_dump.append("loopI")

    # x-axis is energy-bin index, since Ectr values live in counts FITS.
    xk = np.arange(nE, dtype=int)
    save_coeff_table_txt(
        out_txt=os.path.join(outdir, f"fit_coefficients_mcmc_{mcmc_stat}.txt"),
        x=xk,
        coeffs_by_label=coeffs_plot,
        keys=keys_for_coeff_dump,
        x_label="k",
    )
    if nE > 0:
        k0_vals = {k: float(np.asarray(coeffs_plot[k]).reshape(-1)[0]) for k in keys_for_coeff_dump if k in coeffs_plot}
        print(f"[mcmc coeffs] k=0: {k0_vals}")

    comp_counts_dict, model_counts_total = _build_counts_cubes_from_coeffs(
        templates_counts=templates_counts,
        coeffs_by_label=coeffs_by_label,
    )

    # Combine Loop I for plotting
    if ("loopA" in comp_counts_dict) and ("loopB" in comp_counts_dict):
        comp_counts_dict = dict(comp_counts_dict)
        comp_counts_dict["loopI"] = np.asarray(comp_counts_dict["loopA"], float) + np.asarray(comp_counts_dict["loopB"], float)
        del comp_counts_dict["loopA"]
        del comp_counts_dict["loopB"]
        bkg_names = [k for k in templates_counts.keys() if k not in ("loopA", "loopB")] + ["loopI"]
    else:
        bkg_names = list(templates_counts.keys())

    # Fig2 (include disk)
    curves2 = []
    for name in bkg_names:
        E2 = E2_from_pred_counts_maps(
            pred_counts_map=np.asarray(comp_counts_dict[name], float),
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=fit_mask_2d,
        )
        curves2.append((name, E2))
    E2_tot2 = E2_from_pred_counts_maps(
        pred_counts_map=np.asarray(model_counts_total, float),
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask2d=fit_mask_2d,
    )
    curves2.append(("total", E2_tot2))

    _save_curves_txt(
        out_txt=os.path.join(outdir, f"E2_dnde_background_components_mcmc_fig2_{mcmc_stat}.txt"),
        Ectr_mev=Ectr_mev,
        curves=curves2,
    )
    _plot_multi(
        Ectr_mev,
        curves2,
        out_png=os.path.join(outdir, "E2_dnde_background_components_mcmc_fig2.png"),
        title=f"MCMC background components ({mcmc_stat})",
        plot_style=plot_style,
    )

    # Fig3 (exclude disk in plot)
    plot_mask2d = fit_mask_2d & (np.abs(lat) >= float(disk_cut))
    curves3 = []
    for name in bkg_names:
        E2 = E2_from_pred_counts_maps(
            pred_counts_map=np.asarray(comp_counts_dict[name], float),
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=plot_mask2d,
        )
        curves3.append((name, E2))
    E2_tot3 = E2_from_pred_counts_maps(
        pred_counts_map=np.asarray(model_counts_total, float),
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask2d=plot_mask2d,
    )
    curves3.append(("total", E2_tot3))

    _save_curves_txt(
        out_txt=os.path.join(outdir, f"E2_dnde_background_components_mcmc_fig3_{mcmc_stat}.txt"),
        Ectr_mev=Ectr_mev,
        curves=curves3,
    )
    _plot_multi(
        Ectr_mev,
        curves3,
        out_png=os.path.join(outdir, "E2_dnde_background_components_mcmc_fig3.png"),
        title=f"MCMC background components ({mcmc_stat}), |b|>={float(disk_cut):g} deg",
        plot_style=plot_style,
    )

    return {
        "Ectr_mev": Ectr_mev,
        "dE_mev": dE_mev,
        "bkg_names": bkg_names,
        "curves2": curves2,
        "curves3": curves3,
        "coeffs_by_label": coeffs_plot,
    }
