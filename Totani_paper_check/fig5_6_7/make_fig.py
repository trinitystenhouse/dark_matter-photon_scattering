import os

import numpy as np

from totani_helpers.cellwise_fit import fit_cellwise_poisson_mle_counts

# Reuse plotting/spectrum utilities from fig2_3
from fig2_3.make_totani_fig2_fig3_all import (  # noqa: E402
    assert_templates_match_counts,
    data_E2_spectrum_counts,
    load_mu_templates_from_fits,
    model_E2_spectrum_from_cells_counts,
    plot_fit_fig2,
)


def make_fig(
    *,
    labels,
    templates_dir,
    counts,
    expo,
    lon,
    lat,
    fit_mask3d,
    Ectr_mev,
    dE_mev,
    omega,
    roi_lon,
    roi_lat,
    cell_deg,
    out_png,
    out_coeff,
    title,
    invert_negative_bubbles=False,
):

    mu_list, _hdrs = load_mu_templates_from_fits(
        template_dir=templates_dir,
        labels=labels,
        filename_pattern="mu_{label}_counts.fits",
        hdu=0,
    )
    assert_templates_match_counts(counts, mu_list, labels)

    Ectr_gev = Ectr_mev / 1000.0

    res_fit = fit_cellwise_poisson_mle_counts(
        counts=counts,
        templates=mu_list,
        mask3d=fit_mask3d,
        lon=lon,
        lat=lat,
        roi_lon=float(roi_lon),
        roi_lat=float(roi_lat),
        cell_deg=float(cell_deg),
        nonneg=True,
    )
    cells = res_fit["cells"]
    coeff_cells = res_fit["coeff_cells"]

    # Plot region is the same as fit region for Fig.4
    mask3d_plot = fit_mask3d

    E2_data, E2err_data = data_E2_spectrum_counts(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_plot,
    )

    E2_comp, E2_model = model_E2_spectrum_from_cells_counts(
        coeff_cells=coeff_cells,
        cells=cells,
        templates=mu_list,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_plot,
    )

    comp_specs = {lab: E2_comp[j] for j, lab in enumerate(labels)}
    # Plotting convention used by Totani: show residual(+) and residual(-) as positive magnitudes on log-y.
    # Your templates are already constructed as non-negative maps, so keep them positive for plotting.
    # Only allow an explicit sign-flip if a caller requests it for a linear/diagnostic plot.
    if invert_negative_bubbles and ("bubbles_neg" in comp_specs):
        comp_specs["bubbles_neg"] = -comp_specs["bubbles_neg"]

    if "bubbles_pos" in comp_specs:
        comp_specs["residual(+) (FB)"] = comp_specs.pop("bubbles_pos")
    if "bubbles_neg" in comp_specs:
        comp_specs["residual(-)"] = comp_specs.pop("bubbles_neg")

    comp_specs["MODEL_SUM"] = E2_model

    plot_fit_fig2(
        Ectr_gev=Ectr_gev,
        data_y=E2_data,
        data_yerr=E2err_data,
        comp_specs=comp_specs,
        outpath=out_png,
        title=title,
    )

    with open(out_coeff, "w") as f:
        f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
        for k in range(counts.shape[0]):
            csum = np.zeros(len(labels), float)
            for ci, cell2d in enumerate(cells):
                cm = mask3d_plot[k] & cell2d
                if not np.any(cm):
                    continue
                a = coeff_cells[ci, k, :]
                if not np.all(np.isfinite(a)):
                    continue
                for j in range(len(labels)):
                    s = float(np.nansum(mu_list[j][k][cm]))
                    if np.isfinite(s) and s != 0.0:
                        csum[j] += float(a[j]) * s

            f.write(
                f"{k:02d} {Ectr_gev[k]:.6g} "
                + " ".join(f"{csum[j]:.6g}" for j in range(len(labels)))
                + "\n"
            )

    print("✓ wrote", out_png)
    print("✓ wrote", out_coeff)

    return {
        "cells": cells,
        "coeff_cells": coeff_cells,
        "E2_data": E2_data,
        "E2err_data": E2err_data,
        "E2_comp": E2_comp,
        "E2_model": E2_model,
    }