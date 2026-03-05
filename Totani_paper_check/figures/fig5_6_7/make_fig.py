import os

import numpy as np

from totani_helpers.fit_utils import E2_from_pred_counts_maps
from totani_helpers.mcmc_io import combine_loopI, pick_coeff
from totani_helpers.plotting import plot_E2_dnde_multi_totani

# Reuse plotting/spectrum utilities from fig2_3
from fig2_3.make_totani_fig2_fig3_all import (  # noqa: E402
    assert_templates_match_counts,
    data_E2_spectrum_counts,
    load_mu_templates_from_fits,
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
    coeffs_by_label,
    combine_loopI_default=True,
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

    coeffs_plot = dict(coeffs_by_label)
    if bool(combine_loopI_default):
        coeffs_plot = combine_loopI(coeffs_by_label=coeffs_plot, out_key="loopI", drop_inputs=False)

    mask3d_plot = fit_mask3d
    mask2d_plot = np.all(mask3d_plot, axis=0)

    E2_data, E2err_data = data_E2_spectrum_counts(
        counts=counts,
        expo=expo,
        omega=omega,
        dE_mev=dE_mev,
        Ectr_mev=Ectr_mev,
        mask3d=mask3d_plot,
    )

    comp_specs = {}
    E2_model = np.zeros_like(Ectr_mev, dtype=float)

    for lab, mu in zip(labels, mu_list):
        if str(lab) == "loopI" and bool(combine_loopI_default):
            aA = pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopA") if "loopA" in coeffs_plot else None
            aB = pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopB") if "loopB" in coeffs_plot else None
            if aA is None or aB is None:
                a = pick_coeff(coeffs_by_label=coeffs_plot, template_key="loopI")
                pred = np.asarray(a, float)[:, None, None] * np.asarray(mu, float)
            else:
                muA = load_mu_templates_from_fits(
                    template_dir=templates_dir,
                    labels=["loopA"],
                    filename_pattern="mu_{label}_counts.fits",
                    hdu=0,
                )[0][0]
                muB = load_mu_templates_from_fits(
                    template_dir=templates_dir,
                    labels=["loopB"],
                    filename_pattern="mu_{label}_counts.fits",
                    hdu=0,
                )[0][0]
                pred = np.asarray(aA, float)[:, None, None] * np.asarray(muA, float) + np.asarray(aB, float)[:, None, None] * np.asarray(muB, float)
        else:
            a = pick_coeff(coeffs_by_label=coeffs_plot, template_key=str(lab))
            pred = np.asarray(a, float)[:, None, None] * np.asarray(mu, float)

        E2 = E2_from_pred_counts_maps(
            pred_counts_map=pred,
            expo=expo,
            omega=omega,
            dE_mev=dE_mev,
            Ectr_mev=Ectr_mev,
            mask2d=mask2d_plot,
        )
        comp_specs[str(lab)] = E2
        E2_model += E2
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

    curves = [(lab, y) for (lab, y) in comp_specs.items()]
    curves.append(("data", E2_data, E2err_data))
    plot_E2_dnde_multi_totani(
        Ectr_mev=Ectr_mev,
        curves=curves,
        out_png=out_png,
        title=title,
    )

    with open(out_coeff, "w") as f:
        f.write("# k  Ectr(GeV)  " + "  ".join(labels) + "\n")
        for k in range(counts.shape[0]):
            csum = np.zeros(len(labels), float)
            for j, lab in enumerate(labels):
                if str(lab) == "loopI" and bool(combine_loopI_default) and ("loopA" in coeffs_plot) and ("loopB" in coeffs_plot):
                    aA = np.asarray(coeffs_plot["loopA"], float).reshape(-1)[k]
                    aB = np.asarray(coeffs_plot["loopB"], float).reshape(-1)[k]
                    muA = load_mu_templates_from_fits(
                        template_dir=templates_dir,
                        labels=["loopA"],
                        filename_pattern="mu_{label}_counts.fits",
                        hdu=0,
                    )[0][0]
                    muB = load_mu_templates_from_fits(
                        template_dir=templates_dir,
                        labels=["loopB"],
                        filename_pattern="mu_{label}_counts.fits",
                        hdu=0,
                    )[0][0]
                    csum[j] = float(aA) * float(np.nansum(muA[k][mask3d_plot[k]])) + float(aB) * float(np.nansum(muB[k][mask3d_plot[k]]))
                else:
                    a = float(np.asarray(pick_coeff(coeffs_by_label=coeffs_plot, template_key=str(lab)), float).reshape(-1)[k])
                    csum[j] = a * float(np.nansum(mu_list[j][k][mask3d_plot[k]]))

            f.write(
                f"{k:02d} {Ectr_gev[k]:.6g} "
                + " ".join(f"{csum[j]:.6g}" for j in range(len(labels)))
                + "\n"
            )

    print("✓ wrote", out_png)
    print("✓ wrote", out_coeff)

    return {
        "E2_data": E2_data,
        "E2err_data": E2err_data,
        "comp_specs": comp_specs,
        "E2_model": E2_model,
    }