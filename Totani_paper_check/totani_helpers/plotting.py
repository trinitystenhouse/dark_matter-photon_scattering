import numpy as np


def _norm_key(label: str) -> str:
    s = str(label)
    s = s.strip()
    s_low = s.lower()

    if s_low.startswith("nfw") or ("nfw" in s_low):
        return "nfw"

    if s_low in ("ps", "point_sources", "pointsource", "point source", "point sources"):
        return "ps"

    if s_low in ("ics",):
        return "ics"

    if s_low in ("gas",):
        return "gas"

    if s_low in ("iso", "isotropic"):
        return "iso"

    if s_low in ("loopi", "loop i", "loop_i", "loop") or s_low in ("loopa", "loopb"):
        return "loopI"

    if "fb" in s_low or "bubble" in s_low:
        if "neg" in s_low:
            return "fb_neg"
        if "flat" in s_low:
            return "fb_flat"
        if "pos" in s_low:
            return "fb_pos"
        return "fb"

    return s


def totani_component_style(label: str) -> dict:
    key = _norm_key(label)

    # Requested scheme
    # - point sources: green, diamond
    # - ICS: black, pentagon
    # - gas: black, square
    # - Loop I: pink, triangle-right
    # - isotropic: brown, triangle-left
    # - Fermi bubbles: blue, up-triangle for flat+pos, down-triangle for neg
    # - NFW: red, circle
    # - dotted lines
    base = {
        "linestyle": ":",
        "linewidth": 1.6,
        "markersize": 6,
        "markeredgewidth": 0.0,
    }

    styles = {
        "ps": {"color": "green", "marker": "D"},
        "ics": {"color": "black", "marker": "p"},
        "gas": {"color": "black", "marker": "s"},
        "loopI": {"color": "hotpink", "marker": ">"},
        "iso": {"color": "saddlebrown", "marker": "<"},
        "fb_flat": {"color": "royalblue", "marker": "^"},
        "fb_pos": {"color": "royalblue", "marker": "^"},
        "fb_neg": {"color": "royalblue", "marker": "v"},
        "fb": {"color": "royalblue", "marker": "^"},
        "nfw": {"color": "red", "marker": "o"},
    }

    out = dict(base)
    out.update(styles.get(key, {}))
    return out


def plot_E2_dnde_multi_totani(
    Ectr_mev,
    curves,
    *,
    out_png=None,
    title=None,
    ax=None,
    yerr_by_label=None,
    xlabel="Energy (GeV)",
    ylabel=r"$E^2 \,\langle \mathrm{d}N/\mathrm{d}E \rangle$  [MeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
    legend=True,
):
    import matplotlib.pyplot as plt

    Ectr_gev = np.asarray(Ectr_mev, float) / 1000.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for item in curves:
        if isinstance(item, (tuple, list)) and (len(item) >= 2):
            lab = item[0]
            y_in = item[1]
            yerr = None
            if len(item) >= 3:
                yerr = item[2]
        else:
            raise TypeError("curves must contain (label, y) or (label, y, yerr) tuples")

        y = np.asarray(y_in, float)
        m = np.isfinite(Ectr_gev) & np.isfinite(y) & (Ectr_gev > 0)
        if not np.any(m):
            continue

        if yerr is None and yerr_by_label is not None:
            yerr = yerr_by_label.get(str(lab))

        style = totani_component_style(str(lab))

        if yerr is not None:
            yerr = np.asarray(yerr, float)
            if yerr.shape == y.shape:
                yerr = yerr[m]
            ax.errorbar(
                Ectr_gev[m],
                y[m],
                yerr=yerr,
                label=str(lab),
                color=style.get("color", None),
                linestyle=style.get("linestyle", ":"),
                marker=style.get("marker", "o"),
                linewidth=style.get("linewidth", 1.6),
                markersize=style.get("markersize", 6),
                capsize=2,
            )
        else:
            ax.plot(
                Ectr_gev[m],
                y[m],
                label=str(lab),
                color=style.get("color", None),
                linestyle=style.get("linestyle", ":"),
                marker=style.get("marker", "o"),
                linewidth=style.get("linewidth", 1.6),
                markersize=style.get("markersize", 6),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=9)

    fig.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

    return ax
