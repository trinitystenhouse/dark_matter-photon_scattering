import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.image as mpimg


def latex_sci(x, sig=2):
    x = float(x)
    if x == 0.0:
        return r"0"
    s = f"{x:.{int(sig)}e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp_i = int(exp)
    return rf"{mant}\times10^{{{exp_i}}}"


def operator_title(op):
    op = str(op)
    if op == "rayleigh_even":
        return "Rayleigh even"
    if op == "rayleigh_odd":
        return "Rayleigh odd"
    if op == "rayleigh_full":
        return "Rayleigh full"
    if op == "dipole_magnetic":
        return "Magnetic dipole"
    if op == "dipole_electric":
        return "Electric dipole"
    if op == "charge_radius":
        return "Charge radius"
    if op == "anapole":
        return "Anapole"
    return op


def add_hatched_region_from_contour(
    *,
    ax,
    X,
    Y,
    Z,
    level,
    upper_level,
    hatch="////",
    edgecolor="c",
    zorder=3,
    outline_lw=1.5,
):
    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=[float(level), float(upper_level)],
        colors=["none"],
        antialiased=False,
    )

    segs = []
    if hasattr(cf, "allsegs") and isinstance(cf.allsegs, (list, tuple)) and len(cf.allsegs) > 0:
        segs = cf.allsegs[0]

    for col in getattr(cf, "collections", []):
        try:
            col.remove()
        except Exception:
            pass

    if not segs:
        return False

    for seg in segs:
        seg = np.asarray(seg, dtype=float)
        if seg.ndim != 2 or seg.shape[0] < 3:
            continue
        if not np.allclose(seg[0], seg[-1]):
            seg = np.vstack([seg, seg[0]])
        codes = np.full(seg.shape[0], Path.LINETO, dtype=int)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(seg, codes)

        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=edgecolor,
            hatch=hatch,
            lw=outline_lw,
            zorder=zorder,
        )

        with mpl.rc_context({"hatch.color": edgecolor, "hatch.linewidth": 1.0}):
            ax.add_patch(patch)

    return True


def make_combined_tau_vs_lambda_beamer(
    *,
    operators,
    compute_curve,
    tau_needed,
    tau_energy_label,
    out_base,
    header_text,
    ncols=3,
):
    operators = [str(o) for o in operators]
    nops = int(len(operators))
    if nops <= 0:
        raise ValueError("operators must be a non-empty list")

    ncols = int(max(1, ncols))
    nrows = int(np.ceil(float(nops) / float(ncols)))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, 2.8 * float(nrows)),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    fig.text(
        0.5,
        0.995,
        str(header_text),
        ha="center",
        va="top",
        fontsize=10,
        color="w",
    )

    for i, op in enumerate(operators):
        r = int(i // ncols)
        c = int(i % ncols)
        ax = axes[r, c]

        Lambda_grid, tau_max_lambda = compute_curve(op)
        Lambda_grid = np.asarray(Lambda_grid, dtype=float)
        tau_max_lambda = np.asarray(tau_max_lambda, dtype=float)

        ax.plot(Lambda_grid, tau_max_lambda, lw=2)
        ax.axhline(float(tau_needed), color="w", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(operator_title(op), fontsize=11)

        if r == (nrows - 1):
            ax.set_xlabel(r"$\\Lambda\\,[\\mathrm{GeV}]$")
        if c == 0:
            ax.set_ylabel(rf"$\\tau_\\max$ ({str(tau_energy_label)})")

    for j in range(nops, nrows * ncols):
        r = int(j // ncols)
        c = int(j % ncols)
        axes[r, c].axis("off")

    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)

    fig.savefig(str(out_base) + ".png", dpi=200)
    fig.savefig(str(out_base) + ".pdf")
    plt.close(fig)


def make_combined_tau_grid_png_beamer(
    *,
    operators,
    png_paths,
    out_base,
    header_text,
    ncols=3,
):
    operators = [str(o) for o in operators]
    png_paths = [str(p) for p in png_paths]
    if len(operators) != len(png_paths):
        raise ValueError("operators and png_paths must have the same length")

    nops = int(len(operators))
    if nops <= 0:
        raise ValueError("operators must be a non-empty list")

    ncols = int(max(1, ncols))
    nrows = int(np.ceil(float(nops) / float(ncols)))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, 3.1 * float(nrows)),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    # --- header ---
    fig.text(
        0.5,
        0.972,            # lower than 0.995 -> less wasted space above header
        str(header_text),
        ha="center",
        va="top",
        fontsize=10,
        color="w",
    )

    # --- spacing ---
    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.10,
        top=0.90,         # increase (e.g. 0.92–0.94) to bring plots closer to header
        hspace=0.15,      # smaller -> subplots closer vertically
        wspace=0.10,      # smaller -> subplots closer horizontally
    )

    for i, (op, p) in enumerate(zip(operators, png_paths)):
        r = int(i // ncols)
        c = int(i % ncols)
        ax = axes[r, c]

        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_title(operator_title(op), fontsize=11, pad=6, color="w")
        ax.axis("off")

    for j in range(nops, nrows * ncols):
        r = int(j // ncols)
        c = int(j % ncols)
        axes[r, c].axis("off")

    fig.savefig(str(out_base) + ".png", dpi=200)
    fig.savefig(str(out_base) + ".pdf")
    plt.close(fig)
