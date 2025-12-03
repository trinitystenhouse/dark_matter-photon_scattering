import matplotlib.pyplot as plt
from cycler import cycler

def set_plot_style(
    style="light",
    font="Times New Roman",
    base_fontsize=15,
    linewidth=2.0,
    n_colors=6,
):
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    light = dict(
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        axes_edgecolor="#262626",
        text_color="#262626",
        tick_color="#262626",
        grid_color="#CCCCCC",
    )
    dark = dict(
        figure_facecolor="#111217",
        axes_facecolor="#111217",
        axes_edgecolor="#EAEAEA",
        text_color="#EAEAEA",
        tick_color="#EAEAEA",
        grid_color="#555555",
    )
    theme = light if style == "light" else dark

    plt.rcParams.update({
        # Figure & font
        "figure.dpi": 150,
        "figure.autolayout": True,
        "font.family": "serif",
        "font.serif": [font],
        "font.size": base_fontsize,

        # Axes & text colors
        "axes.facecolor": theme["axes_facecolor"],
        "figure.facecolor": theme["figure_facecolor"],
        "axes.edgecolor": theme["axes_edgecolor"],
        "axes.labelcolor": theme["text_color"],
        "axes.titlecolor": theme["text_color"],   # <-- title color
        "text.color": theme["text_color"],        # <-- misc text (incl. suptitle)
        "xtick.color": theme["tick_color"],
        "ytick.color": theme["tick_color"],
        "grid.color": theme["grid_color"],

        # Sizes
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,

        # Lines & cycle
        "axes.prop_cycle": cycler(color=colors),
        "lines.linewidth": linewidth,
        "lines.markersize": 5,

        # Grid
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.3,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": base_fontsize - 1,
        "legend.labelcolor": theme["text_color"], # <-- legend text color

        # Colormap
        "image.cmap": "plasma",

        # Save
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    plt.set_cmap("plasma")
