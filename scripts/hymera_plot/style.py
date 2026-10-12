"""Shared matplotlib style for hymera plots.

Colours come from a fixed categorical order, assigned by series identity, never
cycled. The first three slots pass the colourblind-separation and lightness
checks of the dataviz palette validator on the light surface used here
(#fcfcfb). Aqua is below 3:1 contrast against that surface, so every series is
also named in a legend or a direct label, never by colour alone.

Sequential quantities (2-D field maps, phase-space histograms) use one hue,
light to dark ("Blues"); signed quantities use a two-hue diverging map with a
neutral midpoint ("RdBu_r"), centred on zero.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3dd"

# Fixed categorical order (slot 1 first). Do not extend by generating hues: a
# ninth series goes into small multiples instead.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"


def apply():
    """Set the rcParams every hymera plot uses."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "legend.frameon": False,
        "font.size": 10,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def series_color(index):
    """Colour of the series in slot `index` (0-based); fails past slot 8."""
    if index >= len(CATEGORICAL):
        raise ValueError("more than 8 series: split into small multiples")
    return CATEGORICAL[index]
