"""Shared plotting style so every figure in the project looks consistent."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import CLASS_NAMES  # noqa: E402

TEXT = "#333333"
MUTED = "#6b6a66"
GRID = "#e5e5e3"
NEUTRAL = "#a8a7a2"
PRIMARY = "#2a78d6"

# Categorical slots from a colour-vision-deficiency validated palette.
# "No Failure" is neutral grey so the four failure modes stand out.
CLASS_COLORS = {
    "No Failure": NEUTRAL,
    "TWF": "#2a78d6",
    "HDF": "#eb6834",
    "PWF": "#1baf7a",
    "OSF": "#eda100",
}
assert set(CLASS_COLORS) == set(CLASS_NAMES)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXT,
        "axes.edgecolor": "#8c8b87",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "legend.frameon": False,
    }
)


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
