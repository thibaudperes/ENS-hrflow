"""Matplotlib style, palette by role and figure helpers shared by every README figure."""

from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

FIG_DIR = Path("reports/figures")

PALETTE: dict[str, str] = {
    # Results
    "final": "#12b886",  # green: the submitted model, or the validation I trust
    "alternative": "#5b4bc4",  # violet: the learned ranker, compared with the final model
    "leak": "#d9480f",  # orange: anything that uses or suffers from the fragment leak
    # Context
    "neutral": "#a3a29b",
    "neutral_light": "#e6e5df",
    # Ink
    "ink": "#1a1a19",
    "ink_secondary": "#5f5e5a",
    "muted": "#8a8983",
    "grid": "#ebeae5",
    "axis": "#c3c2b7",
    "background": "#ffffff",
}

RESULT_ROLES: tuple[str, ...] = ("final", "alternative", "leak")

MIN_DELTA_E = 12.0


def set_style() -> None:
    p = PALETTE
    mpl.rcParams.update(
        {
            # White background: the PNG stays readable in GitHub dark mode.
            "figure.facecolor": p["background"],
            "axes.facecolor": p["background"],
            "savefig.facecolor": p["background"],
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.constrained_layout.use": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlecolor": p["ink_secondary"],
            "axes.titlepad": 8,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": p["ink_secondary"],
            "axes.edgecolor": p["axis"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": p["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": p["muted"],
            "ytick.color": p["muted"],
            "xtick.labelcolor": p["ink_secondary"],
            "ytick.labelcolor": p["ink_secondary"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": p["ink"],
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "axes.prop_cycle": mpl.cycler(color=[p[key] for key in RESULT_ROLES]),
            "patch.linewidth": 0,
        }
    )


def set_conclusion(fig: Figure, text: str) -> None:
    """Bold left-aligned figure title that states the conclusion; panel subjects go in grey."""
    fig.suptitle(text, x=0.0, ha="left")


def save_fig(fig: Figure, name: str, fig_dir: Path = FIG_DIR) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path)
    return path


# Colour-vision deficiency simulation (Machado, Oliveira and Fernandes 2009, severity 1.0),
# applied to linear RGB.
CVD_MATRICES = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}


def linear_rgb(hex_color):
    srgb = np.array(mpl.colors.to_rgb(hex_color))
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def oklab(linear):
    lms = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    ) @ np.clip(linear, 0, 1)
    return np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    ) @ np.cbrt(lms)


def palette_separation(colors: list[str]) -> dict[str, float]:
    """Smallest OKLab Delta E (x100) between two colours, per type of vision."""
    visions = {"normal": np.eye(3), **CVD_MATRICES}
    result = {}
    for vision, matrix in visions.items():
        points = [oklab(matrix @ linear_rgb(color)) for color in colors]
        result[vision] = float(min(100 * np.linalg.norm(a - b) for a, b in combinations(points, 2)))
    return result
