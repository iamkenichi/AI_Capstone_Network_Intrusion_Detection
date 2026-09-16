"""
Shared figure styling and plot helpers.

One house style, applied once, so every figure in ``figures/`` is visually
consistent: same typeface, same grid weight, same colour-blind-safe palette
(Okabe-Ito derived), no chart junk, and a caption line that states what the
reader should take away rather than restating the axis labels.

Only matplotlib is used. There is no seaborn dependency anywhere in this
project - the plots below are simple enough that the extra layer would add
install weight without adding clarity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import config

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def use_house_style() -> None:
    """Apply the project-wide matplotlib style. Idempotent."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": config.FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    })


def despine(ax: plt.Axes, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    """Remove chart-junk spines."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def caption(fig: plt.Figure, text: str, y: float = -0.02) -> None:
    """Add an interpretive caption beneath a figure."""
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8.5,
             color="#555555", wrap=True)


def save(fig: plt.Figure, name: str, close: bool = True) -> Path:
    """Save ``fig`` into ``figures/`` as ``<name>.png`` and return the path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"[plot] wrote figures/{path.name}")
    return path


# --------------------------------------------------------------------------- #
# Reusable primitives
# --------------------------------------------------------------------------- #
def annotate_bars(
    ax: plt.Axes,
    bars,
    labels: list[str],
    *,
    horizontal: bool = False,
    pad: float = 0.01,
    fontsize: float = 8.5,
) -> None:
    """Write a value label at the end of each bar."""
    if horizontal:
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        for bar, text in zip(bars, labels):
            ax.text(bar.get_width() + span * pad,
                    bar.get_y() + bar.get_height() / 2,
                    text, va="center", ha="left", fontsize=fontsize, color="#333333")
    else:
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for bar, text in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + span * pad,
                    text, ha="center", va="bottom", fontsize=fontsize, color="#333333")


def class_colors(n: int = 2) -> list[str]:
    """Benign/attack colour pair, or a longer categorical ramp."""
    if n <= 2:
        return [config.COLOR_BENIGN, config.COLOR_ATTACK]
    palette = list(config.CATEGORICAL_PALETTE)
    return [palette[i % len(palette)] for i in range(n)]


def rate_to_color(rate: float) -> str:
    """Colour a bar by attack rate: blue (benign-dominated) to red (malicious)."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "benign_attack", [config.COLOR_BENIGN, "#EEEEEE", config.COLOR_ATTACK]
    )
    return matplotlib.colors.to_hex(cmap(float(np.clip(rate, 0.0, 1.0))))
