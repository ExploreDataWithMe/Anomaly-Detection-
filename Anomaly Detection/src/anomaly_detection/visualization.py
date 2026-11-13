"""Publication-quality visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def configure_plot_style() -> None:
    """Apply a clean, high-resolution house style for portfolio figures."""

    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "font.size": 11,
            "figure.figsize": (10, 7),
        }
    )


def save_figure(fig: plt.Figure, output_path: str, *, transparent: bool = False) -> None:
    """Save a figure as a vector-friendly PDF and a high-resolution PNG."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300, transparent=transparent)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent)
