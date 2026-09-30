"""Generate publication-ready figure of listening test wavetables and timbre feature curves.

Loads the six listening test wavetables (Warmth, Brightness, Richness x Synthetic, Real)
and plots a 6x4 grid where each row shows:
1. Wavetable surface plot (imshow over normalized phase and position)
2. Warmth curve
3. Brightness (Spectral Centroid) curve
4. Richness curve
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from torch import Tensor as T

from features import SpectralCentroid, compute_richness_curve, compute_warmth_curve
from paths import DATA_DIR, OUT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# ------------------------------------------------------------------------------
# Figure layout, dimensions, typography, and styling constants
# ------------------------------------------------------------------------------
SERIES_COLOR: str = "#2a78d6"
AXIS_COLOR: str = "#52514e"

# The listening test wavetables, one row per wavetable in this order
LT_DIMS: list[str] = ["warmth", "brightness", "richness"]
LT_VARIANTS: list[str] = ["synthetic", "real"]

# Columns of the listening test overview: the wavetable and three of its curves
LT_FEATURES: list[str] = ["Warmth", "Spectral Centroid", "Richness"]
LT_COL_TITLES: list[str] = ["Wavetable", "Warmth", "Brightness", "Richness"]
LT_FIG_SIZE: tuple[float, float] = (8.0, 10.5)
LT_DPI: int = 300
LT_FONT_SIZE: float = 12.0
LT_TICK_FONT_SIZE: float = 10.0


def compute_features(wt: T, sr: int = 44100) -> dict[str, T]:
    """Compute Warmth, Spectral Centroid (Brightness), and Richness curves for a wavetable.

    Parameters
    ----------
    wt : torch.Tensor
        Wavetable tensor of shape (num_frames, frame_size).
    sr : int, default=44100
        Sample rate in Hz used for spectral centroid computation.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping feature names ('Warmth', 'Spectral Centroid', 'Richness')
        to their computed 1D curve tensors across wavetable positions.
    """
    centroid_metric = SpectralCentroid(
        sr, window="flat_top", compress=True, floor=1e-4, scaling="kazazis"
    )

    return {
        "Warmth": compute_warmth_curve(wt),
        "Spectral Centroid": centroid_metric(wt),
        "Richness": compute_richness_curve(wt),
    }


def plot_listening_test_wavetables(
    wt_dir: Union[str, Path],
    sr: int = 44100,
    output_path: Optional[Union[str, Path]] = None,
    fig_size: tuple[float, float] = LT_FIG_SIZE,
    dpi: int = LT_DPI,
    show: bool = True,
) -> list[tuple[str, dict[str, tuple[float, float]]]]:
    """Plot 6x4 grid of listening test wavetables and their timbre feature curves.

    Each row represents one wavetable (Warmth Synthetic/Real, Brightness Synthetic/Real,
    Richness Synthetic/Real) and displays:
    1. Wavetable surface imshow across normalized phase and position.
    2. Warmth curve.
    3. Brightness (Spectral Centroid) curve.
    4. Richness curve.

    Parameters
    ----------
    wt_dir : Union[str, Path]
        Directory containing the listening test `.pt` wavetable files.
    sr : int, default=44100
        Sample rate in Hz for feature extraction.
    output_path : Optional[Union[str, Path]], default=None
        Path to save figure (.pdf, .png, .svg). Parent directories are created if missing.
    fig_size : tuple[float, float], default=(8.0, 10.5)
        Figure width and height in inches.
    dpi : int, default=300
        Resolution in dots per inch for image export.
    show : bool, default=True
        Whether to display the plot interactively.

    Returns
    -------
    list[tuple[str, dict[str, tuple[float, float]]]]
        List of tuples containing (row_name, {feature_name: (min_val, max_val)})
        for each wavetable.

    Raises
    ------
    FileNotFoundError
        If the specified wavetable directory does not exist.
    AssertionError
        If any expected wavetable file is missing or ambiguous.
    """
    wt_dir_path = Path(wt_dir)
    if not wt_dir_path.exists():
        raise FileNotFoundError(f"Wavetable directory not found at: {wt_dir_path}")

    wt_dir_str = str(wt_dir_path)
    rows: list[tuple[str, str]] = []
    for dim in LT_DIMS:
        for variant in LT_VARIANTS:
            paths = sorted(glob.glob(os.path.join(wt_dir_str, f"{dim}_{variant}*.pt")))
            assert len(paths) == 1, f"Expected one {dim} {variant} wt, found {paths}"
            rows.append((f"{dim.capitalize()} ({variant.capitalize()})", paths[0]))

    fig, axs = plt.subplots(
        len(rows),
        len(LT_COL_TITLES),
        figsize=fig_size,
        sharex="all",
        sharey="col",
        squeeze=False,
        layout="constrained",
    )
    ranges: list[tuple[str, dict[str, tuple[float, float]]]] = []
    for row_idx, (row_name, wt_path) in enumerate(rows):
        wt = tr.load(wt_path, weights_only=True)
        log.info(f"{row_name}: {os.path.basename(wt_path)}, wt.shape: {wt.shape}")
        features = compute_features(wt, sr=sr)
        ranges.append(
            (
                row_name,
                {
                    feat_name: (
                        float(features[feat_name].min()),
                        float(features[feat_name].max()),
                    )
                    for feat_name in LT_FEATURES
                },
            )
        )

        wt_np = wt.detach().cpu().numpy()
        peak = float(np.abs(wt_np).max())
        ax = axs[row_idx][0]
        # Frames along x so that every plot in the row shares the position axis
        ax.imshow(
            wt_np.T,
            aspect="auto",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="coolwarm",
            vmin=-peak,
            vmax=peak,
        )
        # Two lines so the label fits the height of a square plot
        ax.set_ylabel(row_name.replace(" (", "\n("), fontsize=LT_FONT_SIZE)
        # The y axis of a wavetable plot is the phase within each frame
        ax.set_yticks([])

        for col_idx, feat_name in enumerate(LT_FEATURES, start=1):
            vals = features[feat_name].cpu().numpy()
            pos = np.linspace(0.0, 1.0, len(vals))
            ax = axs[row_idx][col_idx]
            ax.plot(pos, vals, color=SERIES_COLOR, linewidth=1.5)
            ax.grid(True, color=AXIS_COLOR, alpha=0.15, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axs.flat:
        ax.set_box_aspect(1.0)  # Square plotting area, not a square figure
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=LT_TICK_FONT_SIZE)
    for col_idx, title in enumerate(LT_COL_TITLES):
        axs[0][col_idx].set_title(title, fontsize=LT_FONT_SIZE)
    fig.supxlabel("Wavetable position", fontsize=LT_FONT_SIZE)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    if output_path:
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi)
        log.info(f"Saved figure to: {out_p}")

    if show:
        plt.show()

    plt.close(fig)
    return ranges


def print_feature_ranges(
    ranges: Sequence[tuple[str, dict[str, tuple[float, float]]]],
) -> None:
    """Print a formatted console table showing min and max values for each timbre feature per wavetable.

    Parameters
    ----------
    ranges : Sequence[tuple[str, dict[str, tuple[float, float]]]]
        List of wavetable feature ranges as returned by `plot_listening_test_wavetables`.
    """
    name_w = max(len(name) for name, _ in ranges)
    header = f"{'Wavetable':<{name_w}}" + "".join(
        f"  {title:^17}" for title in LT_COL_TITLES[1:]
    )
    subheader = f"{'':<{name_w}}" + "".join(
        f"  {'min':>8}{'max':>9}" for _ in LT_FEATURES
    )
    print(header.rstrip())
    print(subheader.rstrip())
    print("-" * len(header))
    for name, feat_ranges in ranges:
        row = f"{name:<{name_w}}"
        for feat_name in LT_FEATURES:
            lo, hi = feat_ranges[feat_name]
            row += f"  {lo:>8.3f}{hi:>9.3f}"
        print(row)


def main() -> None:
    """Parse command line arguments and generate the wavetable overview figure."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figure of listening test wavetables and timbre feature curves."
    )
    parser.add_argument(
        "wt_dir",
        nargs="?",
        default=os.path.join(DATA_DIR, "wavetables"),
        help=f"Directory containing the listening test .pt wavetables (default: {DATA_DIR}/wavetables).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(OUT_DIR, "figures", "wavetables.svg"),
        help=f"Path to save figure (default: {OUT_DIR}/figures/wavetables.svg). Supports .pdf, .png, .svg, etc.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate in Hz for feature extraction (default: 44100)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch (default: 300)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot interactively in a window (only save to file).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress printing feature ranges to console.",
    )

    args = parser.parse_args()

    wt_dir_path = Path(args.wt_dir)
    if not wt_dir_path.exists():
        sys.stderr.write(f"Error: Wavetable directory not found at: {wt_dir_path}\n")
        sys.exit(1)

    out_path = Path(args.output) if args.output else None

    ranges = plot_listening_test_wavetables(
        wt_dir=wt_dir_path,
        sr=args.sr,
        output_path=out_path,
        dpi=args.dpi,
        show=not args.no_show,
    )

    if not args.quiet:
        print_feature_ranges(ranges)


if __name__ == "__main__":
    main()
