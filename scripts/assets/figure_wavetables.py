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
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch as tr
from torch import Tensor as T

# Ensure project root and code directory are in sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "code") not in sys.path:
    sys.path.insert(0, str(_repo_root / "code"))

from features import SpectralCentroid

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Styling constants
SERIES_COLOR = "#2a78d6"
AXIS_COLOR = "#52514e"

# The listening test wavetables, one row per wavetable in this order
LT_DIMS = ["warmth", "brightness", "richness"]
LT_VARIANTS = ["synthetic", "real"]

# Columns of the listening test overview: the wavetable and three of its curves
LT_FEATURES = ["Warmth", "Spectral Centroid", "Richness"]
LT_COL_TITLES = ["Wavetable", "Warmth", "Brightness", "Richness"]
LT_FIG_SIZE = (8.0, 10.5)
LT_DPI = 300
LT_FONT_SIZE = 12
LT_TICK_FONT_SIZE = 10


def compute_warmth_curve(frame_batch: T, eps: float = 1e-8) -> T:
    """Warmth = odd harmonic power ratio (excluding DC)."""
    fft = tr.fft.rfft(frame_batch)
    power = tr.abs(fft) ** 2

    odd_power = power[:, 1::2].sum(dim=1)
    total_power = power[:, 1:].sum(dim=1)

    warmth = odd_power / (total_power + eps)
    return warmth


def compute_richness_curve(frame_batch: T, eps: float = 1e-8) -> T:
    """Spectral spread mapped to richness score."""
    fft = tr.fft.rfft(frame_batch)
    mag = tr.abs(fft)
    power = mag**2

    freqs = tr.linspace(0, 1, power.shape[1], device=power.device)

    total = power.sum(dim=1, keepdim=True) + eps
    centroid = (power * freqs).sum(dim=1, keepdim=True) / total

    spread = tr.sqrt((power * (freqs - centroid) ** 2).sum(dim=1) / total.squeeze(1))

    k = 7.5
    richness = tr.log(spread * (tr.exp(tr.tensor(k)) - 1) + 1) / k
    return richness


def compute_features(wt: T, sr: int = 44100) -> Dict[str, T]:
    """Compute Warmth, Spectral Centroid (Brightness), and Richness curves."""
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
    fig_size: Tuple[float, float] = LT_FIG_SIZE,
    dpi: int = LT_DPI,
    show: bool = True,
) -> List[Tuple[str, Dict[str, Tuple[float, float]]]]:
    """One row per listening test wavetable, showing the wavetable itself
    followed by the curves of the three timbral dimensions. Returns the range
    of each of those curves, per wavetable."""
    wt_dir_str = str(wt_dir)
    rows = []
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
    ranges = []
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
        peak = np.abs(wt_np).max()
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
    ranges: List[Tuple[str, Dict[str, Tuple[float, float]]]],
) -> None:
    """Print a table of the range of each timbre feature, per wavetable."""
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


def resolve_file_path(path_str: str) -> Path:
    """Resolve file path relative to current working dir, repo root, or script location."""
    p = Path(path_str).expanduser()
    if p.exists():
        return p.resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = (repo_root / path_str).resolve()
    if candidate.exists():
        return candidate
    candidate_script = (Path(__file__).resolve().parent / path_str).resolve()
    if candidate_script.exists():
        return candidate_script
    return p.resolve()


def resolve_output_path(path_str: str) -> Path:
    """Resolve output file path properly relative to cwd, repo root, or script location."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(path_str).startswith("../../"):
        rel_stripped = str(path_str)[6:]
        candidate_repo = (repo_root / rel_stripped).resolve()
        if candidate_repo.parent.exists():
            return candidate_repo
    candidate_cwd = p.resolve()
    if candidate_cwd.parent.exists():
        return candidate_cwd
    candidate_script = (Path(__file__).resolve().parent / path_str).resolve()
    if candidate_script.parent.exists():
        return candidate_script
    return (repo_root / path_str).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figure of listening test wavetables and timbre feature curves."
    )
    parser.add_argument(
        "wt_dir",
        nargs="?",
        default="data/listening_test",
        help="Directory containing the listening test .pt wavetables (default: data/listening_test)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../../out/figure_wavetables.pdf",
        help="Path to save figure (default: out/figure_wavetables.pdf). Supports .pdf, .png, .svg, etc.",
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

    wt_dir_path = resolve_file_path(args.wt_dir)
    if not wt_dir_path.exists():
        sys.stderr.write(f"Error: Wavetable directory not found at: {wt_dir_path}\n")
        sys.exit(1)

    out_path = resolve_output_path(args.output) if args.output else None

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
