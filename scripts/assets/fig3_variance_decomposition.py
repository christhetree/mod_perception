"""Generate publication-ready horizontal bar graph of ANOVA variance decompositions.

Reads data/anova_variance_results_pooled_timbre_source.tsv, data/anova_variance_results_pooled_source.tsv,
or data/anova_variance_results.tsv and generates a stacked bar chart decomposing total variance
across factorial components (Modulation Amount, Modulation Type, Timbre Quality, Wavetable Source,
2-Way Interactions, and Higher-Order Residual), with distinct harmonious colors for each component.

Dynamically adapts active components and legend depending on whether factors were pooled.
When only 2 factors remain without explicit interaction terms, the residual variance represents
the confounded 2-way interaction and is labeled '2-Way Inter.' with the matching interaction color.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from paths import OUT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# ------------------------------------------------------------------------------
# Figure layout, dimensions, typography, and styling constants
# ------------------------------------------------------------------------------
BAR_SIZE: float = 0.7
BAR_SPACING: float = 0.0
GROUP_SPACING: float = 0.2
HUMAN_SPACING: float = 0.2
BOTTOM_SPACING: float = 0.05
LEGEND_Y: float = 1.0
LEGEND_LOC: str = "lower right"
LEGEND_X: Optional[float] = None
LEGEND_COLUMNSPACING: Optional[float] = None
COMPONENT_SPACING: float = 2.0
SHOW_GROUP_LINES: bool = True
LABEL_THRESHOLD: float = 5.5

FONT_SIZE: float = 16.0
FONTSIZE_LABELS: float = FONT_SIZE
FONTSIZE_PERCENTAGES: float = FONT_SIZE - 4.0
FONTSIZE_AXIS_LABEL: Optional[float] = None
FONTSIZE_TICKS: float = FONT_SIZE - 4.0
FONTSIZE_LEGEND: float = FONT_SIZE - 3.0

# Method ordering matching correlation and ANOVA tables
ENTITIES = [
    ("human", "Human", "human"),
    ("mss_log_lin", "MSS L+L", "group1"),
    ("mss_rev", "MSS Rev.", "group1"),
    ("mfcc", "MFCC", "group1"),
    ("scat1d_log1p", "Scat1D", "group2"),
    ("jtfs_log1p", "JTFS", "group2"),
    ("vggish", "VGGish", "group3"),
    ("encodec48k", "EnCodec", "group3"),
    ("clap2", "MS-CLAP", "group3"),
    ("panns_wavegram_logmel", "PANNs", "group3"),
]

LOSS_FN_ALIASES = {
    "scat1d": "scat1d_log1p",
    "jtfs": "jtfs_log1p",
    "encodec": "encodec48k",
    "encodec48": "encodec48k",
    "encodec24k": "encodec48k",
    "clap": "clap2",
    "panns_wglm": "panns_wavegram_logmel",
}

# The canonical variance components, display labels, and distinct harmonious colors
COMPONENTS = [
    ("distance", "Modulation Amount", "#082a54"),
    ("mod", "Modulation Type", "#d9534f"),
    ("feat", "Timbre Quality", "#3399a1"),
    ("source", "Wavetable Source", "#f39c12"),
    ("two_way", "2-Way Interactions", "#a559aa"),
    ("higher_order", "Higher Order", "#bbbbbb"),
]


def load_variance_data(
    tsv_path: Path,
) -> tuple[list[str], dict[str, list[float]], list[str], list[tuple[str, str, str]]]:
    """Load and organize variance components for each entity from TSV.

    Dynamically detects active factors and interactions, omitting pooled factors.
    In an unreplicated 2-factor design, the residual variance represents the 2-way
    interaction and is categorized as '2-Way Inter.'.

    Parameters
    ----------
    tsv_path : Path
        Path to ANOVA variance results TSV dataset.

    Returns
    -------
    tuple[list[str], dict[str, list[float]], list[str], list[tuple[str, str, str]]]
        Tuple of (labels, comp_values, groups, active_components).
    """
    sep = "\t" if tsv_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(tsv_path, sep=sep)

    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    present_sources = set(df["Source"].dropna().unique())
    has_two_way = any(k.count(":") == 1 for k in present_sources)
    has_three_way = any(k.count(":") == 2 for k in present_sources)
    has_residual = any("Residual" in k for k in present_sources)

    # Determine potential active components dynamically based on input sources
    candidate_components: list[tuple[str, str, str]] = []
    if "rating_stimulus" in present_sources:
        candidate_components.append(("distance", "Modulation Amount", "#082a54"))
    if "modulation" in present_sources:
        candidate_components.append(("mod", "Modulation Type", "#d9534f"))
    if "feature" in present_sources:
        candidate_components.append(("feat", "Timbre Quality", "#3399a1"))
    if "source" in present_sources:
        candidate_components.append(("source", "Wavetable Source", "#f39c12"))

    # If explicit 2-way terms exist OR if residual in 2-factor design represents the 2-way interaction
    residual_is_two_way = has_residual and not has_two_way and not has_three_way
    if has_two_way or residual_is_two_way:
        candidate_components.append(("two_way", "2-Way Interactions", "#a559aa"))

    if has_three_way:
        candidate_components.append(("three_way", "3-Way Interactions", "#9b59b6"))

    if has_residual and not residual_is_two_way:
        candidate_components.append(("higher_order", "Higher Order", "#bbbbbb"))

    # Fallback to standard 6 components if no recognized sources
    if not candidate_components:
        candidate_components = list(COMPONENTS)

    labels: list[str] = []
    groups: list[str] = []
    comp_values: dict[str, list[float]] = {k: [] for k, _, _ in candidate_components}

    for canon_key, display_name, grp in ENTITIES:
        match = df[df["canonical_loss"] == canon_key]
        if match.empty:
            match = df[df["loss_fn"] == canon_key]
        if match.empty:
            continue

        labels.append(display_name)
        groups.append(grp)

        t_map = dict(zip(match["Source"], match["pct_var"]))
        two_way_pct = sum(v for k, v in t_map.items() if k.count(":") == 1)
        three_way_pct = sum(v for k, v in t_map.items() if k.count(":") == 2)
        residual_pct = next((v for k, v in t_map.items() if "Residual" in k), 0.0)

        # In an unreplicated 2-factor design, residual is mathematically the 2-way interaction
        if residual_is_two_way:
            two_way_pct = residual_pct

        for key, _, _ in candidate_components:
            if key == "distance":
                comp_values["distance"].append(float(t_map.get("rating_stimulus", 0.0)))
            elif key == "mod":
                comp_values["mod"].append(float(t_map.get("modulation", 0.0)))
            elif key == "feat":
                comp_values["feat"].append(float(t_map.get("feature", 0.0)))
            elif key == "source":
                comp_values["source"].append(float(t_map.get("source", 0.0)))
            elif key == "two_way":
                comp_values["two_way"].append(float(two_way_pct))
            elif key == "three_way":
                comp_values["three_way"].append(float(three_way_pct))
            elif key == "higher_order":
                comp_values["higher_order"].append(float(residual_pct))

    # Keep only components that have non-zero variance in at least one entity
    active_components = [
        c for c in candidate_components if any(abs(v) > 1e-4 for v in comp_values[c[0]])
    ]

    return labels, comp_values, groups, active_components


def compute_positions(
    groups: list[str],
    bar_size: float = BAR_SIZE,
    bar_spacing: float = BAR_SPACING,
    group_spacing: float = GROUP_SPACING,
    human_spacing: Optional[float] = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Compute center coordinates for each bar and separator positions between groups.

    Parameters
    ----------
    groups : list[str]
        Group identifiers for each entity.
    bar_size : float, default=BAR_SIZE
        Thickness of each bar.
    bar_spacing : float, default=BAR_SPACING
        Whitespace gap between consecutive loss function bars.
    group_spacing : float, default=GROUP_SPACING
        Additional whitespace gap between loss function groups.
    human_spacing : Optional[float], default=None
        Additional whitespace gap between human data and model representations
        (defaults to group_spacing if not specified).

    Returns
    -------
    positions : np.ndarray
        Coordinate array for bar centers.
    separator_pos : float
        Coordinate midway between Human Listeners and the first model.
    group_separators : list[float]
        List of coordinates midway between all adjacent distinct groups.
    """
    effective_human_spacing = (
        human_spacing if human_spacing is not None else group_spacing
    )
    positions = []
    current_pos = 0.0
    step = bar_size + bar_spacing

    for i, grp in enumerate(groups):
        if i == 0:
            positions.append(0.0)
            current_pos = 0.0
        else:
            delta = step
            if groups[i - 1] == "human" or grp == "human":
                delta += effective_human_spacing
            elif grp != groups[i - 1]:
                delta += group_spacing
            current_pos += delta
            positions.append(current_pos)

    # Compute separator positions between all adjacent distinct groups
    group_separators: list[float] = []
    separator_pos = 0.0
    for i in range(len(groups) - 1):
        if groups[i] != groups[i + 1]:
            sep = (positions[i] + positions[i + 1]) / 2.0
            group_separators.append(sep)
            if groups[i] == "human" or groups[i + 1] == "human":
                separator_pos = sep

    if separator_pos == 0.0 and len(positions) > 1:
        separator_pos = (positions[0] + positions[1]) / 2.0

    return np.array(positions), separator_pos, group_separators


def reorder_legend_left_to_right(
    handles: Sequence,
    labels: Sequence[str],
    ncols: int,
) -> tuple[list, list[str]]:
    """Reorder legend handles and labels so they are read left-to-right, row-by-row.

    By default, matplotlib populates multi-column legends column-by-column (top-to-bottom).
    This rearranges items so that when matplotlib places them into columns, the resulting
    visual layout reads row-by-row from left to right.
    """
    n = len(handles)
    if ncols <= 1 or n <= 1:
        return list(handles), list(labels)

    splits = np.array_split(range(n), ncols)
    col_lens = [len(s) for s in splits]
    nrows = max(col_lens)
    grid = [[None] * ncols for _ in range(nrows)]

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if r < col_lens[c]:
                grid[r][c] = (handles[idx], labels[idx])
                idx += 1

    reordered_handles = []
    reordered_labels = []
    for c in range(ncols):
        for r in range(col_lens[c]):
            h, l = grid[r][c]
            reordered_handles.append(h)
            reordered_labels.append(l)

    return reordered_handles, reordered_labels


def plot_variance(
    labels: list[str],
    comp_values: dict[str, list[float]],
    groups: list[str],
    components: Optional[list[tuple[str, str, str]]] = None,
    output_path: Optional[Path | str] = None,
    bar_size: float = BAR_SIZE,
    bar_spacing: float = BAR_SPACING,
    group_spacing: float = GROUP_SPACING,
    human_spacing: Optional[float] = HUMAN_SPACING,
    bottom_spacing: Optional[float] = BOTTOM_SPACING,
    legend_y: Optional[float] = LEGEND_Y,
    legend_x: Optional[float] = LEGEND_X,
    legend_loc: Optional[str] = LEGEND_LOC,
    legend_rows: int = 1,
    legend_ncol: Optional[int] = None,
    legend_columnspacing: Optional[float] = LEGEND_COLUMNSPACING,
    component_spacing: float = COMPONENT_SPACING,
    show_labels: bool = True,
    show_group_lines: bool = SHOW_GROUP_LINES,
    label_threshold: float = LABEL_THRESHOLD,
    fontsize_labels: float = FONTSIZE_LABELS,
    fontsize_percentages: float = FONTSIZE_PERCENTAGES,
    fontsize_axis_label: Optional[float] = FONTSIZE_AXIS_LABEL,
    fontsize_ticks: Optional[float] = FONTSIZE_TICKS,
    fontsize_legend: Optional[float] = FONTSIZE_LEGEND,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Create a publication-ready horizontal stacked bar chart of ANOVA variance decompositions.

    Parameters
    ----------
    labels : list[str]
        List of entity/method display names.
    comp_values : dict[str, list[float]]
        Dictionary mapping component keys to percentage variance values per entity.
    groups : list[str]
        Group labels for each entity (e.g. 'human', 'group1', 'group2', 'group3').
    components : Optional[list[tuple[str, str, str]]], default=None
        List of active components (key, display_label, color).
    output_path : Optional[Path or str], default=None
        Path to save figure (.pdf, .png, .svg). Parent directories are created if missing.
    bar_size : float, default=0.7
        Thickness of individual bars.
    bar_spacing : float, default=0.0
        Whitespace gap between adjacent loss function bars.
    group_spacing : float, default=0.2
        Additional whitespace gap between loss function groups.
    human_spacing : Optional[float], default=0.2
        Additional whitespace gap between human data and model representations.
    bottom_spacing : Optional[float], default=0.05
        Whitespace padding between the lowest bar and the x-axis.
    legend_y : Optional[float], default=1.0
        Vertical position/offset of legend above the graph.
    legend_x : Optional[float], default=None
        Horizontal position/offset of legend.
    legend_loc : Optional[str], default='lower right'
        Legend placement anchor location.
    legend_rows : int, default=1
        Number of lines/rows to split the legend across.
    legend_ncol : Optional[int], default=None
        Number of columns for the legend (overrides legend_rows).
    legend_columnspacing : Optional[float], default=None
        Spacing between columns in the legend.
    component_spacing : float, default=2.0
        Width of separator line between adjacent components of the same bar.
    show_labels : bool, default=True
        Whether to display numerical percentage labels inside bar segments.
    show_group_lines : bool, default=True
        Whether to draw dashed separator lines between representation groups.
    label_threshold : float, default=5.5
        Minimum percentage required to render label inside a bar segment.
    fontsize_labels : float, default=16.0
        Font size for entity/method labels.
    fontsize_percentages : float, default=12.0
        Font size for percentage text inside bar segments.
    fontsize_axis_label : Optional[float], default=None
        Optional override for 'Explained Variance (%)' axis label font size.
    fontsize_ticks : Optional[float], default=12.0
        Font size for numeric tick values on the x-axis.
    fontsize_legend : Optional[float], default=13.0
        Font size for legend labels.
    dpi : int, default=300
        Resolution in dots per inch for image export.
    show : bool, default=True
        Whether to display the plot interactively.

    Returns
    -------
    plt.Figure
        The generated Matplotlib Figure.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9.0,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )

    comps_to_plot = components if components is not None else COMPONENTS
    eff_axis_label = (
        fontsize_axis_label if fontsize_axis_label is not None else fontsize_labels
    )
    eff_ticks = (
        fontsize_ticks if fontsize_ticks is not None else (fontsize_labels - 4.0)
    )
    eff_legend = (
        fontsize_legend if fontsize_legend is not None else (fontsize_labels - 3.0)
    )

    y_pos, separator_pos, group_separators = compute_positions(
        groups=groups,
        bar_size=bar_size,
        bar_spacing=bar_spacing,
        group_spacing=group_spacing,
        human_spacing=human_spacing,
    )

    n_bars = len(labels)
    total_span = y_pos[-1] - y_pos[0]
    fig_height = max(5.5, 6.0 * (total_span / 9.0))
    fig, ax = plt.subplots(figsize=(7.07, fig_height), dpi=dpi)

    cum_left = np.zeros(n_bars)

    for key, name, color in comps_to_plot:
        vals = np.array(comp_values[key])
        rects = ax.barh(
            y_pos,
            vals,
            left=cum_left,
            height=bar_size,
            color=color,
            edgecolor="white" if component_spacing > 0 else "none",
            linewidth=component_spacing,
            label=name,
        )

        if show_labels:
            for i, (val, rect) in enumerate(zip(vals, rects)):
                if val >= label_threshold:
                    x_center = cum_left[i] + val / 2.0
                    y_center = rect.get_y() + rect.get_height() / 2.0
                    val_str = f"{val:.0f}%"
                    ax.text(
                        x_center,
                        y_center,
                        val_str,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=fontsize_percentages,
                        fontweight="bold",
                    )

        cum_left += vals

    # Format y-axis (invert so Human Listeners is at the top)
    bot_pad = bottom_spacing if bottom_spacing is not None else BOTTOM_SPACING
    if bot_pad <= 0.3:
        bot_pad = 0.5 + bot_pad

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=fontsize_labels, fontweight="bold")
    ax.tick_params(axis="y", labelsize=fontsize_labels)
    ax.set_ylim(y_pos[-1] + bar_size * bot_pad, y_pos[0] - bar_size * 0.60)

    # Visual separator dashed lines between groups (Human, STFT, Wavelet, Neural)
    if show_group_lines:
        for sep in group_separators:
            ax.axhline(sep, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)

    # Format x-axis
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel(
        "Explained Variance (%)", fontsize=eff_axis_label, fontweight="bold", labelpad=8
    )
    ax.tick_params(axis="x", labelsize=eff_ticks)
    ax.grid(axis="x", linestyle=":", color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend at the top split across lines (default: right-aligned up to 100% tick marker)
    if legend_ncol is not None:
        ncol = legend_ncol
    elif legend_rows is not None and legend_rows > 0:
        ncol = int(np.ceil(len(comps_to_plot) / legend_rows))
    else:
        ncol = len(comps_to_plot)

    effective_legend_loc = legend_loc if legend_loc is not None else LEGEND_LOC
    effective_legend_y = legend_y if legend_y is not None else LEGEND_Y
    if effective_legend_y <= 0.5:
        effective_legend_y = 1.0 + effective_legend_y
    effective_legend_x = (
        legend_x
        if legend_x is not None
        else (1.0 if "right" in effective_legend_loc else 0.0)
    )
    col_spacing = legend_columnspacing if legend_columnspacing is not None else 1.2

    handles, leg_labels = ax.get_legend_handles_labels()
    handles, leg_labels = reorder_legend_left_to_right(handles, leg_labels, ncols=ncol)

    ax.legend(
        handles,
        leg_labels,
        loc=effective_legend_loc,
        bbox_to_anchor=(effective_legend_x, effective_legend_y),
        ncol=ncol,
        frameon=False,
        fontsize=eff_legend,
        columnspacing=col_spacing,
        handlelength=1.2,
        handleheight=0.9,
        borderaxespad=0.0,
    )

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        log.info(f"Figure successfully saved to: {out_p}")

    if show:
        plt.show()

    return fig


plot_variance_horizontal = plot_variance


def main() -> None:
    """Parse command line arguments and generate horizontal variance figure."""
    parser = argparse.ArgumentParser(
        description="Generate stacked bar chart of ANOVA variance decompositions."
    )
    parser.add_argument(
        "input",
        nargs="?",
        # default=os.path.join(OUT_DIR, "variance_decomposition_pooled.tsv"),
        default=os.path.join(OUT_DIR, "variance_decomposition.tsv"),
        help=f"Path to ANOVA variance results TSV dataset (default: {OUT_DIR}/variance_decomposition.tsv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        # default=os.path.join(OUT_DIR, "figures", "fig3_variance_decomposition_pooled.pdf"),
        default=os.path.join(OUT_DIR, "figures", "fig3_variance_decomposition.pdf"),
        help=f"Path to save figure image (default: {OUT_DIR}/figures/fig3_variance_decomposition.pdf). Supports .png, .pdf, .svg.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not display numerical percentage labels on bar segments.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LABEL_THRESHOLD,
        help=f"Minimum percentage required to render label inside a bar segment (default: {LABEL_THRESHOLD}).",
    )
    parser.add_argument(
        "--legend-rows",
        type=int,
        # default=1,
        default=2,
        dest="legend_rows",
        help="Number of lines/rows to split the legend across (default: 1).",
    )
    parser.add_argument(
        "--legend-cols",
        "--legend-ncol",
        type=int,
        default=None,
        dest="legend_cols",
        help="Number of columns for the legend (overrides --legend-rows).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch for raster export (default: 300).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot interactively in a window (only save to file).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"ANOVA variance results file not found at: {input_path}")
        sys.exit(1)

    labels, comp_values, groups, active_components = load_variance_data(input_path)

    out_path = Path(args.output) if args.output else None

    plot_variance(
        labels=labels,
        comp_values=comp_values,
        groups=groups,
        components=active_components,
        output_path=out_path,
        bar_size=BAR_SIZE,
        bar_spacing=BAR_SPACING,
        group_spacing=GROUP_SPACING,
        human_spacing=HUMAN_SPACING,
        bottom_spacing=BOTTOM_SPACING,
        legend_y=LEGEND_Y,
        legend_x=LEGEND_X,
        legend_loc=LEGEND_LOC,
        legend_rows=args.legend_rows,
        legend_ncol=args.legend_cols,
        legend_columnspacing=LEGEND_COLUMNSPACING,
        component_spacing=COMPONENT_SPACING,
        show_labels=not args.no_labels,
        show_group_lines=SHOW_GROUP_LINES,
        label_threshold=args.threshold,
        fontsize_labels=FONTSIZE_LABELS,
        fontsize_percentages=FONTSIZE_PERCENTAGES,
        fontsize_axis_label=FONTSIZE_AXIS_LABEL,
        fontsize_ticks=FONTSIZE_TICKS,
        fontsize_legend=FONTSIZE_LEGEND,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
