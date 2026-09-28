"""Generate publication-ready figure of loss function linear fits across modulations.

Plots the 3-panel layout (Amplitude, Frequency (Hz), Irregularity (%)) for loss functions
arranged one per row. Subplots maintain an exact 1:1 aspect ratio. X-axis tick labels
and the shared 'Modulation amount' label appear only once at the very bottom row,
with column titles displayed at the top. Reference points are marked with squares,
subsequent points with circles.

Plotting modes (--plot and --loss-fn):
- 'losses' (default): individual loss functions (or with Human Listeners as row 0 if --human-data is given).
- 'groups': 3 rows of aggregated representation groups (STFT, Wavelet, Neural).
- 'all': combines all individual loss functions and the 3 aggregated groups.
- 'human' (or --human-only): plots exclusively the human listening test data row.
- --loss-fn / -l: filter and reorder analysis to specific loss function(s) (e.g. -l mfcc mss_log_lin, or -l human).

Human benchmark option (--human-data / --human):
- When provided (e.g. data/listening_test_responses_postprocessed.tsv), participant ratings
  are preserved across all participants to compute empirical range, std, and CI before averaging,
  graphed as the first row ('Human Listeners') with y-axis ticks [0, 25, 50, 75, 100].

Fit & Error Bar options:
- Linear line of best fit & R^2: toggleable via --no-fit / --hide-fit (default: shown).
- Anchor linear fit to y=0 at first x value: --anchor-zero (default: anchored).
- Connecting dots: --connect-dots / --connect-points (straight black line segments).
- Error bars:
  - 95% Confidence Intervals: --error-bars ci
  - Standard Deviation: --error-bars std (default)
  - No error bars: --error-bars none
- Error representation style:
  - --error-style shade: plot CI or STD as a semi-transparent shaded region.
  - --error-style bars (default): plot vertical error bars with caps.
  - --error-style both: plot both shaded region and vertical error bars.
  - --shade-range: also shade the [min, max] range with a lighter color.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

from paths import OUT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# ------------------------------------------------------------------------------
# Figure layout, dimensions, typography, and styling constants
# ------------------------------------------------------------------------------
RANGE_ALPHA: float = 0.07
SHADE_ALPHA: float = 0.16
PLOT_SIZE: float = 2.0
GAP_X: float = 0.08
GAP_Y: float = 0.08
GAP_HUMAN: float = 0.32
GAP_GROUP: float = 0.08

R2_DECIMALS: int = 2
LINE_COLOR: str = "#2a78d6"
MARKER_COLOR: str = "#111111"

FONT_SIZE: float = 16.0
FONTSIZE_XLABEL: float = FONT_SIZE
FONTSIZE_YLABEL: float = FONT_SIZE
FONTSIZE_R2: float = FONT_SIZE
FONTSIZE_TICKS: float = FONT_SIZE - 4.0
FONTSIZE_XTICKS: Optional[float] = None
FONTSIZE_YTICKS: Optional[float] = None
FONTSIZE_TITLE: float = FONT_SIZE

# Canonical individual loss functions: (identifier, display label, representation group)
INDIVIDUAL_LOSS_FUNCTIONS: list[tuple[str, str, str]] = [
    # STFT Group
    ("mss_log_lin", "MSS Log + Lin.", "STFT"),
    ("mss_rev", "MSS Revisited", "STFT"),
    ("mfcc", "MFCC", "STFT"),
    # Wavelet Group
    ("scat1d", "Scat1D", "Wavelet"),
    ("jtfs", "JTFS", "Wavelet"),
    # Neural Group
    ("vggish", "VGGish", "Neural"),
    ("encodec48k", "EnCodec 48 kHz", "Neural"),
    ("clap", "MS-CLAP", "Neural"),
    ("panns_wavegram_logmel", "PANNs WGLM", "Neural"),
]

# Aggregated representation groups and constituent loss functions
AGGREGATED_GROUPS: list[dict[str, Any]] = [
    {
        "key": "group_stft",
        "label": "STFT Group",
        "group": "STFT",
        "models": ["mss_log_lin", "mss_rev", "mfcc"],
    },
    {
        "key": "group_wavelet",
        "label": "Wavelet Group",
        "group": "Wavelet",
        "models": ["scat1d", "jtfs"],
    },
    {
        "key": "group_neural",
        "label": "Neural Group",
        "group": "Neural",
        "models": ["vggish", "encodec48k", "clap", "panns_wavegram_logmel"],
    },
]

# Modulation conditions and column configurations
MODULATION_COLUMNS: list[dict[str, Any]] = [
    {
        "key": "amp",
        "title": "Amplitude",
        "tick_labels": ["0.1", "0.3", "0.5", "0.7", "0.9"],
    },
    {
        "key": "freq",
        "title": "Frequency (Hz)",
        "tick_labels": ["0.25", "0.5", "1", "2", "4"],
    },
    {
        "key": "reg",
        "title": "Irregularity (%)",
        "tick_labels": ["0", "12.5", "25", "37.5", "50"],
    },
]

# Modulation parameter value mapping for human listening test stimuli
AMOUNT_MAP: dict[str, dict[str, float]] = {
    "amp": {
        "reference": 0.1,
        "condition_a": 0.3,
        "condition_b": 0.5,
        "condition_c": 0.7,
        "condition_d": 0.9,
    },
    "freq": {
        "reference": 0.25,
        "condition_a": 0.5,
        "condition_b": 1.0,
        "condition_c": 2.0,
        "condition_d": 4.0,
    },
    "reg": {
        "reference": 0.0,
        "condition_a": 0.125,
        "condition_b": 0.25,
        "condition_c": 0.375,
        "condition_d": 0.5,
    },
}


def load_distances_data(data_path: Path) -> pd.DataFrame:
    """Load distances dataset from a TSV or CSV file.

    Parameters
    ----------
    data_path : Path
        Path to the distances data file.

    Returns
    -------
    pd.DataFrame
        Loaded distance dataset.
    """
    sep = "\t" if data_path.suffix in [".tsv", ".txt"] else ","
    return pd.read_csv(data_path, sep=sep)


def load_human_data(data_path: Path) -> pd.DataFrame:
    """Load postprocessed human listening study responses as a benchmark.

    Retains all participant responses so that empirical range, standard deviation,
    and confidence intervals can be computed from the full participant distribution
    before being averaged into condition means.

    Parameters
    ----------
    data_path : Path
        Path to the postprocessed listening test responses TSV or CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame formatted with columns 'mod_type', 'feature', 'source',
        'distance', 'loss_fn' ('human'), 'amount', and 'is_reference'.
    """
    sep = "\t" if data_path.suffix in [".tsv", ".txt"] else ","
    df_human = pd.read_csv(data_path, sep=sep)

    split_cols = df_human["trial_id"].str.split("_", expand=True)
    df_human["mod_type"] = split_cols[0]
    df_human["feature"] = split_cols[1]
    df_human["source"] = split_cols[2]
    df_human["distance"] = df_human["rating_score"].astype(float)
    df_human["loss_fn"] = "human"

    df_human["amount"] = [
        AMOUNT_MAP[m][s]
        for m, s in zip(df_human["mod_type"], df_human["rating_stimulus"])
    ]
    df_human["is_reference"] = df_human["rating_stimulus"] == "reference"

    return df_human


def compute_point_stats(
    values: np.ndarray | Sequence[float],
    error_mode: str = "ci",
    ci_level: float = 0.95,
) -> tuple[float, Optional[float]]:
    """Compute mean and error bar half-width (Confidence Interval, STD, or None).

    Parameters
    ----------
    values : np.ndarray or Sequence[float]
        Array of observed numerical values.
    error_mode : str, default='ci'
        Metric for the error half-width: 'ci' (Student's t confidence interval),
        'std' (sample standard deviation), or 'none' / 'no' / 'off' (no error).
    ci_level : float, default=0.95
        Confidence level used when error_mode is 'ci'.

    Returns
    -------
    mean_val : float
        Sample arithmetic mean.
    err_val : Optional[float]
        Half-width error magnitude, or None if error_mode is disabled.

    Raises
    ------
    ValueError
        If an unrecognized error_mode string is provided.
    """
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    n = len(valid)
    if n == 0:
        return 0.0, None if error_mode in ("none", "no", "off") else 0.0

    mean_val = float(np.mean(valid))

    if error_mode in ("none", "no", "off"):
        return mean_val, None
    if n <= 1:
        return mean_val, 0.0

    s = float(np.std(valid, ddof=1))
    if error_mode == "std":
        return mean_val, s
    elif error_mode == "ci":
        if s == 0.0:
            return mean_val, 0.0
        se = s / np.sqrt(n)
        t_crit = float(stats.t.ppf((1.0 + ci_level) / 2.0, df=n - 1))
        return mean_val, t_crit * se
    else:
        raise ValueError(
            f"Unknown error_mode: {error_mode}. Choose 'ci', 'std', or 'none'."
        )


def compute_nice_5_ticks(v_max: float) -> tuple[list[float | int], float, float]:
    """Compute 5 clean, evenly spaced tickmarks [0, step, 2*step, 3*step, 4*step] for a given max value.

    Parameters
    ----------
    v_max : float
        Maximum observed value to accommodate on the axis.

    Returns
    -------
    ticks : list[float | int]
        List of 5 tick mark values starting at 0.
    y_lower : float
        Suggested lower axis bound (providing ~4% bottom headroom).
    y_upper : float
        Suggested upper axis bound (providing ~4% top headroom).
    """
    if v_max <= 0:
        return [0, 1, 2, 3, 4], -0.16, 4.16

    raw_step = v_max / 4.0
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10**exponent)

    # Standard 1-2-5 and intermediate round step multipliers
    nice_steps = [1.0, 1.2, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    step_mult = min(s for s in nice_steps if s >= fraction - 1e-9)
    step = step_mult * (10**exponent)

    ticks: list[float | int] = [round(i * step, 10) for i in range(5)]
    y_upper = ticks[-1] * 1.04
    y_lower = -0.04 * ticks[-1]

    # Convert to int if all values are exact integers
    if all(abs(t - round(t)) < 1e-8 for t in ticks):
        ticks = [int(round(t)) for t in ticks]

    return ticks, y_lower, y_upper


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    anchor_zero: bool = True,
) -> tuple[float, float, float]:
    """Fit a linear regression line to data points with optional zero-anchoring.

    When anchor_zero is True, the fit is constrained such that y = 0 at x = x[0],
    i.e., y = slope * (x - x[0]).

    Parameters
    ----------
    x : np.ndarray
        Array of x values (must have length >= 2).
    y : np.ndarray
        Array of observed y values.
    anchor_zero : bool, default=True
        If True, constrain line to pass through (x[0], 0).
        If False, fit standard ordinary least squares linear regression.

    Returns
    -------
    slope : float
        Fitted line slope.
    intercept : float
        Fitted line y-intercept.
    r2 : float
        Coefficient of determination (R^2).
    """
    if anchor_zero:
        x0 = float(x[0])
        u = x - x0
        denom = float(np.sum(u**2))
        slope = float(np.sum(u * y) / denom) if denom > 0 else 0.0
        intercept = -slope * x0
        y_pred = slope * u
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        return slope, intercept, r2
    else:
        reg = linregress(x, y)
        slope = float(reg.slope)
        intercept = float(reg.intercept)
        r2 = float(reg.rvalue**2)
        return slope, intercept, r2


def normalize_loss_fns(raw_loss_fns: Sequence[str] | None) -> list[str] | None:
    """Clean and tokenize user-supplied loss function filter list.

    Splits comma-separated or space-separated entries and removes duplicates
    while preserving order.

    Parameters
    ----------
    raw_loss_fns : Sequence[str] or None
        Raw loss function names or list of strings passed from command line.

    Returns
    -------
    list[str] or None
        Cleaned, deduplicated list of loss function identifier strings, or None.
    """
    if not raw_loss_fns:
        return None
    cleaned: list[str] = []
    for item in raw_loss_fns:
        for term in item.replace(",", " ").split():
            c = term.strip()
            if c and c not in cleaned:
                cleaned.append(c)
    return cleaned if cleaned else None


def prepare_plot_items(
    plot_mode: str,
    df: pd.DataFrame,
    loss_fns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare row data specifications according to the chosen plot mode and loss function filter.

    Parameters
    ----------
    plot_mode : str
        Plotting mode: 'losses' (individual models), 'groups' (aggregated representation groups),
        'all' (individual and groups), or 'human' (human benchmark only).
    df : pd.DataFrame
        Dataset containing distances and/or human ratings.
    loss_fns : Sequence[str] or None, default=None
        Optional filter restricting and ordering the loss functions displayed.

    Returns
    -------
    list[dict[str, Any]]
        List of row specifications, each containing 'type', 'group', 'label', and 'sub_df'.
    """
    items: list[dict[str, Any]] = []
    has_human = "human" in df["loss_fn"].unique()

    # If plot_mode is "human", only plot the human benchmark row
    if plot_mode == "human":
        if has_human:
            items.append(
                {
                    "type": "human",
                    "group": "human",
                    "label": "Human Listeners",
                    "sub_df": df[df["loss_fn"] == "human"],
                }
            )
        return items

    # If human benchmark data is present, graph it as the first row
    if has_human:
        items.append(
            {
                "type": "human",
                "group": "human",
                "label": "Human Listeners",
                "sub_df": df[df["loss_fn"] == "human"],
            }
        )

    loss_label_map = {key: label for key, label, _ in INDIVIDUAL_LOSS_FUNCTIONS}
    loss_group_map = {key: grp for key, _, grp in INDIVIDUAL_LOSS_FUNCTIONS}

    if plot_mode in ("losses", "all"):
        if loss_fns is not None:
            # Respect user-specified selection and ordering of loss functions
            for key in loss_fns:
                if key == "human":
                    continue
                sub = df[df["loss_fn"] == key]
                if not sub.empty:
                    label = loss_label_map.get(key, key)
                    group = loss_group_map.get(key, "Other")
                    items.append(
                        {
                            "type": "individual",
                            "group": group,
                            "label": label,
                            "sub_df": sub,
                        }
                    )
        else:
            for key, label, grp in INDIVIDUAL_LOSS_FUNCTIONS:
                sub = df[df["loss_fn"] == key]
                if not sub.empty:
                    items.append(
                        {
                            "type": "individual",
                            "group": grp,
                            "label": label,
                            "sub_df": sub,
                        }
                    )

    if plot_mode in ("groups", "all"):
        for grp in AGGREGATED_GROUPS:
            group_models = grp["models"]
            if loss_fns is not None:
                group_models = [m for m in group_models if m in loss_fns]
            if group_models:
                sub = df[df["loss_fn"].isin(group_models)]
                if not sub.empty:
                    group_id = (
                        f"Aggregated_{grp['group']}"
                        if plot_mode == "all"
                        else grp["group"]
                    )
                    items.append(
                        {
                            "type": "group",
                            "group": group_id,
                            "label": grp["label"],
                            "sub_df": sub,
                        }
                    )

    return items


def plot_loss_linear_fits(
    df: pd.DataFrame,
    plot_mode: str = "losses",
    loss_fns: Sequence[str] | None = None,
    show_fit: bool = True,
    anchor_zero: bool = True,
    connect_dots: bool = False,
    error_mode: str = "ci",
    error_style: str = "bars",
    shade_range: bool = False,
    range_alpha: float = RANGE_ALPHA,
    shade_alpha: float = SHADE_ALPHA,
    ci_level: float = 0.95,
    output_path: Optional[Path | str] = None,
    plot_size: float = PLOT_SIZE,
    gap_x: float = GAP_X,
    gap_y: float = GAP_Y,
    gap_human: float = GAP_HUMAN,
    gap_group: float = GAP_GROUP,
    group_separators: bool = True,
    line_color: str = LINE_COLOR,
    marker_color: str = MARKER_COLOR,
    r2_decimals: int = R2_DECIMALS,
    fontsize_xlabel: float = FONTSIZE_XLABEL,
    fontsize_ylabel: float = FONTSIZE_YLABEL,
    fontsize_r2: float = FONTSIZE_R2,
    fontsize_ticks: float = FONTSIZE_TICKS,
    fontsize_xticks: Optional[float] = FONTSIZE_XTICKS,
    fontsize_yticks: Optional[float] = FONTSIZE_YTICKS,
    fontsize_title: float = FONTSIZE_TITLE,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """Generate publication-ready figure of linear fits for loss functions across modulations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing distances and/or human ratings across modulation amounts.
    plot_mode : str, default='losses'
        Plotting mode: 'losses', 'groups', 'all', or 'human'.
    loss_fns : Sequence[str] or None, default=None
        Optional filter restricting and ordering the loss functions displayed.
    show_fit : bool, default=True
        Whether to display linear line of best fit and annotated R^2 value.
    anchor_zero : bool, default=True
        Whether to anchor linear regression fit to y=0 at first x value.
    connect_dots : bool, default=False
        Whether to connect observed condition means with straight black line segments.
    error_mode : str, default='ci'
        Error bar metric: 'ci', 'std', or 'none'.
    error_style : str, default='bars'
        Visual error style: 'bars' (error bars), 'shade' (shaded ribbon), or 'both'.
    shade_range : bool, default=False
        Whether to shade the full [min, max] range with a lighter color.
    range_alpha : float, default=RANGE_ALPHA
        Opacity for min-max range shading.
    shade_alpha : float, default=SHADE_ALPHA
        Opacity for CI or STD shading ribbon.
    ci_level : float, default=0.95
        Confidence level for Student's t confidence intervals.
    output_path : Optional[Path or str], default=None
        Path to save figure image (.pdf, .png, .svg). Parent directories are created if missing.
    plot_size : float, default=PLOT_SIZE
        Size of each square subplot in inches.
    gap_x : float, default=GAP_X
        Horizontal gap between subplots in inches.
    gap_y : float, default=GAP_Y
        Vertical gap between consecutive rows within the same group in inches.
    gap_human : float, default=GAP_HUMAN
        Vertical gap between human benchmark row and subsequent model rows.
    gap_group : float, default=GAP_GROUP
        Vertical gap between representation groups in inches.
    group_separators : bool, default=True
        Whether to draw dashed horizontal separator lines between representation groups.
    line_color : str, default=LINE_COLOR
        Color for linear regression line and R^2 annotation.
    marker_color : str, default=MARKER_COLOR
        Color for markers, error bars, and ribbons.
    r2_decimals : int, default=R2_DECIMALS
        Number of decimal places for R^2 annotation.
    fontsize_xlabel : float, default=FONTSIZE_XLABEL
        Font size for common x-axis label.
    fontsize_ylabel : float, default=FONTSIZE_YLABEL
        Font size for row y-axis labels.
    fontsize_r2 : float, default=FONTSIZE_R2
        Font size for R^2 value annotations.
    fontsize_ticks : float, default=FONTSIZE_TICKS
        Default font size for axis ticks.
    fontsize_xticks : Optional[float], default=FONTSIZE_XTICKS
        Optional override for x-axis tick font size.
    fontsize_yticks : Optional[float], default=FONTSIZE_YTICKS
        Optional override for y-axis tick font size.
    fontsize_title : float, default=FONTSIZE_TITLE
        Font size for column titles.
    dpi : int, default=300
        Resolution in dots per inch for image export.
    show : bool, default=True
        Whether to display plot window interactively.

    Returns
    -------
    plt.Figure
        The generated Matplotlib Figure.

    Raises
    ------
    ValueError
        If no matching rows are found to plot.
    """
    loss_fns_clean = normalize_loss_fns(loss_fns)
    row_items = prepare_plot_items(plot_mode, df, loss_fns=loss_fns_clean)

    if not row_items:
        raise ValueError(
            f"No data rows available to plot for plot_mode='{plot_mode}' and loss_fns={loss_fns}. "
            "If plotting human data, ensure postprocessed human responses are loaded."
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9.0,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )

    eff_xticks = fontsize_xticks if fontsize_xticks is not None else fontsize_ticks
    eff_yticks = fontsize_yticks if fontsize_yticks is not None else fontsize_ticks

    n_rows = len(row_items)
    n_cols = len(MODULATION_COLUMNS)

    margin_left = 0.85
    margin_right = 0.06
    margin_bottom = 0.44
    margin_top = 0.32

    # Determine vertical gaps between consecutive rows
    row_gaps: list[float] = []
    has_separator: list[bool] = []
    for r in range(n_rows - 1):
        curr_item = row_items[r]
        next_item = row_items[r + 1]
        if curr_item.get("type") == "human":
            row_gaps.append(gap_human)
            has_separator.append(True)
        elif curr_item.get("group") != next_item.get("group"):
            row_gaps.append(gap_group)
            has_separator.append(bool(group_separators))
        else:
            row_gaps.append(gap_y)
            has_separator.append(False)

    fig_w = margin_left + n_cols * plot_size + (n_cols - 1) * gap_x + margin_right
    total_gaps_y = sum(row_gaps)
    fig_h = margin_bottom + n_rows * plot_size + total_gaps_y + margin_top

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Compute bottom coordinate in inches for each row r (0 is topmost, n_rows - 1 is bottommost)
    b_in_list = [0.0] * n_rows
    b_in_list[n_rows - 1] = margin_bottom
    for r in range(n_rows - 2, -1, -1):
        b_in_list[r] = b_in_list[r + 1] + plot_size + row_gaps[r]

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        b_in = b_in_list[r]
        for c in range(n_cols):
            l_in = margin_left + c * (plot_size + gap_x)
            ax = fig.add_axes(
                [
                    l_in / fig_w,
                    b_in / fig_h,
                    plot_size / fig_w,
                    plot_size / fig_h,
                ]
            )
            axes[r, c] = ax

    x_indices = np.arange(5)

    for r_idx, item in enumerate(row_items):
        sub_df = item["sub_df"]
        row_label = item["label"]
        is_human = item.get("type") == "human"

        # Calculate y-limits and ticks (5 tickmarks for human data and all loss functions)
        if is_human:
            y_upper = 104.0
            y_lower = -4.0
            y_ticks: list[float | int] = [0, 25, 50, 75, 100]
        else:
            row_max = 0.0
            for col in MODULATION_COLUMNS:
                sub_mod = sub_df[sub_df["mod_type"] == col["key"]]
                amounts = sorted(sub_mod["amount"].unique())
                for a in amounts:
                    vals = sub_mod[sub_mod["amount"] == a]["distance"].values
                    m, err = compute_point_stats(
                        vals, error_mode=error_mode, ci_level=ci_level
                    )
                    val = m + (err if err is not None else 0.0)
                    if shade_range and len(vals) > 0:
                        val = max(val, float(np.nanmax(vals)))
                    if val > row_max:
                        row_max = val

            y_ticks, y_lower, y_upper = compute_nice_5_ticks(row_max)

        for c_idx, col in enumerate(MODULATION_COLUMNS):
            ax = axes[r_idx, c_idx]
            sub_mod = sub_df[sub_df["mod_type"] == col["key"]]
            amounts = sorted(sub_mod["amount"].unique())

            means: list[float] = []
            errs: list[Optional[float]] = []
            mins: list[float] = []
            maxs: list[float] = []
            for a in amounts:
                vals = sub_mod[sub_mod["amount"] == a]["distance"].values
                m, err = compute_point_stats(
                    vals, error_mode=error_mode, ci_level=ci_level
                )
                means.append(m)
                errs.append(err)
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) > 0:
                    mins.append(float(np.min(valid_vals)))
                    maxs.append(float(np.max(valid_vals)))
                else:
                    mins.append(m)
                    maxs.append(m)

            means_arr = np.array(means)
            mins_arr = np.array(mins)
            maxs_arr = np.array(maxs)
            has_errs = error_mode != "none" and any(e is not None for e in errs)
            yerr_arr = (
                np.array([e if e is not None else 0.0 for e in errs])
                if has_errs
                else None
            )

            # 1. Shaded region for observed [min, max] range (lighter color)
            if shade_range and len(mins_arr) == len(x_indices):
                ax.fill_between(
                    x_indices,
                    mins_arr,
                    maxs_arr,
                    color=marker_color,
                    alpha=range_alpha,
                    edgecolor="none",
                    zorder=1.5,
                )

            # 2. Shaded region for CI or STD ribbon
            if error_style in ("shade", "both") and has_errs and yerr_arr is not None:
                y_lower_band = np.maximum(y_lower, means_arr - yerr_arr)
                y_upper_band = means_arr + yerr_arr
                ax.fill_between(
                    x_indices,
                    y_lower_band,
                    y_upper_band,
                    color=marker_color,
                    alpha=shade_alpha,
                    edgecolor="none",
                    zorder=2,
                )

            # 3. Linear regression line and annotated R^2
            if show_fit:
                slope, intercept, r2 = fit_linear_regression(
                    x_indices, means_arr, anchor_zero=anchor_zero
                )
                x_start = float(x_indices[0]) if anchor_zero else -0.25
                x_line = np.linspace(x_start, 4.25, 100)
                y_line = slope * x_line + intercept

                ax.plot(
                    x_line,
                    y_line,
                    color=line_color,
                    linewidth=2.0,
                    linestyle="-",
                    zorder=2.5,
                )

                # R^2 text in upper left corner
                r2_str = r"$\mathbf{R^2 = " + f"{r2:.{r2_decimals}f}" + r"}$"
                ax.text(
                    0.06,
                    0.92,
                    r2_str,
                    transform=ax.transAxes,
                    color=line_color,
                    fontsize=fontsize_r2,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )

            # 4. Straight black line segments connecting the dots
            if connect_dots:
                ax.plot(
                    x_indices,
                    means_arr,
                    color=marker_color,
                    linewidth=1.5,
                    linestyle="-",
                    zorder=3,
                )

            # 5. Data markers and vertical error bars
            show_vertical_bars = error_style in ("bars", "both") and has_errs
            ref_err = (
                yerr_arr[0] if (show_vertical_bars and yerr_arr is not None) else None
            )
            rem_err = (
                yerr_arr[1:] if (show_vertical_bars and yerr_arr is not None) else None
            )

            ax.errorbar(
                x_indices[0],
                means_arr[0],
                yerr=ref_err,
                fmt="s",
                color=marker_color,
                ecolor=marker_color,
                elinewidth=1.5,
                capsize=3.5 if show_vertical_bars else 0,
                capthick=1.5,
                markersize=5.5,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markeredgewidth=1.1,
                zorder=4,
            )
            ax.errorbar(
                x_indices[1:],
                means_arr[1:],
                yerr=rem_err,
                fmt="o",
                color=marker_color,
                ecolor=marker_color,
                elinewidth=1.5,
                capsize=3.5 if show_vertical_bars else 0,
                capthick=1.5,
                markersize=5.5,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markeredgewidth=1.1,
                zorder=4,
            )

            # Subplot aspect ratio 1:1
            ax.set_box_aspect(1)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(y_lower, y_upper)

            # Y-axis ticks: exactly 5 tickmarks for human data and all loss functions
            ax.set_yticks(y_ticks)

            # Column titles: only on the very top row (r_idx == 0)
            if r_idx == 0:
                ax.set_title(
                    col["title"], fontsize=fontsize_title, fontweight="bold", pad=6
                )

            # X-ticks: set explicitly on all rows so vertical gridlines align
            ax.set_xticks(x_indices)
            if r_idx == n_rows - 1:
                ax.set_xticklabels(col["tick_labels"], fontsize=eff_xticks)
                ax.tick_params(axis="x", labelsize=eff_xticks)
            else:
                ax.tick_params(labelbottom=False, bottom=False)

            # Y-axis label and ticks: only on the leftmost column (c_idx == 0)
            if c_idx == 0:
                ax.set_ylabel(
                    row_label, fontsize=fontsize_ylabel, fontweight="bold", labelpad=5
                )
                ax.tick_params(axis="y", labelsize=eff_yticks)
            else:
                ax.tick_params(labelleft=False, left=False)

            # Dotted gridlines across all subplots
            ax.grid(
                True,
                which="major",
                linestyle=":",
                color="#999999",
                alpha=0.7,
                linewidth=0.7,
            )
            ax.set_axisbelow(True)

    # Subtle dashed separator lines between human benchmark and representation groups
    for r in range(n_rows - 1):
        if has_separator[r]:
            sep_y_in = (b_in_list[r] + b_in_list[r + 1] + plot_size) / 2.0
            sep_y_norm = sep_y_in / fig_h
            fig.add_artist(
                plt.Line2D(
                    [margin_left / fig_w, 1.0 - (margin_right / fig_w)],
                    [sep_y_norm, sep_y_norm],
                    color="#888888",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )
            )

    # Common X-axis label centered at the bottom
    axes[n_rows - 1, 1].set_xlabel(
        "Modulation amount", fontsize=fontsize_xlabel, fontweight="bold", labelpad=5
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        log.info(f"Figure successfully saved to: {out_p}")

    if show:
        plt.show()

    return fig


def main() -> None:
    """Parse command line arguments and execute linear fit plotting."""
    parser = argparse.ArgumentParser(
        description="Generate figure of loss function linear fits across modulations (losses, groups, all, or human)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join(OUT_DIR, "audio_distances.tsv"),
        help=f"Path to distances TSV/CSV dataset (default: {OUT_DIR}/audio_distances.tsv)",
    )
    parser.add_argument(
        "--human-data",
        "--human",
        nargs="?",
        const=os.path.join(OUT_DIR, "listening_test_responses_postprocessed.tsv"),
        default=os.path.join(OUT_DIR, "listening_test_responses_postprocessed.tsv"),
        dest="human_data",
        help="Optional path to postprocessed human listening responses TSV. When provided, human ratings are graphed as row 0.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(OUT_DIR, "figures", "fig2_linear_regression.pdf"),
        help=f"Path to save figure image (default: {OUT_DIR}/figures/fig2_linear_regression.pdf). Supports .png, .pdf, .svg.",
    )
    parser.add_argument(
        "--plot",
        choices=["losses", "groups", "all", "human"],
        default="losses",
        help="Plotting mode: 'losses' (individual loss functions), 'groups' (3 aggregated rows), 'all', or 'human'.",
    )
    parser.add_argument(
        "--human-only",
        "--only-human",
        action="store_true",
        dest="human_only",
        help="Plot only the human listening test data row.",
    )
    parser.add_argument(
        "--loss-fn",
        "--loss-fns",
        "-l",
        nargs="+",
        # default=None,
        default=["mfcc", "jtfs", "panns_wavegram_logmel"],
        help="Filter analysis to specific loss function(s) (e.g. -l mfcc mss_log_lin, or -l human)",
    )
    parser.add_argument(
        "--no-fit",
        "--hide-fit",
        action="store_false",
        dest="show_fit",
        default=True,
        help="Do not display linear line of best fit and R^2 value annotation.",
    )
    parser.add_argument(
        "--anchor-zero",
        "--anchor-origin",
        "--anchor-first",
        "--anchor",
        action="store_true",
        dest="anchor_zero",
        default=True,
        help="Anchor the linear line of best fit to pass through y=0 at the first x value.",
    )
    parser.add_argument(
        "--connect-dots",
        "--connect-points",
        action="store_true",
        dest="connect_dots",
        default=False,
        help="Connect data points with straight black line segments.",
    )

    # Mutually exclusive error bar metric options
    err_group = parser.add_mutually_exclusive_group()
    err_group.add_argument(
        "--error-bars",
        choices=["ci", "std", "none"],
        dest="error_bars",
        default="std",
        help="Error bar metric: 'ci' (95%% confidence intervals), 'std' (sample standard deviation, default), or 'none'.",
    )

    # Shaded region / error style options
    parser.add_argument(
        "--error-style",
        choices=["bars", "shade", "both"],
        default="bars",
        dest="error_style",
        help="Error visual style: 'bars' (default: vertical error bars with caps), 'shade' (shaded ribbon), or 'both'.",
    )
    parser.add_argument(
        "--shade-range",
        "--range-shade",
        "--min-max",
        action="store_true",
        dest="shade_range",
        default=False,
        help="Shade the full [min, max] observed range with a lighter color than the CI/STD shading.",
    )
    parser.add_argument(
        "--group-separators",
        "--group-separator",
        "--group-lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="group_separators",
        help="Draw dashed horizontal separator lines between representation groups (default: False).",
    )
    parser.add_argument(
        "--no-separators",
        "--no-separator",
        action="store_false",
        dest="group_separators",
        help="Disable dashed horizontal separator lines.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for Student's t confidence intervals (default: 0.95).",
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

    loss_fns_clean = normalize_loss_fns(args.loss_fn)
    if args.human_only or (
        loss_fns_clean and [x.lower() for x in loss_fns_clean] == ["human"]
    ):
        args.plot = "human"

    if args.plot == "human":
        human_path = Path(args.human_data)
        if not human_path.exists():
            log.error(f"Human data file not found at: {human_path}")
            sys.exit(1)
        df_dist = load_human_data(human_path)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            log.error(f"Distances data file not found at: {input_path}")
            sys.exit(1)
        df_dist = load_distances_data(input_path)

        if args.human_data:
            human_path = Path(args.human_data)
            if not human_path.exists():
                log.warning(
                    f"Human data file not found at: {human_path}. Proceeding without human data."
                )
            else:
                df_human = load_human_data(human_path)
                df_dist = pd.concat([df_human, df_dist], ignore_index=True)

    out_arg = args.output
    out_path = Path(out_arg) if out_arg else None

    plot_loss_linear_fits(
        df=df_dist,
        plot_mode=args.plot,
        loss_fns=args.loss_fn,
        show_fit=args.show_fit,
        anchor_zero=args.anchor_zero,
        connect_dots=args.connect_dots,
        error_mode=args.error_bars,
        error_style=args.error_style,
        shade_range=args.shade_range,
        ci_level=args.ci_level,
        output_path=out_path,
        group_separators=args.group_separators,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
