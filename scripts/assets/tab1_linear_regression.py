"""Generate publication-ready LaTeX table for loss function linear fit determination coefficients (R^2).

Reads distances dataset (and optionally human listening test responses)
and outputs a formatted LaTeX table comparing the R^2 values across
amplitude, frequency, and irregularity modulations against human listeners
as the perceptual benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from fig2_linear_regression import (
    fit_linear_regression,
    load_distances_data,
    load_human_data,
    normalize_loss_fns,
)
from paths import OUT_DIR
from util import format_sig_figs

TABLE_METHOD_GROUPS: list[list[tuple[str, str]]] = [
    [
        ("mss_log_lin", "MSS Log + Linear"),
        ("mss_rev", "MSS Revisited"),
        ("mfcc", "MFCC"),
    ],
    [
        ("scat1d", "Scat1D"),
        ("jtfs", "JTFS"),
    ],
    [
        ("vggish", "VGGish"),
        ("encodec48k", r"EnCodec 48\,kHz"),
        ("clap", "MS-CLAP"),
        ("panns_wavegram_logmel", "PANNs WGLM"),
    ],
]

TABLE_AGGREGATED_GROUPS: list[dict[str, Any]] = [
    {"name": "STFT--based mean", "models": ["mss_log_lin", "mss_rev", "mfcc"]},
    {"name": "Wavelet--based mean", "models": ["scat1d", "jtfs"]},
    {
        "name": "Neural mean",
        "models": ["vggish", "encodec48k", "clap", "panns_wavegram_logmel"],
    },
]

DEFAULT_CAPTION: str = (
    "Zero-anchored linear fit $R^2$ values and distance family averages across amplitude, frequency, and irregularity modulation types.\n"
    "    Human listening test data acts as the perceptual benchmark.\n"
    r"    Plots are provided for \underline{underlined} methods in Fig.~\ref{fig:plots} which are the highest average $R^2$ scoring method from each distance family."
)
DEFAULT_LABEL: str = "tab:r2"


def compute_series_r2(
    sub_df: pd.DataFrame,
    mod_type: str,
    anchor_zero: bool = True,
) -> float:
    """Compute linear fit coefficient of determination (R^2) for a specific modulation condition.

    Parameters
    ----------
    sub_df : pd.DataFrame
        DataFrame subset containing condition responses.
    mod_type : str
        Modulation key ('amp', 'freq', or 'reg').
    anchor_zero : bool, default=True
        Whether linear regression is anchored to y=0 at first x value.

    Returns
    -------
    float
        R^2 value, or np.nan if data subset is empty.
    """
    sub = sub_df[sub_df["mod_type"] == mod_type]
    if sub.empty:
        return np.nan
    amounts = sorted(sub["amount"].unique())
    means = np.array([sub[sub["amount"] == a]["distance"].mean() for a in amounts])
    x_indices = np.arange(len(means))
    _, _, r2 = fit_linear_regression(x_indices, means, anchor_zero=anchor_zero)
    return r2


def generate_r2_latex_table(
    df: pd.DataFrame,
    df_human: Optional[pd.DataFrame] = None,
    anchor_zero: bool = True,
    sig_digits: int = 3,
    include_mean: bool = False,
    table_env: str = "table",
    loss_fns: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    label: str = DEFAULT_LABEL,
    underline_best: bool = True,
) -> str:
    """Generate publication-ready LaTeX table of R^2 linear fit determination coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Distances DataFrame.
    df_human : Optional[pd.DataFrame], default=None
        Optional human responses DataFrame to display as perceptual benchmark row.
    anchor_zero : bool, default=True
        Whether linear regression is anchored at origin.
    sig_digits : int, default=3
        Number of significant figures in the formatted table.
    include_mean : bool, default=False
        Whether to include a summary Mean column across all 3 modulations.
    table_env : str, default='table'
        LaTeX table environment wrapper ('table').
    loss_fns : Optional[Sequence[str]], default=None
        Optional filter restricting loss functions displayed in the table.
    caption : Optional[str], default=None
        Custom caption for LaTeX table.
    label : str, default='tab:r2'
        LaTeX cross-reference label.
    underline_best : bool, default=True
        Whether to underline the highest scoring method in each distance family.

    Returns
    -------
    str
        Complete LaTeX table markup string.
    """
    df_models = df[df["loss_fn"] != "human"].copy()
    if df_models.empty:
        default_dist_path = Path(OUT_DIR) / "audio_distances.tsv"
        if default_dist_path.exists():
            try:
                df_models = load_distances_data(default_dist_path)
            except Exception:
                pass

    if df_human is None:
        if "human" in df["loss_fn"].unique():
            df_human = df[df["loss_fn"] == "human"]
        else:
            default_hpath = Path(OUT_DIR) / "listening_test_responses_postprocessed.tsv"
            if default_hpath.exists():
                try:
                    df_human = load_human_data(default_hpath)
                except Exception:
                    df_human = None

    cols = ["amp", "freq", "reg"] + (["mean"] if include_mean else [])
    col_headers = {
        "amp": r"\multicolumn{1}{c}{\textbf{Amp. $R^2$}}",
        "freq": r"\multicolumn{1}{c}{\textbf{Freq. $R^2$}}",
        "reg": r"\multicolumn{1}{c}{\textbf{Irreg. $R^2$}}",
        "mean": r"\multicolumn{1}{c}{\textbf{Mean}}",
    }

    filtered_groups: list[list[tuple[str, str]]] = []
    for grp in TABLE_METHOD_GROUPS:
        if loss_fns is not None:
            active_m = [m for m in grp if m[0] in loss_fns]
            if active_m:
                filtered_groups.append(active_m)
        else:
            filtered_groups.append(grp)

    model_vals: dict[str, dict[str, float]] = {}
    for group in filtered_groups:
        for canon_key, _ in group:
            sub = df_models[df_models["loss_fn"] == canon_key]
            if not sub.empty:
                vals = {
                    c: compute_series_r2(sub, c, anchor_zero)
                    for c in ["amp", "freq", "reg"]
                }
                vals["mean"] = float(np.mean([vals[c] for c in ["amp", "freq", "reg"]]))
                model_vals[canon_key] = vals

    # Distance family averages: arithmetic mean of individual model R^2 scores
    group_vals: dict[str, dict[str, float]] = {}
    for g_info in TABLE_AGGREGATED_GROUPS:
        g_name = g_info["name"]
        group_models = g_info["models"]
        if loss_fns is not None:
            group_models = [m for m in group_models if m in loss_fns]
        active_models = [m for m in group_models if m in model_vals]
        if active_models:
            vals = {
                c: float(np.mean([model_vals[m][c] for m in active_models]))
                for c in ["amp", "freq", "reg"]
            }
            if include_mean:
                vals["mean"] = float(np.mean([vals[c] for c in ["amp", "freq", "reg"]]))
            group_vals[g_name] = vals

    human_vals: Optional[dict[str, float]] = None
    if df_human is not None and not df_human.empty:
        human_vals = {
            c: compute_series_r2(df_human, c, anchor_zero)
            for c in ["amp", "freq", "reg"]
        }
        human_vals["mean"] = float(
            np.mean([human_vals[c] for c in ["amp", "freq", "reg"]])
        )

    if caption is None:
        caption = DEFAULT_CAPTION

    lines: list[str] = [
        f"\\begin{{{table_env}}}[t]",
        r"\centering",
        r"\vspace{-\abovecaptionskip} % <-- Pulls the caption up flush to the top",
        r"\caption{",
        f"    {caption}",
        r"}",
        f"\\label{{{label}}}",
        r"\vspace{1pt}",
        r"\sisetup{",
        r"    round-pad = true,",
        r"    reset-text-series=false, ",
        r"    text-series-to-math=true, ",
        r"    mode=text,",
        r"    tight-spacing=false,",
        r"    separate-uncertainty=false,",
        r"    detect-weight=true,",
        r"    detect-inline-weight=math,",
        r"    table-text-alignment=right,",
        r"    table-number-alignment=right,",
        r"}",
        r"\setlength{\tabcolsep}{7.98pt}",
        r"\begin{tabular}{",
        r"    l",
    ]
    for _ in cols:
        lines.append(
            r"    S[round-mode=figures, round-precision=2, table-format=1.2, table-number-alignment=center, table-text-alignment=right]"
        )
    lines.extend(
        [
            r"}",
            r"    \toprule",
            r"    \textbf{Method} ",
        ]
    )
    for c in cols:
        lines.append(f"        & {col_headers[c]} ")
    lines.extend(
        [
            r"    \\",
            r"    \midrule",
        ]
    )

    if human_vals is not None:
        h_strs = [format_sig_figs(human_vals[c], sig_digits) for c in cols]
        h_name = r"\underline{Human Listeners}" if underline_best else "Human Listeners"
        lines.append(f"    {h_name:<26s}  & " + " & ".join(h_strs) + r" \\")
        lines.append(r"    \midrule")

    for g_idx, group in enumerate(filtered_groups):
        if g_idx > 0:
            lines.append(r"    \addlinespace[5pt]")

        # Find highest average scoring method within this family
        best_canon_key = None
        best_mean_r2 = -float("inf")
        for canon_key, _ in group:
            if canon_key in model_vals:
                m_mean = float(
                    np.mean([model_vals[canon_key][c] for c in ["amp", "freq", "reg"]])
                )
                if m_mean > best_mean_r2:
                    best_mean_r2 = m_mean
                    best_canon_key = canon_key

        for canon_key, display_name in group:
            if canon_key not in model_vals:
                continue
            if underline_best and canon_key == best_canon_key:
                full_name = f"\\underline{{{display_name}}}"
            else:
                full_name = display_name
            m_strs = [
                format_sig_figs(model_vals[canon_key][c], sig_digits) for c in cols
            ]
            lines.append(f"    {full_name:<26s} & " + " & ".join(m_strs) + r" \\")

    if group_vals:
        lines.append(r"    \midrule")
        for g_info in TABLE_AGGREGATED_GROUPS:
            g_name = g_info["name"]
            if g_name not in group_vals:
                continue
            g_strs = [format_sig_figs(group_vals[g_name][c], sig_digits) for c in cols]
            lines.append(f"    {g_name:<26s} & " + " & ".join(g_strs) + r" \\")

    lines.extend(
        [
            r"    \bottomrule",
            r"\end{tabular}",
            r"\vspace{-10pt}",
            f"\\end{{{table_env}}}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Parse command line arguments and print publication-ready LaTeX table."""
    parser = argparse.ArgumentParser(
        description="Print compact LaTeX table of loss function linear fit R^2 determination coefficients."
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
        help=f"Optional path to postprocessed human listening responses TSV (default: {OUT_DIR}/listening_test_responses_postprocessed.tsv).",
    )
    parser.add_argument(
        "--loss-fn",
        "--loss-fns",
        "-l",
        nargs="+",
        default=None,
        help="Filter table to specific loss function(s) (e.g. -l mfcc mss_log_lin).",
    )
    parser.add_argument(
        "--anchor-zero",
        "--anchor",
        action="store_true",
        dest="anchor_zero",
        default=True,
        help="Anchor linear fit to pass through y=0 at first x value (default: True).",
    )
    parser.add_argument(
        "--sig-digits",
        "--precision",
        type=int,
        default=3,
        dest="sig_digits",
        help="Number of significant digits / decimal places (default: 3).",
    )
    parser.add_argument(
        "--include-mean",
        action="store_true",
        dest="include_mean",
        default=False,
        help="Include the summary Mean column.",
    )
    parser.add_argument(
        "--no-underline",
        action="store_false",
        dest="underline_best",
        default=True,
        help="Do not underline the highest scoring method in each distance family.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Distances data file not found at: {input_path}\n")
        sys.exit(1)

    df_dist = load_distances_data(input_path)
    df_human = None
    if args.human_data:
        human_path = Path(args.human_data)
        if human_path.exists():
            df_human = load_human_data(human_path)

    loss_fns_clean = normalize_loss_fns(args.loss_fn)

    tex_table = generate_r2_latex_table(
        df=df_dist,
        df_human=df_human,
        anchor_zero=args.anchor_zero,
        sig_digits=args.sig_digits,
        include_mean=args.include_mean,
        table_env="table",
        loss_fns=loss_fns_clean,
        underline_best=args.underline_best,
    )

    print(tex_table)


if __name__ == "__main__":
    main()
