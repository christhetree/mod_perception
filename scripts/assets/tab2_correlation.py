"""Generate publication-ready compact LaTeX table for audio distance group correlations.

Reads audio distance correlations dataset (out/audio_distance_correlations.tsv)
and noise ceiling dataset (out/noise_ceilings.tsv), and outputs a formatted LaTeX table
comparing audio distance loss functions across amplitude, frequency, and irregularity
modulations against the human noise ceiling benchmark and distance family averages.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from paths import OUT_DIR
from util import format_sig_figs

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

__all__ = [
    "CONDITIONS",
    "CONDITION_MAP",
    "TABLE_METHOD_GROUPS",
    "TABLE_AGGREGATED_GROUPS",
    "DEFAULT_CAPTION",
    "DEFAULT_LABEL",
    "get_asterisks",
    "assign_highlights",
    "generate_latex_table",
    "main",
]

CONDITIONS: list[str] = ["amp", "freq", "reg"]

CONDITION_MAP: dict[str, str] = {
    "amp": "Amplitude",
    "freq": "Frequency",
    "reg": "Irregularity",
}

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

LOSS_FN_ALIASES: dict[str, str] = {
    "scat1d_log1p": "scat1d",
    "jtfs_log1p": "jtfs",
    "clap2": "clap",
    "panns_wglm": "panns_wavegram_logmel",
    "encodec": "encodec48k",
    "encodec48": "encodec48k",
    "encodec24k": "encodec48k",
}

DEFAULT_CAPTION: str = (
    "Audio distance function correlation coefficients and distance family averages for the human perceptual data. \n"
    r"Noise ceilings are reported as split-half bootstrapped 95\% confidence intervals." + "\n"
    r"Highest correlation values are in \textbf{bold}; second highest are \underline{underlined}." + "\n"
    r"Statistical significance is denoted by asterisks (unstarred: $p \ge 0.05$ or N/A for the last three aggregate rows, $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$)."
)

DEFAULT_LABEL: str = "tab:correlation"


def get_asterisks(p_val: float) -> str:
    """Return LaTeX significance asterisks based on p-value.

    Parameters
    ----------
    p_val : float
        Two-sided p-value.

    Returns
    -------
    str
        LaTeX formatted asterisk string ($^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$).
    """
    if p_val is None or pd.isna(p_val):
        return ""
    if p_val < 0.001:
        return r"$^{***}$"
    elif p_val < 0.01:
        return r"$^{**}$"
    elif p_val < 0.05:
        return r"$^{*}$"
    return ""


def assign_highlights(
    model_entries: list[tuple[str, float, str]],
) -> dict[str, str]:
    """Identify top 2 models per metric.

    Parameters
    ----------
    model_entries : list[tuple[str, float, str]]
        List of tuples (model_key, numeric_val, formatted_string_val).

    Returns
    -------
    dict[str, str]
        Dictionary mapping model_key to formatted string with \\textbf or \\underline.
    """
    if not model_entries:
        return {}

    sorted_by_val = sorted(model_entries, key=lambda x: x[1], reverse=True)
    top1_key, top1_val, top1_str = sorted_by_val[0]

    top2_key, top2_val, top2_str = None, None, None
    if len(sorted_by_val) > 1:
        top2_key, top2_val, top2_str = sorted_by_val[1]

    res = {}
    for key, val, s_val in model_entries:
        if (
            key == top1_key
            or val == top1_val
            or (s_val == top1_str and s_val != top2_str)
        ):
            res[key] = f"{{\\textbf{{{s_val}}}}}"
        elif (
            key == top2_key
            or val == top2_val
            or (s_val == top2_str and s_val != top1_str)
        ):
            res[key] = f"{{\\underline{{{s_val}}}}}"
        elif s_val == top1_str and top2_str == top1_str:
            if key == sorted_by_val[0][0]:
                res[key] = f"{{\\textbf{{{s_val}}}}}"
            elif key == sorted_by_val[1][0] or (
                len(sorted_by_val) > 2 and key == sorted_by_val[2][0]
            ):
                res[key] = f"{{\\underline{{{s_val}}}}}"
            else:
                res[key] = s_val
        else:
            res[key] = s_val

    return res


def generate_latex_table(
    df: pd.DataFrame,
    nc_df: pd.DataFrame,
    sig_digits: int = 3,
    group_means: bool = True,
    caption: Optional[str] = None,
    label: str = DEFAULT_LABEL,
    tabcolsep: str = "7.82pt",
) -> str:
    """Generate publication-ready compact LaTeX table string from correlation and noise ceiling dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        Audio distance correlation DataFrame.
    nc_df : pd.DataFrame
        Noise ceiling results DataFrame.
    sig_digits : int, default=3
        Number of significant digits for formatted values.
    group_means : bool, default=True
        Whether to include aggregate distance family mean rows.
    caption : Optional[str], default=None
        LaTeX table caption text.
    label : str, default=DEFAULT_LABEL
        LaTeX cross-reference label.
    tabcolsep : str, default='7.82pt'
        LaTeX column separation width.

    Returns
    -------
    str
        Complete LaTeX table string.

    Raises
    ------
    ValueError
        If required condition data is missing from the datasets.
    """
    df = df.copy()
    if "loss_fn" in df.columns:
        df["canonical_loss"] = df["loss_fn"].map(lambda x: LOSS_FN_ALIASES.get(x, x))

    eff_caption = caption if caption is not None else DEFAULT_CAPTION

    lines: list[str] = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\vspace{-\abovecaptionskip} % <-- Pulls the caption up flush to the top",
        r"\caption{",
        f"{eff_caption}",
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
        r"}",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        r"\begin{tabular}{",
        r"    l",
    ]

    for _ in range(6):
        lines.append(
            f"    S[round-mode=figures, round-precision={sig_digits}, table-format=1.{sig_digits}, table-number-alignment=right] @{{\\,}} l"
        )
    lines.extend(
        [
            r"}",
            r"    \toprule",
            r"    \multirow[t]{2}{*}{\textbf{Method}} ",
            r"        & \multicolumn{4}{c}{\textbf{Amplitude}} ",
            r"        & \multicolumn{4}{c}{\textbf{Frequency}} ",
            r"        & \multicolumn{4}{c}{\textbf{Irregularity}} \\",
            r"    \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}",
            r"        & \multicolumn{2}{c}{Spearman ($\rho$) $\uparrow$} & \multicolumn{2}{c}{Pearson ($r$) $\uparrow$} ",
            r"        & \multicolumn{2}{c}{Spearman ($\rho$) $\uparrow$} & \multicolumn{2}{c}{Pearson ($r$) $\uparrow$} ",
            r"        & \multicolumn{2}{c}{Spearman ($\rho$) $\uparrow$} & \multicolumn{2}{c}{Pearson ($r$) $\uparrow$} \\",
            r"    \midrule",
        ]
    )

    # Noise ceiling row across all 3 conditions
    lines.append("    Noise Ceiling ")
    for cond_idx, cond in enumerate(CONDITIONS):
        nc_match = nc_df[
            (nc_df["granularity"] == "modulation") & (nc_df["condition"] == cond)
        ]
        if nc_match.empty:
            raise ValueError(
                f"No noise ceiling row found for condition '{cond}' in noise ceiling data."
            )
        nc_data = nc_match.iloc[0]

        sp_ci_low = format_sig_figs(nc_data["spearman_group_ci95_low"], sig_digits)
        sp_ci_high = format_sig_figs(nc_data["spearman_group_ci95_high"], sig_digits)
        sp_ci = f"$[{sp_ci_low}, {sp_ci_high}]$"

        pe_ci_low = format_sig_figs(nc_data["pearson_group_ci95_low"], sig_digits)
        pe_ci_high = format_sig_figs(nc_data["pearson_group_ci95_high"], sig_digits)
        pe_ci = f"$[{pe_ci_low}, {pe_ci_high}]$"

        line_term = r" \\" if cond_idx == len(CONDITIONS) - 1 else ""
        lines.append(
            f"        & \\multicolumn{{2}}{{c}}{{{sp_ci}}} & \\multicolumn{{2}}{{c}}{{{pe_ci}}}{line_term}"
        )

    lines.append(r"    \midrule")

    # Compute highlights for each condition independently
    hl_by_cond: dict[str, dict[str, dict[str, str]]] = {}
    cond_dfs: dict[str, pd.DataFrame] = {}
    for cond in CONDITIONS:
        c_df = df[(df["condition"] == cond) & (df["granularity"] == "modulation")]
        cond_dfs[cond] = c_df

        sp_entries = []
        pe_entries = []
        for group in TABLE_METHOD_GROUPS:
            for canon_key, _ in group:
                match = c_df[c_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = c_df[c_df["loss_fn"] == canon_key]
                if not match.empty:
                    r = match.iloc[0]
                    sp_entries.append(
                        (
                            canon_key,
                            float(r["spearman_group"]),
                            format_sig_figs(r["spearman_group"], sig_digits),
                        )
                    )
                    pe_entries.append(
                        (
                            canon_key,
                            float(r["pearson_group"]),
                            format_sig_figs(r["pearson_group"], sig_digits),
                        )
                    )

        hl_by_cond[cond] = {
            "sp": assign_highlights(sp_entries),
            "pe": assign_highlights(pe_entries),
        }

    # Render each model across the 3 conditions
    for g_idx, group in enumerate(TABLE_METHOD_GROUPS):
        if g_idx > 0:
            lines.append(r"    \addlinespace[5pt]")
        for canon_key, display_name in group:
            lines.append(f"    {display_name:<18s}")
            for cond_idx, cond in enumerate(CONDITIONS):
                c_df = cond_dfs[cond]
                match = c_df[c_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = c_df[c_df["loss_fn"] == canon_key]
                if match.empty:
                    continue
                row = match.iloc[0]

                sp_val = hl_by_cond[cond]["sp"].get(
                    canon_key, format_sig_figs(row["spearman_group"], sig_digits)
                )
                sp_ast = get_asterisks(row["spearman_group_p"])
                pe_val = hl_by_cond[cond]["pe"].get(
                    canon_key, format_sig_figs(row["pearson_group"], sig_digits)
                )
                pe_ast = get_asterisks(row["pearson_group_p"])

                sp_ast_pad = f"{sp_ast:<15s}" if sp_ast else " " * 15
                pe_ast_pad = f"{pe_ast:<15s}" if pe_ast else " " * 15

                line_term = r" \\" if cond_idx == len(CONDITIONS) - 1 else ""
                lines.append(
                    f"        & {sp_val} & {sp_ast_pad}& {pe_val} & {pe_ast_pad}{line_term}"
                )

    # Optional extra rows with distance family averages
    if group_means:
        lines.append(r"    \midrule")
        group_means_hl: dict[str, dict[str, dict[str, str]]] = {}
        for cond in CONDITIONS:
            c_df = cond_dfs[cond]
            sp_entries = []
            pe_entries = []
            for g_info in TABLE_AGGREGATED_GROUPS:
                g_name = g_info["name"]
                group_models = g_info["models"]
                sp_vals = []
                pe_vals = []
                for canon_key in group_models:
                    match = c_df[c_df["canonical_loss"] == canon_key]
                    if match.empty:
                        match = c_df[c_df["loss_fn"] == canon_key]
                    if not match.empty:
                        sp_vals.append(float(match.iloc[0]["spearman_group"]))
                        pe_vals.append(float(match.iloc[0]["pearson_group"]))

                sp_mean = sum(sp_vals) / len(sp_vals) if sp_vals else float("nan")
                pe_mean = sum(pe_vals) / len(pe_vals) if pe_vals else float("nan")

                sp_entries.append(
                    (g_name, sp_mean, format_sig_figs(sp_mean, sig_digits))
                )
                pe_entries.append(
                    (g_name, pe_mean, format_sig_figs(pe_mean, sig_digits))
                )

            group_means_hl[cond] = {
                "sp": assign_highlights(sp_entries),
                "pe": assign_highlights(pe_entries),
            }

        for g_info in TABLE_AGGREGATED_GROUPS:
            g_name = g_info["name"]
            lines.append(f"    {g_name:<19s}")
            for cond_idx, cond in enumerate(CONDITIONS):
                sp_str = group_means_hl[cond]["sp"].get(g_name, "")
                pe_str = group_means_hl[cond]["pe"].get(g_name, "")

                line_term = r" \\" if cond_idx == len(CONDITIONS) - 1 else ""
                lines.append(
                    f"        & \\multicolumn{{2}}{{l}}{{{sp_str}}} & \\multicolumn{{2}}{{l}}{{{pe_str}}}{line_term}"
                )

    lines.extend(
        [
            r"    \bottomrule",
            r"\end{tabular}",
            r"\vspace{-11pt}",
            r"\end{table*}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Parse command line arguments and print publication-ready LaTeX table."""
    parser = argparse.ArgumentParser(
        description="Print publication-ready compact LaTeX table of audio distance group correlations."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=os.path.join(OUT_DIR, "audio_distance_correlations.tsv"),
        help=f"Path to correlation results TSV dataset (default: {OUT_DIR}/audio_distance_correlations.tsv)",
    )
    parser.add_argument(
        "-nc",
        "--noise-ceiling",
        default=os.path.join(OUT_DIR, "noise_ceilings.tsv"),
        dest="noise_ceiling",
        help=f"Path to noise ceilings TSV dataset (default: {OUT_DIR}/noise_ceilings.tsv)",
    )
    parser.add_argument(
        "--sig-digits",
        "--sig-figs",
        type=int,
        default=3,
        dest="sig_digits",
        help="Number of significant digits for correlation coefficients and CIs (default: 3).",
    )
    parser.add_argument(
        "--group-means",
        "--means",
        action="store_true",
        default=True,
        dest="group_means",
        help="Display 3 extra rows with the mean for each method group (STFT-based, Wavelet-based, and Neural-based) at the bottom of the table.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save generated LaTeX table to a .tex file.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress printing to terminal (useful when exporting only to file).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Correlation results file not found at: {input_path}")

    nc_path = Path(args.noise_ceiling).expanduser().resolve()
    if not nc_path.exists():
        raise FileNotFoundError(f"Noise ceiling results file not found at: {nc_path}")

    sep_in = "\t" if input_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(input_path, sep=sep_in)

    sep_nc = "\t" if nc_path.suffix in [".tsv", ".txt"] else ","
    nc_df = pd.read_csv(nc_path, sep=sep_nc)

    table_latex = generate_latex_table(
        df=df,
        nc_df=nc_df,
        sig_digits=args.sig_digits,
        group_means=args.group_means,
    )

    if not args.quiet:
        print(table_latex)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_latex, encoding="utf-8")
        log.info(f"Table successfully written to: {out_path}")


if __name__ == "__main__":
    main()
