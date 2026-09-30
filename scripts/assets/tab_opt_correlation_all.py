"""Generate publication-ready comprehensive LaTeX table for audio distance correlations.

Reads audio distance correlations dataset (out/audio_distance_correlations.tsv)
and noise ceiling dataset (out/noise_ceilings.tsv), and outputs a formatted LaTeX table
comparing audio distance loss functions across amplitude, frequency, and irregularity
modulations against human listeners and the human noise ceiling benchmark across both
group and individual metrics.
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
    "Audio distance correlation with human perceptual dissimilarity across amplitude, frequency, and irregularity modulations. \n"
    r"The human noise ceiling acts as the reference upper bound, reported as its 95\% bootstrap confidence interval $[95\%\text{ CI}]$ for group correlation and lower noise ceiling ($\mu \pm \sigma$) for individual listener correlation." + "\n"
    r"For models, the highest correlation within each modulation condition is in \textbf{bold}, and the second highest is \underline{underlined}." + "\n"
    r"Statistical significance is denoted by asterisks ($^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$; unstarred indicates $p \ge 0.05$)."
)

DEFAULT_LABEL: str = "tab:correlation_all"


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
    sig_digits_corr: int = 3,
    sig_digits_std: int = 2,
    caption: Optional[str] = None,
    label: str = DEFAULT_LABEL,
) -> str:
    """Generate publication-ready comprehensive LaTeX table string from correlation and noise ceiling dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        Audio distance correlation DataFrame.
    nc_df : pd.DataFrame
        Noise ceiling results DataFrame.
    sig_digits_corr : int, default=3
        Number of significant digits for correlation coefficients and CIs.
    sig_digits_std : int, default=2
        Number of significant digits for standard deviations.
    caption : Optional[str], default=None
        LaTeX table caption text.
    label : str, default=DEFAULT_LABEL
        LaTeX cross-reference label.

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
        r"\begin{tabular}{",
        r"    l",
        r"    l",
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\,}} l",
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\hspace{{8pt}}$\\pm$\\hspace{{4pt}}}} ",
        f"    S[round-mode=figures, round-precision={sig_digits_std}, table-format=1.{sig_digits_std}, table-number-alignment=left]",
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\,}} l",
        f"    S[round-mode=figures, round-precision={sig_digits_corr}, table-format=1.{sig_digits_corr}, table-number-alignment=right] @{{\\hspace{{8pt}}$\\pm$\\hspace{{4pt}}}} ",
        f"    S[round-mode=figures, round-precision={sig_digits_std}, table-format=1.{sig_digits_std}, table-number-alignment=left]",
        r"}",
        r"    \toprule",
        r"    \multirow[t]{2}{*}{\textbf{Mod. Type}} & \multirow[t]{2}{*}{\textbf{Method}} & ",
        r"    \multicolumn{4}{c}{\textbf{Spearman} ($\rho$) $\uparrow$} & ",
        r"    \multicolumn{4}{c}{\textbf{Pearson} ($r$) $\uparrow$} \\",
        r"    \cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"        & & \multicolumn{2}{c}{Group} & \multicolumn{2}{c}{Indiv. ($\mu \pm \sigma$)} ",
        r"        & \multicolumn{2}{c}{Group} & \multicolumn{2}{c}{Indiv. ($\mu \pm \sigma$)} \\",
    ]

    total_models = sum(len(g) for g in TABLE_METHOD_GROUPS)
    num_rows_per_cond = 1 + total_models  # Noise ceiling + models

    for cond in CONDITIONS:
        lines.append(r"    \midrule")
        cond_label = CONDITION_MAP[cond]
        lines.append(f"    \\multirow{{{num_rows_per_cond}}}{{*}}{{{cond_label}}}")

        # Noise ceiling row from noise_ceilings.tsv
        nc_match = nc_df[
            (nc_df["granularity"] == "modulation") & (nc_df["condition"] == cond)
        ]
        if nc_match.empty:
            raise ValueError(
                f"No noise ceiling row found for condition '{cond}' in noise ceiling data."
            )
        nc_row = nc_match.iloc[0]

        # 95% bootstrap CI for group correlation
        sp_ci_low_str = format_sig_figs(
            nc_row["spearman_group_ci95_low"], sig_digits_corr
        )
        sp_ci_high_str = format_sig_figs(
            nc_row["spearman_group_ci95_high"], sig_digits_corr
        )
        sp_ci = f"$[{sp_ci_low_str}, {sp_ci_high_str}]$"

        pe_ci_low_str = format_sig_figs(
            nc_row["pearson_group_ci95_low"], sig_digits_corr
        )
        pe_ci_high_str = format_sig_figs(
            nc_row["pearson_group_ci95_high"], sig_digits_corr
        )
        pe_ci = f"$[{pe_ci_low_str}, {pe_ci_high_str}]$"

        # Lower noise ceiling for individual results (mean +/- std)
        sp_ind = format_sig_figs(nc_row["spearman_indiv_lower"], sig_digits_corr)
        sp_std = format_sig_figs(nc_row["spearman_indiv_lower_std"], sig_digits_std)
        pe_ind = format_sig_figs(nc_row["pearson_indiv_lower"], sig_digits_corr)
        pe_std = format_sig_figs(nc_row["pearson_indiv_lower_std"], sig_digits_std)

        lines.append(
            f"        & Noise Ceiling    & \\multicolumn{{2}}{{c}}{{{sp_ci}}} & {sp_ind:5s} & {sp_std:5s}"
            f"  & \\multicolumn{{2}}{{c}}{{{pe_ci}}} & {pe_ind:5s} & {pe_std:5s}  \\\\"
        )
        lines.append(r"        \cmidrule(lr){2-10}")

        # Extract model values for this condition
        cond_df = df[(df["condition"] == cond) & (df["granularity"] == "modulation")]
        cond_models_data: dict[str, pd.Series] = {}
        for group in TABLE_METHOD_GROUPS:
            for canon_key, _ in group:
                match = cond_df[cond_df["canonical_loss"] == canon_key]
                if match.empty:
                    match = cond_df[cond_df["loss_fn"] == canon_key]
                if not match.empty:
                    cond_models_data[canon_key] = match.iloc[0]

        # Prepare lists for highlight calculations across all models in this condition
        sp_grp_entries: list[tuple[str, float, str]] = []
        sp_ind_entries: list[tuple[str, float, str]] = []
        pe_grp_entries: list[tuple[str, float, str]] = []
        pe_ind_entries: list[tuple[str, float, str]] = []

        for group in TABLE_METHOD_GROUPS:
            for canon_key, _ in group:
                if canon_key in cond_models_data:
                    row = cond_models_data[canon_key]
                    sp_grp_entries.append(
                        (
                            canon_key,
                            float(row["spearman_group"]),
                            format_sig_figs(row["spearman_group"], sig_digits_corr),
                        )
                    )
                    sp_ind_entries.append(
                        (
                            canon_key,
                            float(row["spearman_indiv"]),
                            format_sig_figs(row["spearman_indiv"], sig_digits_corr),
                        )
                    )
                    pe_grp_entries.append(
                        (
                            canon_key,
                            float(row["pearson_group"]),
                            format_sig_figs(row["pearson_group"], sig_digits_corr),
                        )
                    )
                    pe_ind_entries.append(
                        (
                            canon_key,
                            float(row["pearson_indiv"]),
                            format_sig_figs(row["pearson_indiv"], sig_digits_corr),
                        )
                    )

        hl_sp_grp = assign_highlights(sp_grp_entries)
        hl_sp_ind = assign_highlights(sp_ind_entries)
        hl_pe_grp = assign_highlights(pe_grp_entries)
        hl_pe_ind = assign_highlights(pe_ind_entries)

        # Render each model group
        for g_idx, group in enumerate(TABLE_METHOD_GROUPS):
            if g_idx > 0:
                lines.append(r"        \addlinespace[5pt]")
            for canon_key, display_name in group:
                if canon_key not in cond_models_data:
                    continue
                row = cond_models_data[canon_key]
                sp_grp_str = hl_sp_grp.get(
                    canon_key, format_sig_figs(row["spearman_group"], sig_digits_corr)
                )
                sp_p_str = get_asterisks(row["spearman_group_p"])
                sp_ind_str = hl_sp_ind.get(
                    canon_key, format_sig_figs(row["spearman_indiv"], sig_digits_corr)
                )
                sp_std_str = format_sig_figs(row["spearman_indiv_std"], sig_digits_std)

                pe_grp_str = hl_pe_grp.get(
                    canon_key, format_sig_figs(row["pearson_group"], sig_digits_corr)
                )
                pe_p_str = get_asterisks(row["pearson_group_p"])
                pe_ind_str = hl_pe_ind.get(
                    canon_key, format_sig_figs(row["pearson_indiv"], sig_digits_corr)
                )
                pe_std_str = format_sig_figs(row["pearson_indiv_std"], sig_digits_std)

                m_name_pad = f"{display_name:<18s}"
                sp_ast_pad = f"{sp_p_str:<17s}" if sp_p_str else " " * 18
                pe_ast_pad = f"{pe_p_str:<17s}" if pe_p_str else " " * 18

                lines.append(
                    f"        & {m_name_pad}& {sp_grp_str} & {sp_ast_pad}& {sp_ind_str} & {sp_std_str:<5s}"
                    f" & {pe_grp_str} & {pe_ast_pad}& {pe_ind_str} & {pe_std_str:<5s} \\\\"
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
        description="Print comprehensive publication-ready LaTeX table of audio distance correlations."
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
        dest="sig_digits_corr",
        help="Number of significant digits for correlation coefficients and CIs (default: 3).",
    )
    parser.add_argument(
        "--sig-digits-std",
        "--sig-figs-std",
        type=int,
        default=2,
        dest="sig_digits_std",
        help="Number of significant digits for standard deviations (default: 2).",
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
        sig_digits_corr=args.sig_digits_corr,
        sig_digits_std=args.sig_digits_std,
    )

    print(table_latex)


if __name__ == "__main__":
    main()
