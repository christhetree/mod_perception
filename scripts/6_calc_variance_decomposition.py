"""Interaction-Pooled ANOVA and Variance Decomposition for Audio Loss Functions.

Analyzes single-replicate distance measurements from audio distance datasets alongside
human listening study mean ratings from listening test responses.

Treats each metric evaluation as an unreplicated factorial design across:
- modulation (3 levels: amp, freq, reg)
- feature (3 levels: brightness, richness, warmth)
- source (2 levels: real, synthetic)
- rating_stimulus / amount (4 non-reference stimulus levels: condition_a .. condition_d)

Total cells = 3 x 3 x 2 x 4 = 72 observations per evaluation metric.

Optionally allows pooling ratings / distances across one or more factors before
computing and displaying the variance analysis (e.g. pooling across feature and source
leaves only modulation and rating_stimulus as main factors).

ANOVA is computed via statsmodels.api.stats.anova_lm (Type II Sum of Squares).
Vectorized effect sizes (np2, eta_sq) match standard pingouin and ANOVA implementations.
Multiple hypothesis corrections (Bonferroni & Benjamini-Hochberg FDR) are computed via
statsmodels.stats.multitest.multipletests.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from paths import OUT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

__all__ = [
    "ALL_FACTORS",
    "FACTOR_ALIASES",
    "FACTOR_DISPLAY_NAMES",
    "DISPLAY_FACTOR_ORDER",
    "CONDITION_AMOUNTS",
    "CONDITION_LABELS",
    "normalize_pool_factors",
    "pool_data",
    "parse_wavetable",
    "map_amount_to_condition",
    "prepare_loss_data",
    "load_human_data",
    "compute_interaction_pooled_anova",
    "format_table_for_display",
    "extract_variance_record",
    "run_variance_analysis",
    "main",
]

ALL_FACTORS: list[str] = ["modulation", "feature", "source", "rating_stimulus"]

FACTOR_ALIASES: dict[str, str] = {
    "modulation": "modulation",
    "mod": "modulation",
    "mod_type": "modulation",
    "feature": "feature",
    "feat": "feature",
    "timbre": "feature",
    "source": "source",
    "src": "source",
    "rating_stimulus": "rating_stimulus",
    "stimulus": "rating_stimulus",
    "amount": "rating_stimulus",
    "amt": "rating_stimulus",
    "rating": "rating_stimulus",
}

FACTOR_DISPLAY_NAMES: dict[str, str] = {
    "rating_stimulus": "% Var (Amount)",
    "modulation": "% Var (Modulation)",
    "feature": "% Var (Feature)",
    "source": "% Var (Source)",
}

DISPLAY_FACTOR_ORDER: list[str] = [
    "rating_stimulus",
    "modulation",
    "feature",
    "source",
]

CONDITION_AMOUNTS: dict[str, list[float]] = {
    "amp": [0.3, 0.5, 0.7, 0.9],
    "freq": [0.5, 1.0, 2.0, 4.0],
    "reg": [0.125, 0.25, 0.375, 0.5],
}

CONDITION_LABELS: list[str] = [
    "condition_a",
    "condition_b",
    "condition_c",
    "condition_d",
]


def normalize_pool_factors(raw_factors: Optional[Sequence[str]]) -> list[str]:
    """Normalize and validate factors to pool over.

    Parses comma-separated and space-separated strings, resolves aliases, and
    verifies that at least 2 factors remain for unreplicated factorial ANOVA.

    Parameters
    ----------
    raw_factors : Optional[Sequence[str]]
        List or sequence of raw factor names/aliases provided via CLI or API.

    Returns
    -------
    list[str]
        List of unique canonical factor names to pool across.

    Raises
    ------
    ValueError
        If an unrecognized factor alias is encountered or fewer than 2 factors remain.
    """
    if not raw_factors:
        return []

    pooled: list[str] = []
    for item in raw_factors:
        for f in item.replace(",", " ").split():
            clean = f.strip().lower()
            if not clean:
                continue
            if clean not in FACTOR_ALIASES:
                valid = sorted(set(FACTOR_ALIASES.values()))
                raise ValueError(
                    f"Unknown factor '{clean}'. Valid factors are: {valid} "
                    f"(aliases accepted: {list(FACTOR_ALIASES.keys())})"
                )
            canon = FACTOR_ALIASES[clean]
            if canon not in pooled:
                pooled.append(canon)

    remaining = [f for f in ALL_FACTORS if f not in pooled]
    if len(remaining) < 2:
        raise ValueError(
            f"Cannot pool over {pooled}. At least 2 factors must remain for unreplicated factorial ANOVA, "
            f"but only {len(remaining)} remained: {remaining}."
        )

    return pooled


def pool_data(
    df_data: pd.DataFrame,
    remaining_factors: Sequence[str],
) -> pd.DataFrame:
    """Pool ratings/distances across non-selected factors by taking cell means across remaining factors.

    Parameters
    ----------
    df_data : pd.DataFrame
        Input dataset containing factorial columns and dependent variable 'distance'.
    remaining_factors : Sequence[str]
        List of factor columns to retain in the grouped aggregation.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with cell means for each remaining factor combination.
    """
    if set(remaining_factors) == set(ALL_FACTORS):
        return df_data.copy()

    pooled = df_data.groupby(list(remaining_factors), as_index=False)["distance"].mean()
    if "loss_fn" in df_data.columns:
        pooled["loss_fn"] = df_data["loss_fn"].iloc[0]

    return pooled


def parse_wavetable(wt: str) -> tuple[str, str]:
    """Extract feature and sound source from a wavetable identifier.

    Parameters
    ----------
    wt : str
        Wavetable string (e.g. 'brightness_real__harmonics__synced_sines__256_1024').

    Returns
    -------
    tuple[str, str]
        Tuple of (feature, source), e.g. ('brightness', 'real').

    Raises
    ------
    ValueError
        If the wavetable string does not contain at least feature and source separated by underscore.
    """
    parts = wt.split("__")[0].split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Unable to parse feature and source from wavetable identifier: '{wt}'"
        )
    feature = parts[0]
    source = parts[1]
    return feature, source


def map_amount_to_condition(mod_type: str, amount: float) -> str:
    """Map a numeric modulation amount to its corresponding ordinal condition level (a, b, c, d).

    Uses floating-point tolerance matching to avoid precision issues.

    Parameters
    ----------
    mod_type : str
        Modulation type ('amp', 'freq', 'reg').
    amount : float
        Numeric modulation amount value.

    Returns
    -------
    str
        Condition label ('condition_a', 'condition_b', 'condition_c', 'condition_d').

    Raises
    ------
    ValueError
        If mod_type is unknown or amount is not a valid condition value for mod_type.
    """
    amounts = CONDITION_AMOUNTS.get(mod_type)
    if amounts is None:
        raise ValueError(
            f"Unknown modulation type: '{mod_type}'. Expected one of: {list(CONDITION_AMOUNTS.keys())}"
        )

    for idx, target in enumerate(amounts):
        if np.isclose(amount, target, atol=1e-5):
            return CONDITION_LABELS[idx]

    raise ValueError(
        f"Amount {amount} is not a valid condition for modulation type '{mod_type}' ({amounts})"
    )


def prepare_loss_data(df: pd.DataFrame, loss_fn: str) -> pd.DataFrame:
    """Filter and format the 72 non-reference rows for a specific loss function.

    Validates that exactly 72 balanced factorial cells are present.

    Parameters
    ----------
    df : pd.DataFrame
        Source distances DataFrame containing 'loss_fn', 'is_reference', 'wavetable', 'mod_type', 'amount'.
    loss_fn : str
        Name of the audio loss function to extract.

    Returns
    -------
    pd.DataFrame
        Formatted single-replicate factorial dataset for the specified loss function.

    Raises
    ------
    ValueError
        If no non-reference data is found or the cell count differs from the expected 72 cells.
    """
    sub = df[(df["loss_fn"] == loss_fn) & (~df["is_reference"])].copy()
    if sub.empty:
        raise ValueError(f"No non-reference data found for loss function: {loss_fn}")

    features, sources = zip(*[parse_wavetable(w) for w in sub["wavetable"]])
    sub["feature"] = features
    sub["source"] = sources
    sub["modulation"] = sub["mod_type"]
    sub["rating_stimulus"] = [
        map_amount_to_condition(m, a) for m, a in zip(sub["modulation"], sub["amount"])
    ]

    expected_rows = 3 * 3 * 2 * 4  # 72
    factors = ["modulation", "feature", "source", "rating_stimulus"]
    cell_counts = sub.groupby(factors).size()
    if len(cell_counts) != expected_rows or not (cell_counts == 1).all():
        raise ValueError(
            f"{loss_fn}: expected exactly one observation per factorial cell ({expected_rows} total); "
            f"found {len(cell_counts)} unique cells."
        )

    return sub


def load_human_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Aggregate human listening test responses into the standard 72-cell factorial format.

    Excludes reference stimulus ratings and averages participant scores across factorial cells.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to postprocessed human listening responses (TSV or CSV).

    Returns
    -------
    pd.DataFrame
        Aggregated 72-cell human benchmark DataFrame with columns:
        ['modulation', 'feature', 'source', 'rating_stimulus', 'distance', 'loss_fn'].
    """
    path = Path(data_path).expanduser().resolve()
    sep = "\t" if path.suffix in [".tsv", ".txt"] else ","
    df_human = pd.read_csv(path, sep=sep)

    if "rating_stimulus" in df_human.columns:
        df_human = df_human[df_human["rating_stimulus"] != "reference"].copy()

    if "modulation" not in df_human.columns and "trial_id" in df_human.columns:
        split_cols = df_human["trial_id"].str.split("_", expand=True)
        df_human["modulation"] = split_cols[0]
        df_human["feature"] = split_cols[1]
        df_human["source"] = split_cols[2]

    rating_col = "rating_score" if "rating_score" in df_human.columns else "distance"

    # Mean rating score per condition across all participants
    human_cells = (
        df_human.groupby(
            ["modulation", "feature", "source", "rating_stimulus"], as_index=False
        )[rating_col]
        .mean()
        .rename(columns={rating_col: "distance"})
    )
    human_cells["loss_fn"] = "human"
    return human_cells


def compute_interaction_pooled_anova(
    df_data: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
    max_interaction: int = 2,
    dv: str = "distance",
) -> pd.DataFrame:
    """Compute interaction-pooled ANOVA and variance decomposition using statsmodels.

    ANOVA table is computed via statsmodels.api.stats.anova_lm (Type II SS).
    Calculates:
    - Sum of Squares (SS), Degrees of Freedom (DF), Mean Squares (MS)
    - F-statistics and uncorrected p-values (p_unc)
    - Bonferroni (p_bonf) and Benjamini-Hochberg (p_fdr) multiple testing corrections
    - Partial eta-squared: np2 = (F * DF1) / (F * DF1 + DF2)
    - Total variance explained: eta_sq = SS / SS_total, pct_var = eta_sq * 100

    Parameters
    ----------
    df_data : pd.DataFrame
        Input data table containing factors and dependent variable.
    factors : Optional[Sequence[str]], default=None
        List of factorial factor columns to analyze (defaults to ALL_FACTORS).
    max_interaction : int, default=2
        Maximum interaction order to include (clamped if fewer factors remain).
    dv : str, default='distance'
        Dependent variable column name.

    Returns
    -------
    pd.DataFrame
        Detailed ANOVA results table.

    Raises
    ------
    ValueError
        If fewer than 2 factors are provided.
    """
    if factors is None:
        factors = list(ALL_FACTORS)

    effective_max_interaction = min(max_interaction, len(factors) - 1)
    if effective_max_interaction < 1:
        raise ValueError(
            f"At least 2 factors are required for unreplicated interaction-pooled ANOVA; got {len(factors)}."
        )

    factor_terms = " + ".join(f"C({f})" for f in factors)
    if effective_max_interaction == 1:
        formula = f"{dv} ~ {factor_terms}"
    else:
        formula = f"{dv} ~ ({factor_terms})**{effective_max_interaction}"

    model = ols(formula, data=df_data).fit()

    # Core ANOVA table from statsmodels
    aov = sm.stats.anova_lm(model, typ=2).reset_index()
    aov = aov.rename(
        columns={
            "index": "Source",
            "sum_sq": "SS",
            "df": "DF",
            "PR(>F)": "p_unc",
        }
    )

    # Standardize factor names
    aov["Source"] = (
        aov["Source"]
        .str.replace("C(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("Residual", "Residual (pooled)", regex=False)
    )

    # Degrees of Freedom as integer
    aov["DF"] = aov["DF"].astype(int)

    # Mean Squares: MS = SS / DF
    aov["MS"] = aov["SS"] / aov["DF"]

    # Degrees of freedom for residual
    resid_matches = aov.loc[aov["Source"] == "Residual (pooled)", "DF"].values
    resid_df = resid_matches[0] if len(resid_matches) > 0 else 1

    # Partial eta-squared: np2 = (F * DF1) / (F * DF1 + DF2)  (Pingouin anovan standard)
    aov["np2"] = (aov["F"] * aov["DF"]) / (aov["F"] * aov["DF"] + resid_df)

    # Exact Variance Decomposition: eta_sq = SS / SS_total
    ss_total = aov["SS"].sum()
    aov["eta_sq"] = aov["SS"] / ss_total
    aov["pct_var"] = aov["eta_sq"] * 100.0

    # Multiple hypothesis testing corrections:
    # 1. Bonferroni (controls Family-Wise Error Rate, FWER)
    # 2. Benjamini-Hochberg (controls False Discovery Rate, FDR)
    mask = aov["p_unc"].notna() & (aov["Source"] != "Residual (pooled)")
    aov["p_bonf"] = np.nan
    aov["p_fdr"] = np.nan
    if mask.any():
        p_vals = aov.loc[mask, "p_unc"]
        aov.loc[mask, "p_bonf"] = multipletests(p_vals, method="bonferroni")[1]
        aov.loc[mask, "p_fdr"] = multipletests(p_vals, method="fdr_bh")[1]

    col_order = [
        "Source",
        "SS",
        "DF",
        "MS",
        "F",
        "p_unc",
        "p_bonf",
        "p_fdr",
        "np2",
        "eta_sq",
        "pct_var",
    ]
    return aov[col_order]


def format_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format ANOVA table for clean terminal display with adaptive precision.

    Parameters
    ----------
    df : pd.DataFrame
        Input ANOVA results DataFrame.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with numeric columns string-formatted.
    """
    df_disp = df.copy()

    def _fmt(val: Any) -> str:
        if pd.isna(val):
            return ""
        if isinstance(val, (int, np.integer)):
            return str(val)
        if isinstance(val, (float, np.floating)):
            if 0 < abs(val) < 0.0001:
                return f"{val:.2e}"
            return f"{val:.4f}"
        return str(val)

    for col in [
        "SS",
        "MS",
        "F",
        "p_unc",
        "p_bonf",
        "p_fdr",
        "np2",
        "eta_sq",
        "pct_var",
    ]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(_fmt)

    if "DF" in df_disp.columns:
        df_disp["DF"] = df_disp["DF"].astype(int)

    return df_disp


def extract_variance_record(
    res_df: pd.DataFrame,
    label: str,
    factors: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Extract variance decomposition percentages from an ANOVA result DataFrame.

    Parameters
    ----------
    res_df : pd.DataFrame
        ANOVA result table for a single metric.
    label : str
        Display label for the metric row.
    factors : Optional[Sequence[str]], default=None
        List of active factors (defaults to ALL_FACTORS).

    Returns
    -------
    dict[str, Any]
        Dictionary with percentage explained variance for each term.
    """
    if factors is None:
        factors = list(ALL_FACTORS)

    term_map = dict(zip(res_df["Source"], res_df["pct_var"]))
    rec: dict[str, Any] = {"Metric": label}

    for factor in DISPLAY_FACTOR_ORDER:
        if factor in factors:
            disp_col = FACTOR_DISPLAY_NAMES[factor]
            rec[disp_col] = term_map.get(factor, 0.0)

    two_way_terms = [k for k in term_map if k.count(":") == 1]
    if two_way_terms:
        rec["% Var (2-Way Inter.)"] = sum(term_map[k] for k in two_way_terms)

    three_way_terms = [k for k in term_map if k.count(":") == 2]
    if three_way_terms:
        rec["% Var (3-Way Inter.)"] = sum(term_map[k] for k in three_way_terms)

    rec["% Var (Residual)"] = term_map.get("Residual (pooled)", 0.0)
    return rec


def run_variance_analysis(
    data_path: Union[str, Path],
    human_data_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    loss_fns: Optional[Sequence[str]] = None,
    pool_factors: Optional[Sequence[str]] = None,
    max_interaction: int = 2,
) -> dict[str, pd.DataFrame]:
    """Run interaction-pooled ANOVA and variance decomposition for loss functions & human ratings.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to audio distances TSV file.
    human_data_path : Optional[Union[str, Path]], default=None
        Path to postprocessed human listening responses TSV file.
    output_path : Optional[Union[str, Path]], default=None
        Path to save output results TSV (automatically appends '_pooled' if factors are pooled).
    loss_fns : Optional[Sequence[str]], default=None
        Subset of loss functions to evaluate (default: all present in dataset).
    pool_factors : Optional[Sequence[str]], default=None
        Factor(s) to pool ratings / distances across before ANOVA computation.
    max_interaction : int, default=2
        Maximum interaction order to evaluate (default: 2).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping entity names ('human', loss function names) to their ANOVA DataFrames.

    Raises
    ------
    FileNotFoundError
        If data_path does not exist.
    """
    in_data_path = Path(data_path).expanduser().resolve()
    if not in_data_path.exists():
        raise FileNotFoundError(f"Distance data file not found at: {in_data_path}")

    normalized_pool = normalize_pool_factors(pool_factors)
    remaining_factors = [f for f in ALL_FACTORS if f not in normalized_pool]
    effective_max_interaction = min(max_interaction, len(remaining_factors) - 1)

    if normalized_pool:
        log.info(
            f"Pooling ratings/distances across factor(s): {normalized_pool}. "
            f"Remaining ANOVA factors: {remaining_factors}"
        )
        if max_interaction > effective_max_interaction:
            log.info(
                f"Clamping max_interaction from {max_interaction} to {effective_max_interaction} "
                f"to ensure residual degrees of freedom."
            )

    log.info(f"Loading distance data from: {in_data_path}")
    sep = "\t" if in_data_path.suffix in [".tsv", ".txt"] else ","
    df = pd.read_csv(in_data_path, sep=sep)

    all_losses = sorted(df["loss_fn"].unique())
    if loss_fns:
        selected_losses = [l for l in loss_fns if l in all_losses]
        missing = set(loss_fns) - set(selected_losses)
        if missing:
            log.warning(f"Requested loss functions not found in data: {missing}")
    else:
        selected_losses = all_losses

    log.info(
        f"Analyzing {len(selected_losses)} loss functions with max_interaction={effective_max_interaction} "
        f"across factors: {remaining_factors}..."
    )

    results_by_loss: dict[str, pd.DataFrame] = {}

    # 1. Process human listening study benchmark if available
    human_record: Optional[dict[str, Any]] = None
    if human_data_path:
        h_path = Path(human_data_path).expanduser().resolve()
        if h_path.exists():
            log.info(f"Loading human listening benchmark data from: {h_path}")
            df_human = load_human_data(h_path)
            df_human_pooled = pool_data(df_human, remaining_factors)
            human_res = compute_interaction_pooled_anova(
                df_human_pooled,
                factors=remaining_factors,
                max_interaction=effective_max_interaction,
                dv="distance",
            )
            results_by_loss["human"] = human_res
            human_record = extract_variance_record(
                human_res,
                label="Human Listeners (Reference)",
                factors=remaining_factors,
            )

    # 2. Process algorithmic loss functions
    loss_summary_records = []
    for loss in selected_losses:
        sub = prepare_loss_data(df, loss)
        sub_pooled = pool_data(sub, remaining_factors)
        res_df = compute_interaction_pooled_anova(
            sub_pooled,
            factors=remaining_factors,
            max_interaction=effective_max_interaction,
            dv="distance",
        )
        results_by_loss[loss] = res_df
        loss_summary_records.append(
            extract_variance_record(
                res_df,
                label=loss,
                factors=remaining_factors,
            )
        )

    if normalized_pool:
        header_title = (
            f"INTERACTION-POOLED ANOVA & VARIANCE DECOMPOSITION "
            f"(POOLED OVER: {', '.join(normalized_pool).upper()} | "
            f"REMAINING: {', '.join(remaining_factors).upper()} | "
            f"Pooled Order: >{effective_max_interaction}-Way)"
        )
    else:
        header_title = (
            f"INTERACTION-POOLED ANOVA & VARIANCE DECOMPOSITION "
            f"(Pooled Order: >{effective_max_interaction}-Way)"
        )

    sep_bar = "=" * max(135, len(header_title))
    log.info("\n" + sep_bar)
    log.info(header_title)
    log.info(sep_bar)

    # Print human benchmark detailed ANOVA first if present
    if "human" in results_by_loss:
        log.info("\n--- Reference Metric: Human Listeners (Mean Rating Scores) ---")
        disp_human = format_table_for_display(results_by_loss["human"])
        log.info("\n" + disp_human.to_string(index=False))

    # Print loss functions detailed ANOVA
    for loss in selected_losses:
        log.info(f"\n--- Loss Function: {loss} ---")
        disp_df = format_table_for_display(results_by_loss[loss])
        log.info("\n" + disp_df.to_string(index=False))

    # Summary table: Sort loss functions by modulation amount sensitivity (% Var Amount) if present, else first metric
    summary_df = pd.DataFrame(loss_summary_records)
    sort_col = (
        "% Var (Amount)"
        if "% Var (Amount)" in summary_df.columns
        else [c for c in summary_df.columns if c != "Metric"][0]
    )
    summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(
        drop=True
    )

    log.info("\n" + sep_bar)
    log.info("SUMMARY: VARIANCE DECOMPOSITION PROFILES (% OF TOTAL VARIANCE EXPLAINED)")
    log.info(sep_bar)

    # Combine human reference at top followed by loss functions
    all_summary_rows = []
    if human_record:
        all_summary_rows.append(human_record)
    all_summary_rows.extend(summary_df.to_dict("records"))

    combined_summary_df = pd.DataFrame(all_summary_rows)
    disp_summary = combined_summary_df.copy()
    for col in disp_summary.columns[1:]:
        disp_summary[col] = disp_summary[col].apply(lambda x: f"{x:.2f}%")

    log.info("\n" + disp_summary.to_string(index=False))
    log.info(sep_bar + "\n")

    # 3. Export full results TSV (including human reference if present)
    if output_path:
        out_p = Path(output_path).expanduser().resolve()
        if normalized_pool and not out_p.stem.endswith("_pooled"):
            out_p = out_p.with_name(f"{out_p.stem}_pooled{out_p.suffix}")

        out_p.parent.mkdir(parents=True, exist_ok=True)
        export_records = []
        # Include human first
        if "human" in results_by_loss:
            df_h = results_by_loss["human"].copy()
            df_h.insert(0, "loss_fn", "human")
            export_records.append(df_h)
        for loss in selected_losses:
            df_copy = results_by_loss[loss].copy()
            df_copy.insert(0, "loss_fn", loss)
            export_records.append(df_copy)

        combined_df = pd.concat(export_records, ignore_index=True)
        combined_df.to_csv(out_p, sep="\t", index=False)
        log.info(f"Successfully saved detailed ANOVA & variance results to: {out_p}")

    return results_by_loss


def main() -> None:
    """Command-line interface for running interaction-pooled ANOVA variance decomposition."""
    parser = argparse.ArgumentParser(
        description="Interaction-pooled ANOVA and variance decomposition for audio loss functions & human ratings."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=os.path.join(OUT_DIR, "audio_distances.tsv"),
        help=f"Path to distances TSV file (default: {OUT_DIR}/audio_distances.tsv)",
    )
    parser.add_argument(
        "--human-data",
        default=os.path.join(OUT_DIR, "listening_test_responses_postprocessed.tsv"),
        help=f"Path to postprocessed human listening responses (default: {OUT_DIR}/listening_test_responses_postprocessed.tsv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(OUT_DIR, "variance_decomposition.tsv"),
        help=f"Path to save output results TSV (default: {OUT_DIR}/variance_decomposition.tsv; appends '_pooled' if factors are pooled)",
    )
    parser.add_argument(
        "--loss-fn",
        "--loss-fns",
        "-l",
        nargs="+",
        default=None,
        help="Filter analysis to specific loss function(s) (e.g. -l mfcc mss_log_lin)",
    )
    parser.add_argument(
        "--pool-factors",
        "--pool",
        "--pool-over",
        nargs="+",
        default=None,
        # default=["source", "timbre"],
        help=(
            "Factor(s) to pool ratings / distances across before computing variance analysis "
            "(choices: 'feature', 'source', 'modulation', 'rating_stimulus'; aliases: 'amount', 'mod', 'feat', 'src'). "
            "For example: --pool-factors feature source"
        ),
    )
    parser.add_argument(
        "--max-interaction",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Maximum interaction order to include (default: 2; automatically clamped if fewer factors remain)",
    )
    args = parser.parse_args()

    run_variance_analysis(
        data_path=Path(args.data_path),
        human_data_path=Path(args.human_data) if args.human_data else None,
        output_path=Path(args.output) if args.output else None,
        loss_fns=args.loss_fn,
        pool_factors=args.pool_factors,
        max_interaction=args.max_interaction,
    )


if __name__ == "__main__":
    main()
