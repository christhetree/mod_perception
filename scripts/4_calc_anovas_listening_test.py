"""ANOVA analysis for MUSHRA listening test data.

Computes repeated-measures ANOVAs and post-hoc pairwise tests using pingouin and statsmodels.
Assumes the input TSV/CSV dataset has already been cleaned and quality-filtered.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.formula._manager import FormulaManager
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.anova import _not_slice, _ssr_reduced_model

from paths import OUT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Configure pandas terminal output options for cleanly aligned DataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", None)

__all__ = [
    "get_asterisks",
    "add_significance_column",
    "add_pairwise_means_and_diff",
    "compute_rm_anova_with_effect_sizes",
    "compute_anova_4way_unpooled",
    "compute_anova_4way_pooled",
    "compute_anova_3way",
    "compute_anova_2way_modulation_feature",
    "compute_anova_2way_modulation_amount",
    "compute_pairwise_posthocs",
    "compute_simple_effects",
    "compute_normality_tests",
    "prepare_anova_dataframe",
    "run_all_anovas",
    "main",
]


def get_asterisks(p_val: Optional[float]) -> str:
    """Return significance asterisks based on p-value.

    Parameters
    ----------
    p_val : Optional[float]
        The p-value to evaluate.

    Returns
    -------
    str
        Significance marker:
        - '***' for p < 0.001
        - '**' for p < 0.01
        - '*' for p < 0.05
        - '' for p >= 0.05 or NaN/None.
    """
    if p_val is None or pd.isna(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return ""


def add_significance_column(
    df: pd.DataFrame,
    p_col: Optional[str] = None,
    col_name: str = "sig",
) -> pd.DataFrame:
    """Add a significance column with asterisks based on p-values.

    Prefers corrected p-values ('p_corr', 'p-corr', 'p_GG_corr', 'p_adj', 'p_adjust', 'p_bonf'),
    falling back to uncorrected p-values ('p_unc', 'p-unc', 'p') if no corrected column exists.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing statistical test results.
    p_col : Optional[str], default=None
        Name of p-value column to evaluate. If None, auto-detects from standard column names.
    col_name : str, default='sig'
        Name of the new significance column to insert.

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with the significance column inserted.
    """
    df = df.copy()
    if p_col is None:
        candidate_cols = [
            "p_corr",
            "p-corr",
            "p_GG_corr",
            "p_adj",
            "p_adjust",
            "p_bonf",
            "p_unc",
            "p-unc",
            "p",
        ]
        for c in candidate_cols:
            if c in df.columns:
                p_col = c
                break

    if p_col is not None and p_col in df.columns:
        sig_series = df[p_col].apply(get_asterisks)
        loc = df.columns.get_loc(p_col) + 1
        df.insert(loc, col_name, sig_series)
    return df


def add_pairwise_means_and_diff(
    pw_df: pd.DataFrame,
    data: pd.DataFrame,
    dv: str,
    within: str,
) -> pd.DataFrame:
    """Add mean(A), mean(B), and diff (mean(A) - mean(B)) columns to a pairwise test DataFrame.

    Parameters
    ----------
    pw_df : pd.DataFrame
        Pairwise comparison test results containing columns 'A' and 'B'.
    data : pd.DataFrame
        Source data containing the dependent variable and condition factors.
    dv : str
        Name of the dependent variable column.
    within : str
        Factor column name that indexes conditions 'A' and 'B'.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'mean(A)', 'mean(B)', and 'diff' columns inserted after column 'B'.
    """
    pw_df = pw_df.copy()
    means = data.groupby(within)[dv].mean().to_dict()
    mean_a = pw_df["A"].map(means)
    mean_b = pw_df["B"].map(means)
    diff = mean_a - mean_b

    # Insert right after column 'B'
    b_loc = pw_df.columns.get_loc("B") + 1
    pw_df.insert(b_loc, "mean(A)", mean_a)
    pw_df.insert(b_loc + 1, "mean(B)", mean_b)
    pw_df.insert(b_loc + 2, "diff", diff)
    return pw_df


def compute_rm_anova_with_effect_sizes(
    df: pd.DataFrame,
    dv: str,
    subject: str,
    within: Sequence[str],
) -> pd.DataFrame:
    """Compute Repeated-Measures ANOVA for N within-subject factors with effect sizes.

    Calculates:
    - Sums of Squares (SS) and Mean Squares (MS) for effects and errors.
    - F-statistic and uncorrected p-value (p_unc).
    - Partial eta-squared: np2 = SS_effect / (SS_effect + SS_error).
    - Generalized eta-squared: ng2 = SS_effect / (SS_effect + SS_subject + sum(SS_errors))
      following Olejnik & Algina (2003) and Bakeman (2005) for fully within-subject designs.

    Parameters
    ----------
    df : pd.DataFrame
        Input data aggregated to subject cell means.
    dv : str
        Dependent variable column name.
    subject : str
        Subject identifier column name.
    within : Sequence[str]
        List of within-subject factor column names.

    Returns
    -------
    pd.DataFrame
        ANOVA summary table with columns: Source, SS, DF1, DF2, MS, F, p_unc, np2, ng2, sig.

    Raises
    ------
    ValueError
        If independent variables are collinear.
    """
    y = df[dv].values

    # Construct OLS endog and exog from string using patsy sum-to-zero contrasts
    within_terms = [f"C({i}, Sum)" for i in within]
    subject_term = f"C({subject}, Sum)"
    factors = [*within_terms, subject_term]
    mgr = FormulaManager()
    x = mgr.get_matrices("*".join(factors), data=df, pandas=False)
    term_slices = mgr.get_term_name_slices(x)
    for key in term_slices:
        ind = np.array([False] * x.shape[1])
        ind[term_slices[key]] = True
        term_slices[key] = np.array(ind)
    term_exclude = [":".join(factors)]
    ind = _not_slice(term_slices, term_exclude, x.shape[1])
    x = x[:, ind]

    # Fit full OLS model
    model = OLS(y, x)
    results = model.fit()
    if model.rank < x.shape[1]:
        raise ValueError("Independent variables are collinear.")
    for i in term_exclude:
        term_slices.pop(i)
    for key in term_slices:
        term_slices[key] = term_slices[key][ind]
    params = results.params
    df_resid = results.df_resid
    ssr = results.ssr

    # Calculate Subject Sum of Squares
    subj_key = subject_term
    ssr_subj, _ = _ssr_reduced_model(y, x, term_slices, params, [subj_key])
    ss_subject = ssr_subj - ssr

    records = []
    all_ss_error = 0.0
    for key in term_slices:
        if subject not in str(key) and str(key) not in ("Intercept", "1"):
            ssr1, df_resid1 = _ssr_reduced_model(y, x, term_slices, params, [key])
            df1 = df_resid1 - df_resid
            ss_effect = ssr1 - ssr
            msm = ss_effect / df1

            err_key = str(key) + ":" + subject_term
            if err_key in term_slices:
                ssr_err, df_err_res = _ssr_reduced_model(
                    y, x, term_slices, params, [err_key]
                )
                df2 = df_err_res - df_resid
                ss_error = ssr_err - ssr
                mse = ss_error / df2
            else:
                df2 = df_resid
                ss_error = ssr
                mse = ssr / df_resid

            all_ss_error += ss_error
            F = msm / mse if mse > 0 else np.nan
            p = stats.f.sf(F, df1, df2) if not np.isnan(F) else np.nan
            term = str(key).replace("C(", "").replace(", Sum)", "")
            records.append(
                {
                    "Source": term,
                    "SS": ss_effect,
                    "DF1": df1,
                    "DF2": df2,
                    "MS": msm,
                    "F": F,
                    "p_unc": p,
                    "ss_error": ss_error,
                }
            )

    table = pd.DataFrame(records)
    # Partial eta-squared: np2 = (F * DF1) / (F * DF1 + DF2)
    table["np2"] = (table["F"] * table["DF1"]) / (
        table["F"] * table["DF1"] + table["DF2"]
    )
    # Generalized eta-squared: ng2 = SS_effect / (SS_effect + SS_subject + sum(all_error_SS))
    denom_ges = ss_subject + all_ss_error
    table["ng2"] = table["SS"] / (table["SS"] + denom_ges)

    col_order = ["Source", "SS", "DF1", "DF2", "MS", "F", "p_unc", "np2", "ng2"]
    return add_significance_column(table[col_order])


def _resolve_input_col(
    df: pd.DataFrame, dv: str, input_col: Optional[str] = None
) -> str:
    """Resolve the column name to aggregate or measure.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dv : str
        Target dependent variable column name.
    input_col : Optional[str], default=None
        Explicit input column override.

    Returns
    -------
    str
        Resolved column name.

    Raises
    ------
    KeyError
        If neither `input_col`, 'rating_score', nor `dv` is found in `df`.
    """
    if input_col is not None:
        if input_col not in df.columns:
            raise KeyError(
                f"Specified input column '{input_col}' not found in DataFrame."
            )
        return input_col
    if "rating_score" in df.columns:
        return "rating_score"
    if dv in df.columns:
        return dv
    raise KeyError(
        f"Neither default input column 'rating_score' nor dv '{dv}' found in DataFrame columns: {list(df.columns)}"
    )


def _filter_balanced_subjects(
    df: pd.DataFrame, subject_col: str, group_cols: Sequence[str], dv: str
) -> pd.DataFrame:
    """Filter dataset to subjects with complete and balanced data across all within-subject cells.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of cell means or raw observations.
    subject_col : str
        Subject identifier column.
    group_cols : Sequence[str]
        Factor column names forming the factorial design cells.
    dv : str
        Dependent variable column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame retaining only subjects with exactly 1 observation per factorial cell.
    """
    df_valid = df.dropna(subset=[*group_cols, dv])
    if df_valid.empty:
        return df_valid.copy()

    # Total expected cells across the full factorial combinations of group_cols
    expected_cells = int(np.prod([df_valid[col].nunique() for col in group_cols]))

    # Count observations per (subject, *group_cols) cell
    cell_obs = df_valid.groupby([subject_col, *group_cols])[dv].count()

    # A subject is balanced if:
    # 1. Exactly expected_cells unique combinations of group_cols are present
    # 2. Every combination has exactly 1 observation (no duplicate combinations)
    cell_sizes = cell_obs.groupby(level=subject_col).size()
    cell_all_ones = (cell_obs == 1).groupby(level=subject_col).all()

    balanced_mask = (cell_sizes == expected_cells) & cell_all_ones
    complete_subjects = balanced_mask[balanced_mask].index

    all_subjects = df[subject_col].dropna().unique()
    if len(complete_subjects) < len(all_subjects):
        log.info(
            f"Using {len(complete_subjects)} of {len(all_subjects)} subjects "
            f"with complete data for balanced design across {group_cols}."
        )
    return df_valid[df_valid[subject_col].isin(complete_subjects)].copy()


def _prepare_balanced_cell_means(
    df: pd.DataFrame,
    subject: str,
    within: Sequence[str],
    dv: str = "mean_rating",
    input_col: Optional[str] = None,
    drop_na_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate ratings to cell means per subject and filter to balanced complete cases.

    Parameters
    ----------
    df : pd.DataFrame
        Input observations DataFrame.
    subject : str
        Subject column name.
    within : Sequence[str]
        List of within-subject factor column names.
    dv : str, default='mean_rating'
        Output dependent variable column name.
    input_col : Optional[str], default=None
        Input column to average.
    drop_na_cols : Optional[Sequence[str]], default=None
        Columns to check for missing values before aggregation.

    Returns
    -------
    pd.DataFrame
        Aggregated and balanced cell means DataFrame.
    """
    in_col = _resolve_input_col(df, dv, input_col)
    subset_df = df.dropna(subset=drop_na_cols) if drop_na_cols else df
    cell_means = (
        subset_df.groupby([subject, *within], as_index=False)[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )
    return _filter_balanced_subjects(cell_means, subject, within, dv)


def _run_single_pairwise_posthoc(
    df: pd.DataFrame,
    within: str,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
    drop_reference: bool = False,
) -> pd.DataFrame:
    """Run pairwise t-tests with Bonferroni correction on balanced cell means for a single factor.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    within : str
        Within-subject factor column name.
    dv : str, default='mean_rating'
        Dependent variable column name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Input rating column name to aggregate.
    drop_reference : bool, default=False
        Whether to exclude rows where `within == 'reference'`.

    Returns
    -------
    pd.DataFrame
        Pairwise comparison table with Bonferroni-adjusted p-values and effect sizes.
    """
    sub_df = (
        df[df[within] != "reference"].copy()
        if drop_reference and "reference" in df[within].values
        else df
    )
    cell_means = _prepare_balanced_cell_means(sub_df, subject, [within], dv, input_col)
    pw = pg.pairwise_tests(
        data=cell_means,
        dv=dv,
        within=within,
        subject=subject,
        padjust="bonf",
    )
    pw = add_pairwise_means_and_diff(pw, cell_means, dv, within)
    return add_significance_column(pw)


def _run_sliced_pairwise_tests(
    df: pd.DataFrame,
    dv: str,
    within: str,
    slice_col: str,
    subject: str,
    padjust: str = "bonf",
) -> pd.DataFrame:
    """Run pairwise tests on slices of a DataFrame, adding slice identifier, means, diffs, and sig.

    Parameters
    ----------
    df : pd.DataFrame
        Input data table.
    dv : str
        Dependent variable column name.
    within : str
        Within-subject factor to compare.
    slice_col : str
        Grouping column to slice the dataset on.
    subject : str
        Subject identifier column name.
    padjust : str, default='bonf'
        P-value correction method (e.g. 'bonf', 'holm', 'fdr_bh').

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all sliced pairwise comparisons.
    """
    slices = []
    for val in sorted(df[slice_col].unique()):
        slice_df = df[df[slice_col] == val]
        pw = pg.pairwise_tests(
            data=slice_df,
            dv=dv,
            within=within,
            subject=subject,
            padjust=padjust,
        )
        pw = add_pairwise_means_and_diff(pw, slice_df, dv, within)
        pw.insert(0, slice_col, val)
        slices.append(pw)
    if not slices:
        return pd.DataFrame()
    res = pd.concat(slices, ignore_index=True)
    if "p_corr" not in res.columns and "p_unc" in res.columns:
        p_corr_idx = res.columns.get_loc("p_unc") + 1
        res.insert(p_corr_idx, "p_corr", np.minimum(1.0, res["p_unc"] * len(res)))
        res.insert(p_corr_idx + 1, "p_adjust", padjust)
    return add_significance_column(res)


def compute_anova_4way_unpooled(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> pd.DataFrame:
    """4-way Repeated-Measures ANOVA (unpooled): modulation x feature x source x rating_stimulus (4 amounts).

    Uses all 4 individual stimulus condition ratings directly (DF1=3 for amount), without
    dichotomizing into Low vs High groups.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    pd.DataFrame
        Repeated-measures ANOVA results table.
    """
    log.info(
        "Computing 4-way Repeated-Measures ANOVA (unpooled: modulation x feature x source x rating_stimulus)..."
    )
    within = ["modulation", "feature", "source", "rating_stimulus"]
    mod_avg_bal = _prepare_balanced_cell_means(df, subject, within, dv, input_col)

    # Save intermediate balanced data to .tsv if needed
    mod_avg_bal.to_csv(
        os.path.join(OUT_DIR, "anova_4way_unpooled_balanced_data.tsv"),
        sep="\t",
        index=False,
    )

    return compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )


def compute_anova_4way_pooled(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> pd.DataFrame:
    """4-way Repeated-Measures ANOVA (pooled): modulation x feature x source x amount_group (Low vs High).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    pd.DataFrame
        Repeated-measures ANOVA results table.
    """
    log.info(
        "Computing 4-way Repeated-Measures ANOVA (pooled: modulation x feature x source x amount_group)..."
    )
    within = ["modulation", "feature", "source", "amount_group"]
    mod_avg_bal = _prepare_balanced_cell_means(
        df, subject, within, dv, input_col, drop_na_cols=["amount_group"]
    )

    return compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )


def compute_anova_3way(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> pd.DataFrame:
    """3-way Repeated-Measures ANOVA: modulation x feature x source.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    pd.DataFrame
        Repeated-measures ANOVA results table.
    """
    log.info(
        "Computing 3-way Repeated-Measures ANOVA (modulation x feature x source)..."
    )
    within = ["modulation", "feature", "source"]
    mod_avg_bal = _prepare_balanced_cell_means(df, subject, within, dv, input_col)

    return compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )


def compute_anova_2way_modulation_feature(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """2-way Repeated-Measures ANOVA: modulation x feature.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (pingouin_anova_result, statsmodels_anova_result).
    """
    log.info("Computing 2-way Repeated-Measures ANOVA (modulation x feature)...")
    within = ["modulation", "feature"]
    mod_avg_bal = _prepare_balanced_cell_means(df, subject, within, dv, input_col)

    aov_pg = add_significance_column(
        pg.rm_anova(
            data=mod_avg_bal,
            dv=dv,
            within=within,
            subject=subject,
            detailed=True,
        )
    )
    aov_helper = compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )
    return aov_pg, aov_helper


def compute_anova_2way_modulation_amount(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """2-way Repeated-Measures ANOVA: modulation x amount_group (Low vs High).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (pingouin_anova_result, statsmodels_anova_result).
    """
    log.info("Computing 2-way Repeated-Measures ANOVA (modulation x amount_group)...")
    within = ["modulation", "amount_group"]
    mod_avg_bal = _prepare_balanced_cell_means(
        df, subject, within, dv, input_col, drop_na_cols=["amount_group"]
    )

    aov_pg = add_significance_column(
        pg.rm_anova(
            data=mod_avg_bal,
            dv=dv,
            within=within,
            subject=subject,
            detailed=True,
        )
    )
    aov_helper = compute_rm_anova_with_effect_sizes(
        mod_avg_bal, dv=dv, subject=subject, within=within
    )
    return aov_pg, aov_helper


def compute_pairwise_posthocs(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Pairwise post-hoc paired t-tests with Bonferroni correction across main factors.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping factor names ('modulation', 'feature', 'amount') to pairwise comparison DataFrames.
    """
    log.info("Computing pairwise post-hoc tests (Bonferroni adjusted)...")
    in_col = _resolve_input_col(df, dv, input_col)

    results = {
        "modulation": _run_single_pairwise_posthoc(
            df, "modulation", dv=dv, subject=subject, input_col=in_col
        ),
        "feature": _run_single_pairwise_posthoc(
            df, "feature", dv=dv, subject=subject, input_col=in_col
        ),
    }

    amount_col = (
        "rating_stimulus"
        if "rating_stimulus" in df.columns
        else ("amount" if "amount" in df.columns else None)
    )
    if amount_col is not None:
        pw_amount = _run_single_pairwise_posthoc(
            df,
            amount_col,
            dv=dv,
            subject=subject,
            input_col=in_col,
            drop_reference=True,
        )
        results["amount"] = pw_amount
        results["rating_stimulus"] = pw_amount

    return results


def compute_simple_effects(
    df: pd.DataFrame,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Break down significant two-way interactions via simple main effects analysis.

    Decomposes:
    1. feature:source interaction:
       - Pairwise comparisons of source (real vs synthetic) within each feature level (brightness, richness, warmth).
    2. modulation:rating_stimulus interaction:
       - Pairwise comparisons of modulation (amp vs freq vs reg) within each rating_stimulus condition.
       - Pairwise comparisons across rating_stimulus conditions within each modulation type.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping interaction contrast names to DataFrames of Bonferroni-adjusted pairwise comparisons.
    """
    log.info(
        "Computing simple effects for significant two-way interactions (Bonferroni adjusted)..."
    )
    in_col = _resolve_input_col(df, dv, input_col)
    results = {}

    # 1. Feature x Source interaction breakdown
    if "feature" in df.columns and "source" in df.columns:
        sub_fs_bal = _prepare_balanced_cell_means(
            df, subject, ["feature", "source"], dv=dv, input_col=in_col
        )
        results["source_within_feature"] = _run_sliced_pairwise_tests(
            sub_fs_bal, dv=dv, within="source", slice_col="feature", subject=subject
        )

    # 2. Modulation x Rating Stimulus interaction breakdown
    amount_col = (
        "rating_stimulus"
        if "rating_stimulus" in df.columns
        else ("amount" if "amount" in df.columns else None)
    )
    if amount_col is not None and "modulation" in df.columns:
        df_amount = (
            df[df[amount_col] != "reference"].copy()
            if "reference" in df[amount_col].values
            else df
        )
        sub_mr_bal = _prepare_balanced_cell_means(
            df_amount, subject, ["modulation", amount_col], dv=dv, input_col=in_col
        )
        results["modulation_within_stimulus"] = _run_sliced_pairwise_tests(
            sub_mr_bal,
            dv=dv,
            within="modulation",
            slice_col=amount_col,
            subject=subject,
        )
        results["stimulus_within_modulation"] = _run_sliced_pairwise_tests(
            sub_mr_bal,
            dv=dv,
            within=amount_col,
            slice_col="modulation",
            subject=subject,
        )

    return results


def compute_normality_tests(
    df: pd.DataFrame,
    group_by: Optional[Sequence[str]] = None,
    dv: str = "mean_rating",
    subject: str = "session_uuid",
    input_col: Optional[str] = None,
) -> pd.DataFrame:
    """Shapiro-Wilk test for normality across factor combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared MUSHRA listening test DataFrame.
    group_by : Optional[Sequence[str]], default=None
        Factors to group by (default: ['modulation', 'feature', 'source']).
    dv : str, default='mean_rating'
        Dependent variable name.
    subject : str, default='session_uuid'
        Subject identifier column name.
    input_col : Optional[str], default=None
        Raw rating input column to aggregate.

    Returns
    -------
    pd.DataFrame
        Normality test statistics per factor cell.
    """
    if group_by is None:
        group_by = ["modulation", "feature", "source"]

    log.info(f"Computing Shapiro-Wilk normality tests grouped by {group_by}...")
    in_col = _resolve_input_col(df, dv, input_col)
    grouped = (
        df.groupby([subject, *group_by], as_index=False)[in_col]
        .mean()
        .rename(columns={in_col: dv})
    )

    records = []
    for keys, group in grouped.groupby(list(group_by)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ratings = group[dv].dropna()
        if len(ratings) >= 3:
            w_stat, p_val = stats.shapiro(ratings)
        else:
            w_stat, p_val = float("nan"), float("nan")

        rec = dict(zip(group_by, keys))
        rec["n"] = len(ratings)
        rec["W"] = w_stat
        rec["p"] = p_val
        rec["normality"] = "Normal" if p_val > 0.05 else "Non-normal"
        records.append(rec)

    return add_significance_column(pd.DataFrame(records))


def prepare_anova_dataframe(
    data_source: Union[str, Path, pd.DataFrame],
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Load and prepare MUSHRA listening test data for ANOVA analysis.

    - Loads TSV or CSV from file path if a string/Path is given.
    - Excludes reference stimulus rows if exclude_reference is True.
    - Extracts factor columns 'modulation', 'feature', 'source' from 'trial_id' if missing.
    - Maps conditions to 'amount_group' (Low: condition_a/b, High: condition_c/d) if missing.

    Parameters
    ----------
    data_source : Union[str, Path, pd.DataFrame]
        Input file path or DataFrame.
    exclude_reference : bool, default=True
        Whether to exclude reference stimulus rows.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for ANOVA factorial analysis.
    """
    if isinstance(data_source, (str, Path)):
        file_path = Path(data_source).expanduser().resolve()
        log.info(f"Loading prepared data from {file_path}")
        sep = "\t" if file_path.suffix == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = data_source.copy()

    # Exclude reference stimulus ratings for ANOVA if present
    if exclude_reference and "rating_stimulus" in df.columns:
        df = df[df["rating_stimulus"] != "reference"].copy()

    # Separate trial_id into modulation, feature, source if not already present
    if "trial_id" in df.columns and "modulation" not in df.columns:
        split_cols = df["trial_id"].str.split("_", expand=True)
        if split_cols.shape[1] >= 3:
            df["modulation"] = split_cols[0]
            df["feature"] = split_cols[1]
            df["source"] = split_cols[2]

    # Map conditions to amount_group: Low (conditions a/b) vs High (conditions c/d) if not already present
    if "amount_group" not in df.columns and "rating_stimulus" in df.columns:
        cond_map = {
            "condition_a": "Low",
            "condition_b": "Low",
            "condition_c": "High",
            "condition_d": "High",
        }
        df["amount_group"] = df["rating_stimulus"].map(cond_map)

    return df


def run_all_anovas(
    data_source: Union[str, Path, pd.DataFrame],
    dv: str = "mean_rating",
    input_col: Optional[str] = None,
    include_lower_order: bool = False,
) -> Dict[str, Any]:
    """Execute ANOVA analyses and print summaries to console.

    Assumes the input TSV/CSV dataset has already been cleaned and quality-filtered.
    Extracts within-subject factors (modulation, feature, source, amount_group) if not already present.

    Parameters
    ----------
    data_source : Union[str, Path, pd.DataFrame]
        Input data file path or DataFrame.
    dv : str, default='mean_rating'
        Dependent variable name.
    input_col : Optional[str], default=None
        Raw rating input column name to aggregate.
    include_lower_order : bool, default=False
        Whether to also compute and display pooled 4-way, 3-way, and 2-way ANOVAs.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing computed ANOVA, post-hoc, and simple effects tables.
    """
    df = prepare_anova_dataframe(data_source, exclude_reference=True)
    results: Dict[str, Any] = {}

    log.info("\n" + "=" * 80)
    log.info(
        " 1. FOUR-WAY REPEATED MEASURES ANOVA (modulation x feature x source x rating_stimulus [UNPOOLED])"
    )
    log.info("=" * 80)
    aov_4way_unpooled = compute_anova_4way_unpooled(df, dv=dv, input_col=input_col)
    log.info("\n" + aov_4way_unpooled.to_string(index=False))
    results["anova_4way_unpooled"] = aov_4way_unpooled

    if include_lower_order:
        log.info("\n" + "=" * 80)
        log.info(
            " 2. FOUR-WAY REPEATED MEASURES ANOVA (modulation x feature x source x amount_group [POOLED])"
        )
        log.info("=" * 80)
        aov_4way_pooled = compute_anova_4way_pooled(df, dv=dv, input_col=input_col)
        log.info("\n" + aov_4way_pooled.to_string(index=False))
        results["anova_4way_pooled"] = aov_4way_pooled

        log.info("\n" + "=" * 80)
        log.info(" 3. THREE-WAY REPEATED MEASURES ANOVA (modulation x feature x source)")
        log.info("=" * 80)
        aov_3way = compute_anova_3way(df, dv=dv, input_col=input_col)
        log.info("\n" + aov_3way.to_string(index=False))
        results["anova_3way"] = aov_3way

        log.info("\n" + "=" * 80)
        log.info(" 4. TWO-WAY REPEATED MEASURES ANOVA (modulation x feature)")
        log.info("=" * 80)
        aov_2way_mf_pg, aov_2way_mf_helper = compute_anova_2way_modulation_feature(
            df, dv=dv, input_col=input_col
        )
        log.info(
            "\n[Pingouin (pg.rm_anova, detailed=True)]:\n"
            + aov_2way_mf_pg.to_string(index=False)
        )
        log.info(
            "\n[Helper Function (compute_rm_anova_with_effect_sizes)]:\n"
            + aov_2way_mf_helper.to_string(index=False)
        )
        results["anova_2way_modulation_feature"] = (aov_2way_mf_pg, aov_2way_mf_helper)

        log.info("\n" + "=" * 80)
        log.info(" 5. TWO-WAY REPEATED MEASURES ANOVA (modulation x amount_group)")
        log.info("=" * 80)
        aov_2way_ma_pg, aov_2way_ma_helper = compute_anova_2way_modulation_amount(
            df, dv=dv, input_col=input_col
        )
        log.info(
            "\n[Pingouin (pg.rm_anova, detailed=True)]:\n"
            + aov_2way_ma_pg.to_string(index=False)
        )
        log.info(
            "\n[Helper Function (compute_rm_anova_with_effect_sizes)]:\n"
            + aov_2way_ma_helper.to_string(index=False)
        )
        results["anova_2way_modulation_amount"] = (aov_2way_ma_pg, aov_2way_ma_helper)

    log.info("\n" + "=" * 80)
    log.info(" 6. POST-HOC PAIRWISE TESTS (BONFERRONI)")
    log.info("=" * 80)
    posthocs = compute_pairwise_posthocs(df, dv=dv, input_col=input_col)
    log.info(
        "\n[Post-hoc: Modulation]\n" + posthocs["modulation"].to_string(index=False)
    )
    log.info("\n[Post-hoc: Feature]\n" + posthocs["feature"].to_string(index=False))
    if "amount" in posthocs:
        log.info(
            "\n[Post-hoc: Amount (4 conditions)]\n"
            + posthocs["amount"].to_string(index=False)
        )
    results["posthocs"] = posthocs

    log.info("\n" + "=" * 80)
    log.info(" 7. SIMPLE EFFECTS ANALYSIS (BREAKDOWN OF 2-WAY INTERACTIONS)")
    log.info("=" * 80)
    simple_effects = compute_simple_effects(df, dv=dv, input_col=input_col)
    for name, res_df in simple_effects.items():
        log.info(f"\n[Simple Effects: {name}]\n" + res_df.to_string(index=False))
    results["simple_effects"] = simple_effects

    return results


def main() -> None:
    """Command-line entry point to execute ANOVA analyses on MUSHRA data."""
    parser = argparse.ArgumentParser(
        description="Run ANOVA analyses on prepared MUSHRA listening test data."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=os.path.join(OUT_DIR, "listening_test_responses_postprocessed.tsv"),
        help=f"Path to prepared MUSHRA data file (tsv or csv; default: {OUT_DIR}/listening_test_responses_postprocessed.tsv)",
    )
    parser.add_argument(
        "--dv",
        default="mean_rating",
        help="Dependent variable column name for ANOVA (default: mean_rating)",
    )
    parser.add_argument(
        "--input-col",
        default=None,
        help="Input rating column to aggregate (default: rating_score if present, else dv)",
    )
    parser.add_argument(
        "--include-lower-order",
        action="store_true",
        default=False,
        help="Also compute and display pooled 4-way, 3-way, and 2-way ANOVAs (default: False)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        log.error(f"Data file not found at: {args.data_path}")
        parser.print_help()
        sys.exit(1)

    run_all_anovas(
        args.data_path,
        dv=args.dv,
        input_col=args.input_col,
        include_lower_order=args.include_lower_order,
    )


if __name__ == "__main__":
    main()
