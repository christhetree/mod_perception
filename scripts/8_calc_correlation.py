"""Correlation analysis between audio distance functions and human MUSHRA listening test data.

Compares audio distance functions (MSE, MSS, MFCC, CLAP, PANNs, Wavelet Scattering, JTFS, etc.)
from audio distance datasets against human perceptual difference ratings from postprocessed MUSHRA responses.

Granularity Levels:
1. Entire dataset (all 18 trials, 72 non-reference stimuli per participant)
2. Per modulation type (frequency, amplitude, regularity; 6 trials, 24 stimuli each)
3. Per modulation x timbre combination (3 modulations x 3 timbres = 9 conditions; 2 trials, 8 stimuli each)

Metrics:
- Group Correlation:
    Correlation of model distance with the group mean human ratings across participants:
    - Pearson r (linear alignment) and two-sided p-value
    - Spearman rho (monotonic rank alignment) and two-sided p-value
    - Kendall tau (pairwise concordance) and two-sided p-value
- Individual Correlation:
    Correlation of model distance with each participant's individual ratings:
    - Mean and unbiased sample standard deviation (ddof=1) across participants.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from paths import OUT_DIR

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# Mapping of modulation type and physical amount to MUSHRA rating_stimulus
AMOUNT_TO_STIMULUS: dict[str, dict[float, str]] = {
    "amp": {
        0.10: "reference",
        0.30: "condition_a",
        0.50: "condition_b",
        0.70: "condition_c",
        0.90: "condition_d",
    },
    "freq": {
        0.25: "reference",
        0.50: "condition_a",
        1.00: "condition_b",
        2.00: "condition_c",
        4.00: "condition_d",
    },
    "reg": {
        0.000: "reference",
        0.125: "condition_a",
        0.250: "condition_b",
        0.375: "condition_c",
        0.500: "condition_d",
    },
}


def prepare_data(
    data_source: Union[str, Path, pd.DataFrame],
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Load and prepare MUSHRA TSV/CSV data for correlation analyses.

    Extracts `modulation`, `timbre` (feature), and `source` from `trial_id`
    (e.g., 'freq_brightness_real' -> modulation='freq', timbre='brightness', source='real').

    Parameters
    ----------
    data_source : Union[str, Path, pd.DataFrame]
        TSV/CSV file path or existing pandas DataFrame containing listening responses.
    exclude_reference : bool, default=True
        Whether to drop reference anchor ratings (typically rated 0 in difference MUSHRA).

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with added factorial columns: 'modulation', 'timbre',
        'source', and 'mod_timbre'.

    Raises
    ------
    FileNotFoundError
        If data_source path does not exist.
    ValueError
        If required columns are missing from the dataset.
    """
    if isinstance(data_source, (str, Path)):
        file_path = Path(data_source).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(
                f"Listening test response file not found: {file_path}"
            )
        log.info(f"Loading responses from {file_path}")
        sep = "\t" if file_path.suffix in [".tsv", ".txt"] else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = data_source.copy()

    required = ["session_uuid", "trial_id", "rating_stimulus", "rating_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    if exclude_reference:
        df = df[df["rating_stimulus"] != "reference"].copy()

    df = df[df["trial_id"] != "training"].copy()

    split_cols = df["trial_id"].str.split("_", expand=True)
    if split_cols.shape[1] >= 2:
        df["modulation"] = split_cols[0]
        df["timbre"] = split_cols[1]
    if split_cols.shape[1] >= 3:
        df["source"] = split_cols[2]

    df["mod_timbre"] = df["modulation"] + "_" + df["timbre"]

    return df


def prepare_distances(
    distances_source: Union[str, Path, pd.DataFrame],
    exclude_reference: bool = True,
) -> pd.DataFrame:
    """Load and parse audio distance data for correlation analysis.

    Extracts `trial_id` and `rating_stimulus` to match the MUSHRA response format:
    - `trial_id`: <modulation>_<timbre>_<source> (e.g. 'freq_brightness_real')
    - `rating_stimulus`: 'condition_a', 'condition_b', 'condition_c', 'condition_d', or 'reference'

    Parameters
    ----------
    distances_source : Union[str, Path, pd.DataFrame]
        TSV/CSV file path or DataFrame containing computed audio distance values.
    exclude_reference : bool, default=True
        Whether to drop reference stimulus comparisons.

    Returns
    -------
    pd.DataFrame
        Prepared distances DataFrame.

    Raises
    ------
    FileNotFoundError
        If distances_source path does not exist.
    ValueError
        If required columns are missing or amounts cannot be parsed.
    """
    if isinstance(distances_source, (str, Path)):
        file_path = Path(distances_source).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Distances file not found: {file_path}")
        log.info(f"Loading distances from {file_path}")
        sep = "\t" if file_path.suffix in [".tsv", ".txt"] else ","
        df = pd.read_csv(file_path, sep=sep)
    else:
        df = distances_source.copy()

    required = ["loss_fn", "wavetable", "mod_type", "amount", "distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in distances dataset: {missing}")

    # Extract timbre and source from wavetable name prefix
    extracted = df["wavetable"].str.extract(
        r"^(?P<timbre>brightness|richness|warmth)_(?P<source>real|synthetic)"
    )
    if extracted["timbre"].isna().any() or extracted["source"].isna().any():
        unmatched = df[extracted["timbre"].isna() | extracted["source"].isna()][
            "wavetable"
        ].unique()
        raise ValueError(f"Cannot parse timbre/source from wavetables: {unmatched}")

    df["timbre"] = extracted["timbre"]
    df["source"] = extracted["source"]
    df["modulation"] = df["mod_type"].astype(str)
    df["trial_id"] = df["modulation"] + "_" + df["timbre"] + "_" + df["source"]
    df["mod_timbre"] = df["modulation"] + "_" + df["timbre"]
    df["amount"] = df["amount"].astype(float)
    df["distance"] = df["distance"].astype(float)
    df["loss_fn"] = df["loss_fn"].astype(str)

    # Lookup mapping: (mod_type, rounded_amount) -> rating_stimulus
    lookup_map = {
        (mod, round(amt, 4)): stim
        for mod, amt_dict in AMOUNT_TO_STIMULUS.items()
        for amt, stim in amt_dict.items()
    }

    stimuli = [
        lookup_map.get((m, round(a, 4))) for m, a in zip(df["modulation"], df["amount"])
    ]
    if any(s is None for s in stimuli):
        unmatched_rows = [
            (m, a)
            for m, a, s in zip(df["modulation"], df["amount"], stimuli)
            if s is None
        ]
        raise ValueError(
            f"Unrecognized modulation amount mapping: {set(unmatched_rows)}"
        )

    df["rating_stimulus"] = stimuli
    df["is_reference"] = df.get("is_reference", False) | (
        df["rating_stimulus"] == "reference"
    )

    if exclude_reference:
        df = df[~df["is_reference"]].copy()

    columns = [
        "loss_fn",
        "trial_id",
        "rating_stimulus",
        "modulation",
        "timbre",
        "source",
        "mod_timbre",
        "amount",
        "is_reference",
        "distance",
    ]
    return df[columns]


def compute_slice_correlations(
    human_matrix: pd.DataFrame,
    dist_series: pd.Series,
) -> dict[str, float]:
    """Compute group-level and individual-level correlations between distances and human ratings.

    Parameters
    ----------
    human_matrix : pd.DataFrame
        Stimulus x subject rating DataFrame (index: ['trial_id', 'rating_stimulus']).
    dist_series : pd.Series
        Model distances aligned to human_matrix index.

    Returns
    -------
    dict[str, float]
        Dictionary containing Pearson, Spearman, and Kendall metrics with p-values
        and individual participant mean/std distributions.
    """
    n_stimuli, n_subjects = human_matrix.shape
    aligned_dist = dist_series.loc[human_matrix.index]

    if n_stimuli < 3 or n_subjects < 1 or aligned_dist.nunique() <= 1:
        return {
            "n_stimuli": n_stimuli,
            "n_subjects": n_subjects,
            "pearson_group": float("nan"),
            "pearson_group_p": float("nan"),
            "spearman_group": float("nan"),
            "spearman_group_p": float("nan"),
            "kendall_group": float("nan"),
            "kendall_group_p": float("nan"),
            "pearson_indiv": float("nan"),
            "pearson_indiv_std": float("nan"),
            "spearman_indiv": float("nan"),
            "spearman_indiv_std": float("nan"),
            "kendall_indiv": float("nan"),
            "kendall_indiv_std": float("nan"),
        }

    # 1. Group-level correlation (loss distance vs grand mean human rating)
    group_mean = human_matrix.mean(axis=1)

    pr_g, p_val_pr = stats.pearsonr(aligned_dist, group_mean)
    sr_g, p_val_sr = stats.spearmanr(aligned_dist, group_mean)
    kr_g, p_val_kr = stats.kendalltau(aligned_dist, group_mean)

    # 2. Individual-level correlation (loss distance vs each individual participant)
    indiv_p = []
    indiv_s = []
    indiv_k = []

    for col in human_matrix.columns:
        sub_ratings = human_matrix[col]
        if sub_ratings.nunique() <= 1:
            continue
        p_sub, _ = stats.pearsonr(aligned_dist, sub_ratings)
        s_sub, _ = stats.spearmanr(aligned_dist, sub_ratings)
        k_sub, _ = stats.kendalltau(aligned_dist, sub_ratings)
        indiv_p.append(p_sub)
        indiv_s.append(s_sub)
        indiv_k.append(k_sub)

    has_indiv = len(indiv_p) >= 2
    return {
        "n_stimuli": n_stimuli,
        "n_subjects": n_subjects,
        "pearson_group": float(pr_g),
        "pearson_group_p": float(p_val_pr),
        "spearman_group": float(sr_g),
        "spearman_group_p": float(p_val_sr),
        "kendall_group": float(kr_g),
        "kendall_group_p": float(p_val_kr),
        "pearson_indiv": float(np.mean(indiv_p)) if indiv_p else float("nan"),
        "pearson_indiv_std": (
            float(np.std(indiv_p, ddof=1)) if has_indiv else float("nan")
        ),
        "spearman_indiv": float(np.mean(indiv_s)) if indiv_s else float("nan"),
        "spearman_indiv_std": (
            float(np.std(indiv_s, ddof=1)) if has_indiv else float("nan")
        ),
        "kendall_indiv": float(np.mean(indiv_k)) if indiv_k else float("nan"),
        "kendall_indiv_std": (
            float(np.std(indiv_k, ddof=1)) if has_indiv else float("nan")
        ),
    }


def compute_correlations(
    distances_source: Union[str, Path, pd.DataFrame],
    mushra_source: Union[str, Path, pd.DataFrame],
    level: Literal["all", "entire", "modulation", "modulation_timbre"] = "all",
    complete_subjects: Literal["slice", "global"] = "slice",
    exclude_reference: bool = True,
    show_progress: bool = True,
    sort_by: Optional[str] = "pearson_group",
) -> pd.DataFrame:
    """Compute correlations between audio loss functions and human perceptual ratings.

    Parameters
    ----------
    distances_source : Union[str, Path, pd.DataFrame]
        CSV/TSV path or DataFrame of audio loss distances.
    mushra_source : Union[str, Path, pd.DataFrame]
        TSV/CSV path or DataFrame of preprocessed MUSHRA responses.
    level : Literal["all", "entire", "modulation", "modulation_timbre"], default="all"
        Granularity level ('entire', 'modulation', 'modulation_timbre', or 'all').
    complete_subjects : Literal["slice", "global"], default="slice"
        'slice' (complete data for that condition) or 'global' (complete across all 18 trials).
    exclude_reference : bool, default=True
        Exclude reference stimulus rating (default: True).
    show_progress : bool, default=True
        Display tqdm progress bar.
    sort_by : Optional[str], default="pearson_group"
        Column to sort loss functions by in descending order within each granularity + condition group.

    Returns
    -------
    pd.DataFrame
        DataFrame containing full correlation results across all evaluated losses and conditions.
    """
    df_dist = prepare_distances(distances_source, exclude_reference=exclude_reference)
    df_human = prepare_data(mushra_source, exclude_reference=exclude_reference)

    if complete_subjects == "global":
        full_pivot = df_human.pivot_table(
            index=["trial_id", "rating_stimulus"],
            columns="session_uuid",
            values="rating_score",
        )
        valid_subjects = set(full_pivot.dropna(axis=1).columns)
        df_human = df_human[df_human["session_uuid"].isin(valid_subjects)].copy()
        log.info(
            f"Using {len(valid_subjects)} global complete subjects across all analyses."
        )

    all_losses = sorted(df_dist["loss_fn"].unique())

    slices: list[tuple[str, str, int, pd.DataFrame]] = []

    # 1. Entire dataset (18 trials)
    if level in ("all", "entire"):
        piv_entire = df_human.pivot_table(
            index=["trial_id", "rating_stimulus"],
            columns="session_uuid",
            values="rating_score",
        ).dropna(axis=1)
        slices.append(("entire", "all_18_trials", 18, piv_entire))

    # 2. Per modulation type (6 trials each)
    if level in ("all", "modulation"):
        for mod in sorted(df_human["modulation"].dropna().unique()):
            sub_m = df_human[df_human["modulation"] == mod]
            piv_mod = sub_m.pivot_table(
                index=["trial_id", "rating_stimulus"],
                columns="session_uuid",
                values="rating_score",
            ).dropna(axis=1)
            slices.append(("modulation", mod, sub_m["trial_id"].nunique(), piv_mod))

    # 3. Per modulation x timbre (2 trials each)
    if level in ("all", "modulation_timbre"):
        for mt in sorted(df_human["mod_timbre"].dropna().unique()):
            sub_mt = df_human[df_human["mod_timbre"] == mt]
            piv_mt = sub_mt.pivot_table(
                index=["trial_id", "rating_stimulus"],
                columns="session_uuid",
                values="rating_score",
            ).dropna(axis=1)
            slices.append(
                ("modulation_timbre", mt, sub_mt["trial_id"].nunique(), piv_mt)
            )

    records = []
    total_tasks = len(all_losses) * len(slices)
    pbar = tqdm(
        total=total_tasks,
        desc="Computing loss correlations",
        unit="eval",
        disable=not show_progress,
    )

    for loss_name in all_losses:
        loss_df = df_dist[df_dist["loss_fn"] == loss_name].set_index(
            ["trial_id", "rating_stimulus"]
        )

        for granularity, condition, n_trials, human_matrix in slices:
            pbar.set_postfix_str(f"{loss_name} | {condition}")

            res = compute_slice_correlations(human_matrix, loss_df["distance"])
            res["loss_fn"] = loss_name
            res["granularity"] = granularity
            res["condition"] = condition
            res["n_trials"] = n_trials

            records.append(res)
            pbar.update(1)

    pbar.close()

    result_df = pd.DataFrame(records)

    first_cols = [
        "loss_fn",
        "granularity",
        "condition",
        "n_trials",
        "n_stimuli",
        "n_subjects",
    ]
    other_cols = [c for c in result_df.columns if c not in first_cols]
    result_df = result_df[first_cols + other_cols]

    if sort_by is not None and str(sort_by).lower() != "none" and not result_df.empty:
        col_map = {c.lower(): c for c in result_df.columns}
        if str(sort_by).lower() not in col_map:
            raise ValueError(
                f"Cannot sort by '{sort_by}'. Available columns: {list(result_df.columns)}"
            )
        matched_sort_col = col_map[str(sort_by).lower()]
        sorted_dfs = []
        for granularity, condition, _, _ in slices:
            mask = (result_df["granularity"] == granularity) & (
                result_df["condition"] == condition
            )
            sub_df = result_df[mask]
            if not sub_df.empty:
                sub_sorted = sub_df.sort_values(
                    by=matched_sort_col, ascending=False, kind="mergesort"
                )
                sorted_dfs.append(sub_sorted)
        if sorted_dfs:
            result_df = pd.concat(sorted_dfs, ignore_index=True)

    return result_df


def main() -> None:
    """Command-line interface for calculating correlations between audio loss functions and human perceptual ratings."""
    parser = argparse.ArgumentParser(
        description="Compute correlations between audio distance loss functions and MUSHRA listening test responses."
    )
    parser.add_argument(
        "distances_path",
        nargs="?",
        default=os.path.join(OUT_DIR, "audio_distances.tsv"),
        help=f"Path to distances TSV file (default: {OUT_DIR}/audio_distances.tsv)",
    )
    parser.add_argument(
        "mushra_path",
        nargs="?",
        default=os.path.join(OUT_DIR, "listening_test_responses_postprocessed.tsv"),
        help=f"Path to postprocessed human listening responses (default: {OUT_DIR}/listening_test_responses_postprocessed.tsv)",
    )
    parser.add_argument(
        "--level",
        choices=["all", "entire", "modulation", "modulation_timbre"],
        default="modulation",
        help="Granularity level to compute (default: modulation)",
    )
    parser.add_argument(
        "--complete-subjects",
        choices=["slice", "global"],
        default="slice",
        help="Subjects to include: 'slice' (complete for that condition) or 'global' (complete for all 18 trials).",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include reference stimulus rating in the calculation (default: excluded).",
    )
    parser.add_argument(
        "--sort-by",
        default="spearman_group",
        help="Column to sort loss functions by within each granularity + condition group (default: spearman_group). Set to 'none' to disable sorting.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(OUT_DIR, "audio_distance_correlations.tsv"),
        help=f"Optional path to save results as CSV or TSV (default: {OUT_DIR}/audio_distance_correlations.tsv).",
    )
    args = parser.parse_args()

    distances_path = Path(args.distances_path).expanduser().resolve()
    if not distances_path.exists():
        raise FileNotFoundError(f"Distances file not found: {distances_path}")

    mushra_path = Path(args.mushra_path).expanduser().resolve()
    if not mushra_path.exists():
        raise FileNotFoundError(f"MUSHRA file not found: {mushra_path}")

    results_df = compute_correlations(
        distances_source=distances_path,
        mushra_source=mushra_path,
        level=args.level,
        complete_subjects=args.complete_subjects,
        exclude_reference=not args.include_reference,
        sort_by=args.sort_by,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.precision", 3)

    sep_bar = "=" * 120
    log.info("\n" + sep_bar)
    log.info("AUDIO DISTANCE CORRELATION WITH HUMAN PERCEPTUAL DATA (DataFrame)")
    log.info(sep_bar)
    log.info("\n" + results_df.to_string(index=False))
    log.info(sep_bar + "\n")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sep = "\t" if out_path.suffix == ".tsv" else ","
        results_df.to_csv(out_path, sep=sep, index=False)
        log.info(f"Results successfully exported to: {out_path}")


if __name__ == "__main__":
    main()
