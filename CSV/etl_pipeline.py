"""
ETL and Analysis Pipeline

Features:
- Read CSV files from input_folder.
- Remove selected columns.
- Detect first available sort column from a priority list and sort.
- Limit rows per file or per group by fraction (deterministic: top rows after sorting).
- Filter rows by allowed values loaded from a newline-separated file.
- Split entries by one or more parameters (groupby on list of columns).
- For each group and for each target variable (if present):
    - Compute univariate descriptive statistics and save to CSV.
    - Compute Pearson correlations between target and other numeric columns; save CSV.
    - Create scatter plots with regression line for target vs each numeric column; save plots.
- Save trimmed per-group CSVs, per-group-target stats, per-group-target correlations, plots, and a merged summary CSV.
- Robust: skips missing columns, handles non-numeric data, creates directories, logs progress.
"""

import os
import re
import math
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Configuration (edit here)
# ---------------------------

# Folders
INPUT_FOLDER = "input_csv"
OUTPUT_FOLDER = "processed_csv"          # trimmed per-group CSVs and merged summary
STATS_FOLDER = "stats"                   # per-group-target univariate stats
CORR_FOLDER = "correlations"             # per-group-target correlation CSVs
PLOTS_FOLDER = "plots"                   # scatter/regression plots

# Cleaning
COLUMNS_TO_REMOVE = ["ColumnA", "ColumnB"]  # remove if present

# Filtering
FILTER_COLUMN = "Status"                    # column to filter on (if present)
FILTER_VALUES_FILE = "allowed_values.txt"   # newline-separated allowed values; if missing, no filtering

# Splitting (grouping)
SPLIT_COLUMNS = ["Category", "Subcategory"]  # list of columns to group by; can be empty for no splitting

# Sorting and limiting
# For per-file limiting (applied before splitting)
ROW_LIMIT_FRACTION_PER_FILE = 0.0            # 0.0 means no limiting; otherwise 0 < fraction < 1
POSSIBLE_SORT_COLUMNS = ["Timestamp", "Date", "CreatedAt"]  # priority list; first present is used

# For per-group limiting (applied after splitting)
GROUP_LIMIT_FRACTION = 0.0                   # 0.0 means no limiting; otherwise 0 < fraction < 1
POSSIBLE_GROUP_SORT_COLUMNS = ["Timestamp", "Date", "CreatedAt"]

# Targets for repeated analysis
TARGET_VARIABLES = ["ValueA", "ValueB", "ValueC"]  # list of target columns to analyze (priority order not required)

# Statistics to compute (univariate)
UNIVARIATE_STATS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

# Plot settings
PLOT_DPI = 150
PLOT_FIGSIZE = (6, 4)

# Misc
RANDOM_STATE = 42  # not used for deterministic top-slicing, kept for future use
VERBOSE = True


# ---------------------------
# Helper utilities
# ---------------------------

def ensure_dirs():
    """Create all required output directories."""
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(STATS_FOLDER, exist_ok=True)
    os.makedirs(CORR_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)


def log(msg: str):
    """Simple logger controlled by VERBOSE flag."""
    if VERBOSE:
        print(msg)


def sanitize_for_filename(s: str) -> str:
    """Sanitize a string to be safe for filenames."""
    s = str(s)
    s = s.strip()
    # Replace spaces with underscore and remove problematic characters
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\._=]", "", s)
    return s or "NA"


def group_key_from_row(row: pd.Series, group_cols: List[str]) -> str:
    """Create a group key string like 'col1=val1__col2=val2' sanitized for filenames."""
    parts = []
    for col in group_cols:
        val = row.get(col, "")
        parts.append(f"{col}={val}")
    key = "__".join(parts)
    return sanitize_for_filename(key)


def find_first_existing_column(df: pd.DataFrame, possible_columns: List[str]) -> Optional[str]:
    """Return the first column from possible_columns that exists in df, or None."""
    for col in possible_columns:
        if col in df.columns:
            return col
    return None


def load_allowed_values(path: str) -> Optional[set]:
    """Load allowed values from a newline-separated file. Return None if file missing."""
    if not path or not os.path.isfile(path):
        log(f"Allowed values file not found: {path} (no filtering will be applied).")
        return None
    with open(path, "r", encoding="utf-8") as f:
        vals = {line.strip() for line in f if line.strip()}
    log(f"Loaded {len(vals)} allowed filter values from {path}.")
    return vals


# ---------------------------
# Data cleaning and trimming
# ---------------------------

def remove_selected_columns(df: pd.DataFrame, cols_to_remove: List[str]) -> pd.DataFrame:
    """Remove selected columns if present."""
    cols_present = [c for c in cols_to_remove if c in df.columns]
    if cols_present:
        log(f"Removing columns: {cols_present}")
        return df.drop(columns=cols_present)
    return df


def drop_empty_and_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are all NaN or have a single unique value (including NaN-only)."""
    # Drop all-NaN
    df = df.dropna(axis=1, how="all")
    # Drop constant columns (nunique <= 1 counting NaN)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        log(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def sort_and_limit_rows(df: pd.DataFrame, fraction: float, possible_sort_cols: List[str]) -> pd.DataFrame:
    """
    Sort by the first existing column in possible_sort_cols and keep the top fraction of rows.
    If fraction <= 0 or >= 1, returns df unchanged.
    """
    if not (0 < fraction < 1):
        return df
    sort_col = find_first_existing_column(df, possible_sort_cols)
    if sort_col is None:
        log("No sort column found for limiting; skipping per-file limiting.")
        return df
    # Sort ascending (deterministic). If you want descending, change ascending=False.
    df_sorted = df.sort_values(by=sort_col, na_position="last", ascending=False)
    limit = int(math.floor(len(df_sorted) * fraction))
    log(f"Sorting by '{sort_col}' and limiting to top {limit} rows ({fraction*100:.1f}%).")
    return df_sorted.iloc[:limit].reset_index(drop=True)


# ---------------------------
# Splitting (grouping)
# ---------------------------

def split_dataframe(df: pd.DataFrame, group_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Split df into groups by group_cols.
    Returns a dict mapping group_key -> group_df.
    If group_cols is empty or none present, returns a single group with key 'all'.
    """
    if not group_cols:
        return {"all": df.copy().reset_index(drop=True)}

    # Only use group columns that exist
    existing_group_cols = [c for c in group_cols if c in df.columns]
    if not existing_group_cols:
        log("No group columns found in file; treating entire file as single group.")
        return {"all": df.copy().reset_index(drop=True)}

    groups = {}
    grouped = df.groupby(existing_group_cols, dropna=False)
    for group_values, group_df in grouped:
        # group_values can be a scalar if only one group column, else a tuple
        if isinstance(group_values, tuple):
            # Build a row-like mapping for naming
            row = pd.Series({col: val for col, val in zip(existing_group_cols, group_values)})
        else:
            row = pd.Series({existing_group_cols[0]: group_values})
        key = group_key_from_row(row, existing_group_cols)
        groups[key] = group_df.reset_index(drop=True)
    log(f"Split into {len(groups)} groups using columns: {existing_group_cols}")
    return groups


# ---------------------------
# Analysis functions
# ---------------------------

def compute_univariate_stats(series: pd.Series) -> pd.Series:
    """Compute univariate descriptive stats similar to pandas describe() for a single series."""
    # Convert to numeric if possible
    numeric = pd.to_numeric(series, errors="coerce")
    desc = numeric.describe(percentiles=[0.25, 0.5, 0.75])
    # Ensure keys match UNIVARIATE_STATS mapping
    # pandas describe returns count, mean, std, min, 25%, 50%, 75%, max
    # We'll reindex to ensure consistent order and presence
    stats_map = {
        "count": desc.get("count", np.nan),
        "mean": desc.get("mean", np.nan),
        "std": desc.get("std", np.nan),
        "min": desc.get("min", np.nan),
        "25%": desc.get("25%", np.nan),
        "50%": desc.get("50%", np.nan),
        "75%": desc.get("75%", np.nan),
        "max": desc.get("max", np.nan),
    }
    return pd.Series(stats_map)


def compute_correlations(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Compute Pearson correlation between target and every other numeric column.
    Returns a Series indexed by other column name with correlation values (NaN for non-numeric or insufficient data).
    """
    if target not in df.columns:
        return pd.Series(dtype=float)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric_df.columns:
        # Try to coerce target to numeric
        numeric_df[target] = pd.to_numeric(df[target], errors="coerce")

    if target not in numeric_df.columns:
        return pd.Series(dtype=float)

    corrs = {}
    for col in numeric_df.columns:
        if col == target:
            continue
        # Drop NA pairs
        pair = numeric_df[[target, col]].dropna()
        if len(pair) < 2:
            corrs[col] = np.nan
            continue
        try:
            corr = pair[target].corr(pair[col])
            corrs[col] = corr
        except Exception:
            corrs[col] = np.nan
    return pd.Series(corrs).sort_values(ascending=False)


def save_univariate_stats(stats: pd.Series, out_path: str):
    """Save univariate stats Series to CSV (single-row with stat names as columns)."""
    df = stats.to_frame().T  # single-row DataFrame
    df.to_csv(out_path, index=False)


def save_correlations(corrs: pd.Series, out_path: str):
    """Save correlations Series to CSV with columns: variable, correlation."""
    df = corrs.reset_index()
    df.columns = ["variable", "correlation"]
    df.to_csv(out_path, index=False)


def plot_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str, out_path: str):
    """
    Create a scatter plot with regression line (using seaborn.regplot).
    Handles non-numeric by coercion; skips if insufficient numeric data.
    """
    # Prepare numeric data
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    plot_df = pd.concat([x, y], axis=1).dropna()
    if plot_df.shape[0] < 2:
        log(f"Not enough numeric data to plot {y_col} vs {x_col}. Skipping plot.")
        return

    plt.figure(figsize=PLOT_FIGSIZE)
    sns.set(style="whitegrid")
    try:
        sns.regplot(x=x_col, y=y_col, data=plot_df, scatter_kws={"s": 20}, line_kws={"color": "red"})
    except Exception:
        # fallback: scatter only
        plt.scatter(plot_df[x_col], plot_df[y_col], s=20)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


# ---------------------------
# Main processing pipeline
# ---------------------------

def process_single_file(filepath: str,
                        allowed_values: Optional[set],
                        merged_rows: List[Dict]):
    """Process one CSV file end-to-end and append summary rows to merged_rows list."""
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    log(f"\n--- Processing file: {filename} ---")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        log(f"Failed to read {filename}: {e}")
        return

    # 1) Remove selected columns
    df = remove_selected_columns(df, COLUMNS_TO_REMOVE)

    # 2) Drop empty and constant columns
    df = drop_empty_and_constant_columns(df)

    # 3) Per-file sort & limit (deterministic)
    df = sort_and_limit_rows(df, ROW_LIMIT_FRACTION_PER_FILE, POSSIBLE_SORT_COLUMNS)

    # 4) Filter rows by allowed values if configured
    if allowed_values is not None and FILTER_COLUMN in df.columns:
        before = len(df)
        df = df[df[FILTER_COLUMN].isin(allowed_values)].reset_index(drop=True)
        after = len(df)
        log(f"Filtered by {FILTER_COLUMN}: {before} -> {after} rows")

    # 5) Split into groups
    groups = split_dataframe(df, SPLIT_COLUMNS)

    # 6) For each group: optionally limit rows, save trimmed group, analyze targets
    for group_key, group_df in groups.items():
        log(f"\nProcessing group: {group_key} (rows: {len(group_df)})")

        # Limit rows per group (sorted by possible group sort columns)
        if 0 < GROUP_LIMIT_FRACTION < 1:
            group_df = sort_and_limit_rows(group_df, GROUP_LIMIT_FRACTION, POSSIBLE_GROUP_SORT_COLUMNS)
            log(f"After group limiting: {len(group_df)} rows")

        # Save trimmed group CSV
        out_group_csv = os.path.join(OUTPUT_FOLDER, f"{base_name}__{group_key}.csv")
        try:
            group_df.to_csv(out_group_csv, index=False)
            log(f"Saved trimmed group CSV: {out_group_csv}")
        except Exception as e:
            log(f"Failed to save group CSV {out_group_csv}: {e}")

        # For each target variable, perform analysis if present
        for target in TARGET_VARIABLES:
            if target not in group_df.columns:
                log(f"Target '{target}' not in group; skipping.")
                continue

            # Univariate stats
            uni_stats = compute_univariate_stats(group_df[target])
            stats_filename = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_stats.csv"
            stats_path = os.path.join(STATS_FOLDER, stats_filename)
            save_univariate_stats(uni_stats, stats_path)
            log(f"Saved univariate stats: {stats_path}")

            # Correlations
            corrs = compute_correlations(group_df, target)
            corr_filename = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_correlations.csv"
            corr_path = os.path.join(CORR_FOLDER, corr_filename)
            save_correlations(corrs, corr_path)
            log(f"Saved correlations: {corr_path}")

            # Plots: for each other numeric column, create scatter + regression
            numeric_cols = [c for c in group_df.select_dtypes(include=[np.number]).columns if c != target]
            # Also attempt to coerce non-numeric columns to numeric and include them if coercion yields numeric
            for col in group_df.columns:
                if col == target:
                    continue
                if col in numeric_cols:
                    other_col = col
                else:
                    # try coercion
                    coerced = pd.to_numeric(group_df[col], errors="coerce")
                    if coerced.notna().sum() >= 2:
                        other_col = col
                    else:
                        continue

                plot_name = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_vs_{sanitize_for_filename(other_col)}.png"
                plot_path = os.path.join(PLOTS_FOLDER, plot_name)
                try:
                    plot_scatter_with_regression(group_df, other_col, target, plot_path)
                    log(f"Saved plot: {plot_path}")
                except Exception as e:
                    log(f"Failed to create plot {plot_path}: {e}")

            # Prepare merged summary row: include univariate stats and top correlated variable
            top_corr_var = None
            top_corr_val = None
            if not corrs.empty:
                # drop NaN correlations and take absolute value to find strongest
                corrs_nonan = corrs.dropna()
                if not corrs_nonan.empty:
                    # choose variable with highest absolute correlation
                    abs_corrs = corrs_nonan.abs()
                    top_var = abs_corrs.idxmax()
                    top_corr_val = corrs_nonan.loc[top_var]
                    top_corr_var = top_var

            merged_row = {
                "source_file": base_name,
                "group_key": group_key,
                "target_variable": target,
            }
            # Add univariate stats fields
            for stat_name in UNIVARIATE_STATS:
                merged_row[f"stat_{stat_name}"] = uni_stats.get(stat_name, np.nan)
            merged_row["top_correlated_variable"] = top_corr_var
            merged_row["top_correlation_value"] = top_corr_val

            merged_rows.append(merged_row)


def run_pipeline():
    """Main entry point: iterate files, process, and save merged summary."""
    ensure_dirs()
    allowed_values = load_allowed_values(FILTER_VALUES_FILE)

    merged_rows = []

    # Process each CSV file in input folder
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".csv")]
    if not files:
        log(f"No CSV files found in {INPUT_FOLDER}. Place files there and re-run.")
        return

    for fname in files:
        path = os.path.join(INPUT_FOLDER, fname)
        process_single_file(path, allowed_values, merged_rows)

    # Save merged summary
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        merged_out = os.path.join(OUTPUT_FOLDER, "merged_summary.csv")
        merged_df.to_csv(merged_out, index=False)
        log(f"\nMerged summary saved to: {merged_out}")
    else:
        log("No analysis rows were produced; merged summary not created.")


# ---------------------------
# Run when executed as script
# ---------------------------

if __name__ == "__main__":
    run_pipeline()
