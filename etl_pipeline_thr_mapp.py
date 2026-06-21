"""
ETL and Analysis Pipeline with Threshold Filtering and Value Mapping

Features added:
- Threshold filtering: keep rows where a column meets a threshold condition.
- Value mapping: map values of a column to new values via dicts or mapping CSV files.

Other features:
- Read CSVs from input folder, clean, optional per-file limiting, filter by allowed values,
  split by one or more columns, per-group limiting, univariate and bivariate analysis,
  save trimmed groups, stats, correlations, plots, and a merged summary CSV.

Configuration block below controls behavior.
"""

import os
import re
import math
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Configuration
# ---------------------------

# Folders
INPUT_FOLDER = "csv_all"
INPUT_FOLDER = "csv_serie"
# INPUT_FOLDER = "csv_sweeps"
OUTPUT_FOLDER = f"processed_{INPUT_FOLDER}"
STATS_FOLDER = "stats"
CORR_FOLDER = "correlations"
PLOTS_FOLDER = "plots"
MAPPINGS_FOLDER = "mappings"  # optional CSV mapping files can be placed here

IOU_COLS = ["3_test_GDA_iou", "1_test/GDA/iou", "2_test/GDA/iou", 
            "3_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_BASELINE = 0.617

# Cleaning
COLUMNS_TO_REMOVE = ["Created", "Runtime"]
NO_COLS = 14
COLUMNS_TO_KEEP = ["batch_size", "epochs", "warmup_epochs", "wd", "ema"]

# Filtering by allowed values (file with newline separated allowed values)
FILTER_COLUMN = "Status"
FILTER_VALUES_FILE = "allowed_values.txt"  # if missing, no allowed-values filtering 

# Threshold filters
# Each entry: (column, operator, value)
# operator one of: ">", ">=", "<", "<=", "==", "!="
THRESHOLD_FILTERS = [
    # Example: keep rows where Amount > 100
    # ("Amount", ">", 100),
    # Example: keep rows where Score >= 0.8
    # ("Score", ">=", 0.8),
    (m, '>', IOU_BASELINE) for m in IOU_COLS
]

# Value mappings
# Two ways to specify mappings:
# 1) Inline dictionary mapping: column -> {old_value: new_value, ...}
# 2) Mapping CSV files placed in MAPPINGS_FOLDER with filename <column>_map.csv
#    CSV format: old,new  (header optional)
VALUE_MAPPINGS_INLINE: Dict[str, Dict[str, str]] = {
    # Example:
    # "Status": {"OK": "Approved", "NOK": "Rejected"},
    "val": {          # map raw values → output filename suffix
    "/users/project1/pt01299/synt/gda70/train/index_val.csv": "gda",
    "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv": "s",
    "/users/project1/pt01299/synt/mix_val.csv": "m",
    "/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv": "dk"
} # jeszcze Sweep
}

# Splitting (grouping)
SPLIT_COLUMNS = ['val', 'Sweep'] # ["batch_size", "epochs", "warmup_epochs", "wd", "ema"]  # list of columns to group by; can be empty

# Sorting and limiting per-file
ROW_LIMIT_FRACTION_PER_FILE = 0.0  # 0.0 means no limiting
POSSIBLE_SORT_COLUMNS = IOU_COLS

# Per-group limiting
GROUP_LIMIT_FRACTION = 0.0
# GROUP_LIMIT_FRACTION = 0.3
POSSIBLE_GROUP_SORT_COLUMNS = IOU_COLS

# Targets for repeated analysis
TARGET_VARIABLES = IOU_COLS

# Univariate stats to include in merged summary
UNIVARIATE_STATS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

# Plot settings
PLOT_DPI = 150
PLOT_FIGSIZE = (6, 4)

# Misc
VERBOSE = True


# ---------------------------
# Utilities
# ---------------------------

def log(msg: str):
    if VERBOSE:
        print(msg)


def ensure_dirs():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(STATS_FOLDER, exist_ok=True)
    os.makedirs(CORR_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(MAPPINGS_FOLDER, exist_ok=True)


def sanitize_for_filename(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\._=]", "", s)
    return s or "NA"


def find_first_existing_column(df: pd.DataFrame, possible_columns: List[str]) -> Optional[str]:
    for col in possible_columns:
        if col in df.columns:
            return col
    return None


# ---------------------------
# Mapping utilities
# ---------------------------

def load_mapping_file_for_column(column: str) -> Dict[str, str]:
    """
    Load mapping CSV for a column from MAPPINGS_FOLDER.
    Expected CSV format: old,new  (header optional)
    Returns mapping dict old -> new. If file not found, returns empty dict.
    """
    fname_csv = os.path.join(MAPPINGS_FOLDER, f"{column}_map.csv")
    mapping = {}
    if not os.path.isfile(fname_csv):
        return mapping
    try:
        # Try reading with header; if header present, pandas will parse it
        df_map = pd.read_csv(fname_csv, dtype=str, header=0)
        # If there are at least two columns, take first two as old,new
        if df_map.shape[1] >= 2:
            old_col = df_map.columns[0]
            new_col = df_map.columns[1]
            for _, row in df_map.iterrows():
                old = row[old_col]
                new = row[new_col]
                if pd.isna(old):
                    continue
                mapping[str(old)] = "" if pd.isna(new) else str(new)
        else:
            # Fallback: try reading lines "old,new"
            with open(fname_csv, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
    except Exception as e:
        log(f"Failed to load mapping file for {column}: {e}")
    return mapping


def build_value_mappings() -> Dict[str, Dict[str, str]]:
    """
    Combine inline mappings and mapping files into a single mapping dict.
    Inline mappings take precedence over file mappings.
    """
    combined = {}
    # Start with file-based mappings
    # For each file in MAPPINGS_FOLDER that matches pattern <column>_map.csv, load it
    for fname in os.listdir(MAPPINGS_FOLDER):
        if not fname.endswith("_map.csv"):
            continue
        col = fname[:-len("_map.csv")]
        file_map = load_mapping_file_for_column(col)
        if file_map:
            combined[col] = file_map

    # Overlay inline mappings (take precedence)
    for col, mapping in VALUE_MAPPINGS_INLINE.items():
        if mapping:
            combined[col] = {str(k): str(v) for k, v in mapping.items()}

    return combined


def apply_value_mappings(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Apply mappings to DataFrame in-place (returns a new DataFrame).
    For each column in mappings, replace values according to mapping dict.
    """
    if not mappings:
        return df
    df = df.copy()
    for col, mapping in mappings.items():
        if col not in df.columns:
            continue
        # Use map with fillna to preserve unmapped values
        df[col] = df[col].astype(object).map(lambda x: mapping.get(str(x), x))
        log(f"Applied mapping for column '{col}' with {len(mapping)} entries.")
    return df


# ---------------------------
# Threshold filtering
# ---------------------------

def _apply_single_threshold(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.DataFrame:
    """
    Apply a single threshold filter to df and return filtered df.
    Supported operators: >, >=, <, <=, ==, !=
    Non-numeric values are coerced to numeric where appropriate; if coercion fails, comparison uses original values.
    """
    if column not in df.columns:
        log(f"Threshold filter skipped: column '{column}' not present.")
        return df

    # Try numeric comparison first
    series = df[column]
    # If value is numeric, coerce series to numeric
    numeric_value = None
    try:
        numeric_value = float(value)
    except Exception:
        numeric_value = None

    if numeric_value is not None:
        # Coerce series to numeric
        s_num = pd.to_numeric(series, errors="coerce")
        # Build mask based on operator
        if operator == ">":
            mask = s_num > numeric_value
        elif operator == ">=":
            mask = s_num >= numeric_value
        elif operator == "<":
            mask = s_num < numeric_value
        elif operator == "<=":
            mask = s_num <= numeric_value
        elif operator == "==":
            mask = s_num == numeric_value
        elif operator == "!=":
            mask = s_num != numeric_value
        else:
            log(f"Unsupported operator '{operator}' for threshold filter on column '{column}'. Skipping.")
            return df
        # Replace NaN in mask with False (rows where coercion failed)
        mask = mask.fillna(False)
    else:
        # Non-numeric comparison: compare as strings
        s_str = series.astype(str)
        cmp_val = str(value)
        if operator == "==":
            mask = s_str == cmp_val
        elif operator == "!=":
            mask = s_str != cmp_val
        else:
            log(f"Operator '{operator}' not supported for non-numeric threshold on column '{column}'. Skipping.")
            return df

    before = len(df)
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    log(f"Applied threshold filter {column} {operator} {value}: {before} -> {after} rows")
    return df_filtered


def apply_threshold_filters(df: pd.DataFrame, thresholds: List[Tuple[str, str, Any]]) -> pd.DataFrame:
    """
    Apply a list of threshold filters sequentially.
    Each threshold is a tuple (column, operator, value).
    """
    if not thresholds:
        return df
    df_out = df
    for col, op, val in thresholds:
        df_out = _apply_single_threshold(df_out, col, op, val)
        if df_out.empty:
            log("All rows filtered out by threshold filters.")
            break
    return df_out


# ---------------------------
# Existing cleaning, splitting, analysis functions
# ---------------------------

def remove_selected_columns(df: pd.DataFrame, cols_to_remove: List[str]) -> pd.DataFrame:
    cols_present = [c for c in cols_to_remove if c in df.columns]
    if cols_present:
        log(f"Removing columns: {cols_present}")
        df = df.drop(columns=cols_present)
        cols_rm2 = df.columns[NO_COLS:]
        log(f"And columns: {cols_rm2}")
        df = df.drop(columns=cols_rm2)
        log(f"Left with columns: {df.columns}")
        return df
    return df


def drop_empty_and_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        log(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def sort_and_limit_rows(df: pd.DataFrame, fraction: float, possible_sort_cols: List[str]) -> pd.DataFrame:
    if not (0 < fraction < 1):
        return df
    sort_col = find_first_existing_column(df, possible_sort_cols)
    if sort_col is None:
        log("No sort column found for limiting; skipping limiting.")
        return df
    df_sorted = df.sort_values(by=sort_col, na_position="last")
    limit = int(math.floor(len(df_sorted) * fraction))
    log(f"Sorting by '{sort_col}' and limiting to top {limit} rows ({fraction*100:.1f}%).")
    return df_sorted.iloc[:limit].reset_index(drop=True)


def split_dataframe(df: pd.DataFrame, group_cols: List[str]) -> Dict[str, pd.DataFrame]:
    if not group_cols:
        return {"all": df.copy().reset_index(drop=True)}
    existing_group_cols = [c for c in group_cols if c in df.columns]
    if not existing_group_cols:
        log("No group columns found in file; treating entire file as single group.")
        return {"all": df.copy().reset_index(drop=True)}
    groups = {}
    grouped = df.groupby(existing_group_cols, dropna=False)
    for group_values, group_df in grouped:
        if isinstance(group_values, tuple):
            row = pd.Series({col: val for col, val in zip(existing_group_cols, group_values)})
        else:
            row = pd.Series({existing_group_cols[0]: group_values})
        key = "__".join(f"{col}={row[col]}" for col in existing_group_cols)
        key = sanitize_for_filename(key)
        groups[key] = group_df.reset_index(drop=True)
    log(f"Split into {len(groups)} groups using columns: {existing_group_cols}")
    return groups


def compute_univariate_stats(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    desc = numeric.describe(percentiles=[0.25, 0.5, 0.75])
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
    if target not in df.columns:
        return pd.Series(dtype=float)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric_df.columns:
        numeric_df[target] = pd.to_numeric(df[target], errors="coerce")
    if target not in numeric_df.columns:
        return pd.Series(dtype=float)
    corrs = {}
    for col in numeric_df.columns:
        if col == target:
            continue
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
    df = stats.to_frame().T
    df.to_csv(out_path, index=False)


def save_correlations(corrs: pd.Series, out_path: str):
    df = corrs.reset_index()
    df.columns = ["variable", "correlation"]
    df.to_csv(out_path, index=False)


def plot_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str, out_path: str):
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
        plt.scatter(plot_df[x_col], plot_df[y_col], s=20)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


# ---------------------------
# Main processing per file
# ---------------------------

def load_allowed_values(path: str) -> Optional[set]:
    if not path or not os.path.isfile(path):
        log(f"Allowed values file not found: {path} (no filtering will be applied).")
        return None
    with open(path, "r", encoding="utf-8") as f:
        vals = {line.strip() for line in f if line.strip()}
    log(f"Loaded {len(vals)} allowed filter values from {path}.")
    return vals


def process_single_file(filepath: str, allowed_values: Optional[set], mappings: Dict[str, Dict[str, str]], merged_rows: List[Dict]):
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    log(f"\n--- Processing file: {filename} ---")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        log(f"Failed to read {filename}: {e}")
        return

    # Apply value mappings early so subsequent steps use mapped values
    if mappings:
        df = apply_value_mappings(df, mappings)

    # Per-file sort & limit
    df = sort_and_limit_rows(df, ROW_LIMIT_FRACTION_PER_FILE, POSSIBLE_SORT_COLUMNS)

    # Drop empty and constant columns
    df = drop_empty_and_constant_columns(df)

    # Remove selected columns
    df = remove_selected_columns(df, COLUMNS_TO_REMOVE)

    # Filter by allowed values if configured
    if allowed_values is not None and FILTER_COLUMN in df.columns:
        before = len(df)
        df = df[df[FILTER_COLUMN].isin(allowed_values)].reset_index(drop=True)
        after = len(df)
        log(f"Filtered by {FILTER_COLUMN}: {before} -> {after} rows")

    # Apply threshold filters (per-file)
    if THRESHOLD_FILTERS:
        df = apply_threshold_filters(df, THRESHOLD_FILTERS)
        if df.empty:
            log("No rows left after threshold filtering; skipping file.")
            return

    # Split into groups
    groups = split_dataframe(df, SPLIT_COLUMNS)

    # Process each group
    for group_key, group_df in groups.items():
        log(f"\nProcessing group: {group_key} (rows: {len(group_df)})")

        # Per-group limiting (sorted by possible group sort columns)
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
            # Include numeric columns and those coercible to numeric
            for col in group_df.columns:
                if col == target:
                    continue
                # Check if column has at least two numeric values after coercion
                coerced = pd.to_numeric(group_df[col], errors="coerce")
                coerced_target = pd.to_numeric(group_df[target], errors="coerce")
                plot_df = pd.concat([coerced, coerced_target], axis=1).dropna()
                if plot_df.shape[0] < 2:
                    continue
                plot_name = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_vs_{sanitize_for_filename(col)}.png"
                plot_path = os.path.join(PLOTS_FOLDER, plot_name)
                try:
                    plot_scatter_with_regression(group_df, col, target, plot_path)
                    log(f"Saved plot: {plot_path}")
                except Exception as e:
                    log(f"Failed to create plot {plot_path}: {e}")

            # Prepare merged summary row
            top_corr_var = None
            top_corr_val = None
            if not corrs.empty:
                corrs_nonan = corrs.dropna()
                if not corrs_nonan.empty:
                    abs_corrs = corrs_nonan.abs()
                    top_var = abs_corrs.idxmax()
                    top_corr_val = corrs_nonan.loc[top_var]
                    top_corr_var = top_var

            merged_row = {
                "source_file": base_name,
                "group_key": group_key,
                "target_variable": target,
            }
            for stat_name in UNIVARIATE_STATS:
                merged_row[f"stat_{stat_name}"] = uni_stats.get(stat_name, np.nan)
            merged_row["top_correlated_variable"] = top_corr_var
            merged_row["top_correlation_value"] = top_corr_val

            merged_rows.append(merged_row)


def run_pipeline():
    ensure_dirs()
    allowed_values = load_allowed_values(FILTER_VALUES_FILE)
    mappings = build_value_mappings()

    merged_rows = []

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".csv")]
    if not files:
        log(f"No CSV files found in {INPUT_FOLDER}. Place files there and re-run.")
        return

    for fname in files:
        path = os.path.join(INPUT_FOLDER, fname)
        process_single_file(path, allowed_values, mappings, merged_rows)

    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        merged_out = os.path.join(OUTPUT_FOLDER, "merged_summary.csv")
        merged_df.to_csv(merged_out, index=False)
        log(f"\nMerged summary saved to: {merged_out}")
    else:
        log("No analysis rows were produced; merged summary not created.")


if __name__ == "__main__":
    run_pipeline()
