"""
ETL and Analysis Pipeline with violin plots, categorical x handling, fixed y-range,
per-source/group plot folders, and 'all' grouping.

Requirements:
- pandas, numpy, matplotlib, seaborn installed
- Place CSV files in INPUT_FOLDER
- Configure options in the CONFIGURATION block below
"""

import os
import re
import math
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


FVAR = '_thr'
INPUT_FOLDER = "3"
PREF = 'CSV/ETL'
OUTPUT_FOLDER = f"{PREF}/processed_{INPUT_FOLDER}{FVAR}"
STATS_FOLDER = f"{PREF}/stats_{INPUT_FOLDER}{FVAR}"
CORR_FOLDER = f"{PREF}/correlations_{INPUT_FOLDER}{FVAR}"
PLOTS_FOLDER = f"{PREF}/plots_{INPUT_FOLDER}{FVAR}"

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK
IOU_BASELINE = 0.617
IOU_BASELINE_GDA = 0.617

SPLIT_COLUMNS = ['tr_val']
CORRELATION_THR = 0.25

# Threshold filters
# Each entry: (column, operator, value)
# operator one of: ">", ">=", "<", "<=", "==", "!="
THRESHOLD_FILTERS = [
    (m, '>', IOU_BASELINE) for m in IOU_COLS_GDA
]

VALUE_MAPPINGS_INLINE: Dict[str, Dict[str, str]] = {
    "val": {
    "/users/project1/pt01299/synt/gda70/train/index_val.csv": "gda",
    "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv": "s",
    "/users/project1/pt01299/synt/mix_val.csv": "m",
    "/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv": "dk"
    },
    "Sweep": {
        "rbnt81jd": 'dk-dk',
        "5i7wk6qc": 'dk-gda',
        "ym6v8fkb": 's-gda',
        "5d42fd7s": 's-s',
        "baz3utdt": 'sub-gda',
        "9lpqaisr": 'sub-s',
        "kdjseui2": 'subM-gda',
        "r4oohhlg": 'subM-m',
        "gdw76omg": 'subM-s',
        "cww3aozn": '16',
        "7l3d9j8z": '16A',
        "8ej2yv2y": 'sub-m',
    }
}

ROW_LIMIT_FRACTION_PER_FILE = 0.0  # 0.0 means no limiting
POSSIBLE_SORT_COLUMNS = IOU_COLS
GROUP_LIMIT_FRACTION = 0.0
POSSIBLE_GROUP_SORT_COLUMNS = IOU_COLS

TARGET_VARIABLES = IOU_COLS

UNIVARIATE_STATS = ["count", "mean", "std", "50%", "75%", "max"]

PLOT_DPI = 100
PLOT_FIGSIZE = (6, 4)

# Treat x as categorical if number of unique values <= this threshold
CATEGORICAL_UNIQUE_THRESHOLD = 6

# Force same y-axis range for all plots: set to (ymin, ymax) or None for auto
Y_AXIS_RANGE: Optional[Tuple[float, float]] = (0.4, 0.8)
Y_AXIS_RANGE: Optional[Tuple[float, float]] = (0, 1)
Y_AXIS_RANGE: Optional[Tuple[float, float]] = None

# Misc
VERBOSE = True


# ---------------------------
# UTILITIES
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


def apply_value_mappings(df: pd.DataFrame, mappings: Dict[str, Dict[str, str]] = VALUE_MAPPINGS_INLINE) -> pd.DataFrame:
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
        df[col] = df[col].astype(object).map(lambda x: mapping.get(str(x), x))
        log(f"Applied mapping for column '{col}' with {len(mapping)} entries.")
    return df


# ---------------------------
# THRESHOLD FILTERS
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
    # if series.isna():
    #     return df
    if numeric_value is not None:
        # Coerce series to numeric
        s_num = series # pd.to_numeric(series, errors="coerce")
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
        if df_out.size != df.size: # only the first working filter
            break
    return df_out


# ---------------------------
# CLEANING, SPLITTING, ANALYSIS (modified to include 'all' group)
# ---------------------------

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


def split_dataframe_with_all(df: pd.DataFrame, group_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Split df into groups by group_cols and always include an 'all' group.
    Returns dict mapping group_key -> group_df. 'all' key contains the whole df.

    Behavior:
    - Always include the 'all' group containing the full dataframe.
    - Create groups for each individual column in `group_cols` that exists in the df.
    - Also create combined groups for the full set of existing group columns (as before).
    """
    groups = {} # {"all": df.copy().reset_index(drop=True)}
    if not group_cols:
        return {"all": df.copy().reset_index(drop=True)}
    existing_group_cols = [c for c in group_cols if c in df.columns]
    if not existing_group_cols:
        log("No group columns found in file; only 'all' group will be used.")
        return {"all": df.copy().reset_index(drop=True)}

    # Create groups for each individual column
    for col in existing_group_cols:
        grouped_single = df.groupby(col, dropna=False)
        for val, group_df in grouped_single:
            # Normalize missing values to None so sanitize_for_filename yields 'NA'
            key_val = None if pd.isna(val) else val
            key = f"{col}={key_val}"
            key = sanitize_for_filename(key)
            groups[key] = group_df.reset_index(drop=True)
            groups[key] = drop_empty_and_constant_columns(groups[key])
            groups[key] = apply_threshold_filters(groups[key], THRESHOLD_FILTERS)
            groups[key] = group_df.reset_index(drop=True)

    # Create combination groups (preserve previous behavior)
    if len(existing_group_cols) > 1:
        grouped = df.groupby(existing_group_cols, dropna=False)
        for group_values, group_df in grouped:
            if isinstance(group_values, tuple):
                row = pd.Series({col: val for col, val in zip(existing_group_cols, group_values)})
            else:
                row = pd.Series({existing_group_cols[0]: group_values})
            parts = []
            for col in existing_group_cols:
                val = row[col]
                val = None if pd.isna(val) else val
                parts.append(f"{col}={val}")
            key = "__".join(parts)
            key = sanitize_for_filename(key)
            groups[key] = group_df.reset_index(drop=True)
            groups[key] = drop_empty_and_constant_columns(groups[key])
            groups[key] = apply_threshold_filters(groups[key], THRESHOLD_FILTERS)
            groups[key] = group_df.reset_index(drop=True)
    
    groups.update({"all": df.copy().reset_index(drop=True)})
    log(f"Split into {len(groups)-1} specific groups (+ 'all') using columns: {existing_group_cols}")
    return groups


def compute_univariate_stats(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    desc = numeric.describe(percentiles=[0.5, 0.75])
    stats_map = {
        "count": desc.get("count", np.nan),
        "mean": desc.get("mean", np.nan),
        "std": desc.get("std", np.nan),
        "50%": desc.get("50%", np.nan),
        "75%": desc.get("75%", np.nan),
        "max": desc.get("max", np.nan),
    }
    return pd.Series(stats_map)


def compute_correlations(df: pd.DataFrame, target: str) -> pd.Series:
    if target not in df.columns:
        return pd.Series(dtype=float)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    # if target not in numeric_df.columns:
    #     numeric_df[target] = pd.to_numeric(df[target], errors="coerce")
    if target not in numeric_df.columns:
        return pd.Series(dtype=float)
    corrs = {}
    for col in numeric_df.columns:
        if col in IOU_COLS:
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


# ---------------------------
# PLOTTING: scatter/regression + violin for categorical x
# ---------------------------

def is_categorical_for_plot(series: pd.Series, threshold: int = CATEGORICAL_UNIQUE_THRESHOLD) -> bool:
    """
    Decide whether to treat a variable as categorical for plotting.
    - Non-numeric types are categorical.
    - Numeric with few unique values (<= threshold) is categorical.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return True
    try:
        nunique = series.nunique(dropna=True)
        if nunique <= threshold:
            return True
    except Exception:
        return False
    return False


def ensure_plot_subfolder(source_base: str, group_key: str) -> str:
    """
    Create and return a subfolder path for plots for a given source file and group.
    Structure: PLOTS_FOLDER/<source_base>/<group_key>/
    """
    folder = os.path.join(PLOTS_FOLDER, sanitize_for_filename(source_base), sanitize_for_filename(group_key))
    os.makedirs(folder, exist_ok=True)
    return folder


def plot_violin_and_strip(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, y_range: Optional[Tuple[float, float]] = Y_AXIS_RANGE):
    """
    Create a violin plot (x categorical, y numeric) with an overlaid stripplot.
    Categories are sorted by their values.
    """
    # Prepare data
    x = df[x_col].astype(object)
    y = df[y_col]
    plot_df = pd.concat([x, y], axis=1).dropna()
    if plot_df.shape[0] < 2:
        # log(f"Not enough data to plot violin for {y_col} vs {x_col}. Skipping.")
        return

    # Sort by x_col to get ordered categories
    if x_col != 'ID':
        plot_df = plot_df.sort_values(by=x_col)
    # Get unique sorted categories
    sorted_categories = plot_df[x_col].unique().tolist()

    plt.figure(figsize=PLOT_FIGSIZE)
    sns.set(style="whitegrid")
    try:
        sns.violinplot(x=x_col, y=y_col, data=plot_df, inner=None, color="lightgray", order=sorted_categories)
        sns.stripplot(x=x_col, y=y_col, data=plot_df, color="black", size=3, jitter=True, order=sorted_categories)
    except Exception:
        # fallback: boxplot + strip
        sns.boxplot(x=x_col, y=y_col, data=plot_df, color="lightgray", order=sorted_categories)
        sns.stripplot(x=x_col, y=y_col, data=plot_df, color="black", size=3, jitter=True, order=sorted_categories)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} by {x_col}")
    if y_range is not None:
        plt.ylim(y_range)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    log(f"Saved violin plot: {out_path}")


def plot_box_and_strip(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, y_range: Optional[Tuple[float, float]] = Y_AXIS_RANGE):
    """
    Create a violin plot (x categorical, y numeric) with an overlaid stripplot.
    Categories are sorted by their values.
    """
    # Prepare data
    x = df[x_col].astype(object)
    y = df[y_col]
    plot_df = pd.concat([x, y], axis=1).dropna()
    if plot_df.shape[0] < 2:
        # log(f"Not enough data to plot violin for {y_col} vs {x_col}. Skipping.")
        return

    # Sort by x_col to get ordered categories
    plot_df = plot_df.sort_values(by=x_col)
    # Get unique sorted categories
    sorted_categories = plot_df[x_col].unique().tolist()

    plt.figure(figsize=PLOT_FIGSIZE)
    sns.set(style="whitegrid")
    sns.boxplot(x=x_col, y=y_col, data=plot_df, color="lightgray", order=sorted_categories)
    sns.stripplot(x=x_col, y=y_col, data=plot_df, color="black", size=3, jitter=True, order=sorted_categories)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} by {x_col}")
    if y_range is not None:
        plt.ylim(y_range)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    log(f"Saved box plot: {out_path}")


def plot_scatter_with_regression_and_fixed_y(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, y_range: Optional[Tuple[float, float]] = Y_AXIS_RANGE):
    """
    Scatter plot with regression line for numeric x and numeric y.
    """
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = df[y_col]
    plot_df = pd.concat([x, y], axis=1).dropna()
    if plot_df.shape[0] < 2:
        # log(f"Not enough numeric data to plot {y_col} vs {x_col}. Skipping plot.")
        return

    # Sort by x for consistent visualization
    plot_df = plot_df.sort_values(by=x_col)

    plt.figure(figsize=PLOT_FIGSIZE)
    sns.set(style="whitegrid")
    try:
        sns.regplot(x=x_col, y=y_col, data=plot_df, scatter_kws={"s": 20}, line_kws={"color": "red"})
    except Exception:
        plt.scatter(plot_df[x_col], plot_df[y_col], s=20)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    if y_range is not None:
        plt.ylim(y_range)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    log(f"Saved scatter/regression plot: {out_path}")


# ---------------------------
# MAIN PROCESSING PER FILE
# ---------------------------

def process_single_file(filepath: str):
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    log(f"\n--- Processing file: {filename} ---")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        log(f"Failed to read {filename}: {e}")

    df = apply_value_mappings(df)
    df = df.drop(columns=['Unnamed: 0', 'train', 'val'])

    for col in TARGET_VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = sort_and_limit_rows(df, ROW_LIMIT_FRACTION_PER_FILE, POSSIBLE_SORT_COLUMNS)

    # if THRESHOLD_FILTERS:
    #     df = apply_threshold_filters(df, THRESHOLD_FILTERS)
    #     if df.empty:
    #         log("No rows left after threshold filtering; skipping file.")
    #         return

    groups = split_dataframe_with_all(df, SPLIT_COLUMNS)

    for group_key, group_df in groups.items():
        log(f"\nProcessing group: {group_key} (rows: {len(group_df)})")

        if 0 < GROUP_LIMIT_FRACTION < 1:
            group_df = sort_and_limit_rows(group_df, GROUP_LIMIT_FRACTION, POSSIBLE_GROUP_SORT_COLUMNS)
            log(f"After group limiting: {len(group_df)} rows")

        out_group_csv = os.path.join(OUTPUT_FOLDER, f"{base_name}__{group_key}.csv")
        try:
            group_df.to_csv(out_group_csv, index=False)
            # log(f"Saved trimmed group CSV: {out_group_csv}")
        except Exception as e:
            log(f"Failed to save group CSV {out_group_csv}: {e}")

        plot_subfolder = ensure_plot_subfolder(base_name, group_key)

        for target in TARGET_VARIABLES:
            if target not in group_df.columns:
                # log(f"Target '{target}' not in group; skipping.")
                continue

            uni_stats = compute_univariate_stats(group_df[target])
            stats_filename = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_stats.csv"
            stats_path = os.path.join(STATS_FOLDER, stats_filename)
            save_univariate_stats(uni_stats, stats_path)
            log(f"Saved univariate stats: {stats_path}")

            corrs = compute_correlations(group_df, target)
            corr_filename = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_correlations.csv"
            corr_path = os.path.join(CORR_FOLDER, corr_filename)
            save_correlations(corrs, corr_path)
            log(f"Saved correlations: {corr_path}")

            for col in group_df.columns:
                if col in TARGET_VARIABLES or col == 'ID' or col == 'fn' or group_df[col].nunique(dropna=False) == 1 \
                    or (col in corrs and abs(corrs[col]) < CORRELATION_THR):
                    log(f"Skipped: {col} at first plotting attempt")
                    continue

                treat_as_cat = is_categorical_for_plot(group_df[col], threshold=CATEGORICAL_UNIQUE_THRESHOLD)

                try:
                    plot_name = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_vs_{sanitize_for_filename(col)}__scatter.png"
                    plot_path = os.path.join(plot_subfolder, plot_name)
                    plot_scatter_with_regression_and_fixed_y(group_df, col, target, plot_path, y_range=Y_AXIS_RANGE)

                    if treat_as_cat:
                        plot_name = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_vs_{sanitize_for_filename(col)}__violin.png"
                        plot_path = os.path.join(plot_subfolder, plot_name)
                        plot_violin_and_strip(group_df, col, target, plot_path, y_range=Y_AXIS_RANGE)

                        plot_name = f"{base_name}__{group_key}__{sanitize_for_filename(target)}_vs_{sanitize_for_filename(col)}__box.png"
                        plot_path = os.path.join(plot_subfolder, plot_name)
                        plot_box_and_strip(group_df, col, target, plot_path, y_range=Y_AXIS_RANGE)

                except Exception as e:
                    log(f"Failed to create plot {plot_path}: {e}")


def run_pipeline():
    ensure_dirs()
    file_paths = ['CSV/joint_ph_charts/modf/cnc123b.csv', 'CSV/joint_ph_charts/modf/ph123b.csv']
    file_paths = ['CSV/joint_ph_charts/modf/cnc123b.csv']
    for path in file_paths:
        process_single_file(path)


if __name__ == "__main__":
    run_pipeline()
