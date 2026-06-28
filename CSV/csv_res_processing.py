import os
import pandas as pd

# --- Configuration ---
input_folder = "./csv_all"
output_folder = "./processed_csv_all2"

# Columns to remove
columns_to_remove = ["Created"]

# Filtering configuration
filter_column = "model_pth1"
filter_values_file = "allowed_values.txt"

# Splitting configuration
split_column = "val"   # column whose values define groups
group_name_map = {          # map raw values → output filename suffix
    "/users/project1/pt01299/synt/gda70/train/index_val.csv": "VALgda",
    "/users/project1/pt01299/synt/segformer_dataset255_all/val/index.csv": "VALs",
    "/users/project1/pt01299/synt/mix_val.csv": "VALm",
    "/users/project1/pt01299/synt/DK/osfstorage/dataset_v2/solardk_dataset_neurips_v2/gentofte_trainval/val/index.csv": "VALdk"
}

# Statistics output
stats_folder = "statistics2"


# ------------------ Helper Functions ------------------

def load_filter_values(path):
    """Load allowed values from a newline-separated file."""
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def clean_dataframe(df):
    """Apply all cleaning rules to a DataFrame."""
    # Remove selected columns
    df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors="ignore")

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    # Remove constant-value columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols)

    return df


def filter_dataframe(df, allowed_values):
    """Filter rows based on allowed values."""
    if filter_column in df.columns:
        df = df[df[filter_column].isin(allowed_values)]
    return df


def split_dataframe(df):
    """Split DataFrame into groups based on split_column."""
    if split_column not in df.columns:
        return {}

    groups = {}
    for value, group_df in df.groupby(split_column):
        if value in group_name_map:
            groups[group_name_map[value]] = group_df
    return groups


def save_statistics(df, filename):
    """Save descriptive statistics for a DataFrame."""
    os.makedirs(stats_folder, exist_ok=True)
    stats = df.describe(include="all")
    stats.to_csv(os.path.join(stats_folder, f"{filename}_stats.csv"))


# ------------------ Main Processing ------------------

def process_csv_files():
    os.makedirs(output_folder, exist_ok=True)

    # allowed_values = load_filter_values(filter_values_file)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".csv"):
            continue

        print(f"Processing: {filename}")

        input_path = os.path.join(input_folder, filename)
        df = pd.read_csv(input_path)

        # Clean + filter
        df = clean_dataframe(df)
        # df = filter_dataframe(df, allowed_values)

        # Split into groups
        groups = split_dataframe(df)

        # Save each group
        for group_name, group_df in groups.items():
            out_name = f"{os.path.splitext(filename)[0]}_{group_name}.csv"
            out_path = os.path.join(output_folder, out_name)

            group_df.to_csv(out_path, index=False)
            save_statistics(group_df, f"{os.path.splitext(filename)[0]}_{group_name}")

    print("Processing complete.")


if __name__ == "__main__":
    process_csv_files()
