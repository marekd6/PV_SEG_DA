import os
import pandas as pd

stats_folder = "statistics2"
stats_folder = "statistics30"
output_file = "merged_statistics30.csv"

# Columns we are interested in (in priority order)
possible_columns = ["3_test_GDA_iou", "1_test/GDA/iou", "2_test/GDA/iou", "3_test/GDA/iou", 
                    "1_test_GDA_iou", "2_test_GDA_iou"]
statistics_to_extract = ["mean", "std", "min", "max", "count", "25%", "50%", "75%", "top"]


def extract_values(stats_df):
    """Return (column_name, {stat: value}) for the first matching column."""
    for col in possible_columns:
        if col in stats_df.columns:
            extracted = {}
            for stat in statistics_to_extract:
                if stat in stats_df.index:
                    extracted[stat] = stats_df.loc[stat, col]
                else:
                    extracted[stat] = None
            return col, extracted
    return None, {stat: None for stat in statistics_to_extract}


def merge_statistics():
    rows = []

    for filename in os.listdir(stats_folder):
        if not filename.endswith("_stats.csv"):
            continue

        path = os.path.join(stats_folder, filename)
        stats_df = pd.read_csv(path, index_col=0)

        col, values = extract_values(stats_df)

        row = {
            "file": filename.replace("_stats.csv", ""),
            "column_used": col
        }

        # Add extracted statistics
        for stat, val in values.items():
            row[stat] = val

        rows.append(row)

    merged_df = pd.DataFrame(rows)
    merged_df.to_csv(output_file, index=False)
    print("Merged statistics written to:", output_file)


if __name__ == "__main__":
    merge_statistics()
