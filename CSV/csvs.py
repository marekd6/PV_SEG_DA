import os
import pandas as pd


input_folder = "./csv_results_wandb2"
input_folder = "./csv_results_wandb_ph2"
output_folder = "./processed_csv_ph2"

columns_to_remove = ["Created"]


def process_csv_files():
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing: {filename}")

            df = pd.read_csv(input_path)

            # Remove selected columns
            df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors="ignore")

            # Remove empty columns (all NaN)
            df = df.dropna(axis=1, how="all")

            # Remove columns where all values are identical
            constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
            df = df.drop(columns=constant_cols)

            # Save cleaned CSV
            df.to_csv(output_path, index=False)

    print("Processing complete.")


if __name__ == "__main__":
    process_csv_files()
