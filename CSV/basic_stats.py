import pandas as pd

input_file = 'joint_ph_charts/res_dfs7/proc_ph1.csv'
output_file = "statistics_ph1.csv"

df = pd.read_csv(input_file)

selected_columns = ['1_test_DK_iou', '1_test_GDA_iou', '1_test_SYNT_iou']

result = pd.DataFrame({
    "Q1": df[selected_columns].quantile(0.25),
    "Q2": df[selected_columns].quantile(0.50),
    "Q3": df[selected_columns].quantile(0.75),
    "Q4": df[selected_columns].quantile(1.00),
    "Mean": df[selected_columns].mean(),
    "Std": df[selected_columns].std()
})

result.index.name = "Set"

# Save as CSV
result.to_csv(output_file)

print(f"Statistics saved to {output_file}")
print(result)
