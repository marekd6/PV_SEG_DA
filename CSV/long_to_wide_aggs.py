import pandas as pd 

inp = 'joint_ph_charts/ph123_avgs.csv'
outp = ''

df = pd.read_csv(inp)

# wide = df.pivot(index=['comb_key', ], columns="variable", values="value").reset_index()
# wide = df.pivot_table(index=['comb_key', ], columns="variable", values="value").reset_index()

# Combine phase and test set into the column names
df["phase_test"] = "phase" + df["phase"].astype(str) + "_" + df["test set"]


wide = df.pivot(
    index="comb_key",
    columns="phase_test",
    values="IoU_mean"
).reset_index()

# Remove the columns index name
wide.columns.name = None

# Save
wide.to_csv("wide_3ph.csv", index=False)

