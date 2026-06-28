import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK

df = pd.read_csv('CSV/joint_ph_charts/modf/ph123b.csv')
# df = pd.read_csv('CSV/joint_ph_charts/modf/proc_ph1.csv')

cols = df.columns
group_key = 'comb_key'
# group_key = 'tr_val'
cols = list(set(cols) & set(IOU_COLS)) + [group_key]

df = df[cols]

# 4. Group by the new group_key and aggregate all the numeric columns (e.g., taking the mean)
df_agg = df.groupby(group_key).max().reset_index()
# df_agg = df.groupby(group_key).median().reset_index()
# df_agg = df.groupby(group_key).mean().reset_index()
# df_agg = df.groupby(group_key).quantile(0.75).reset_index()

# 5. Melt the dataframe
# This takes all columns EXCEPT group_key and turns them into two columns: 
# 'col_name' (e.g., "1_DK") and 'mean' (the actual value)
df_long = df_agg.melt(
    id_vars=[group_key],         # Keep comb_key as the row identifier
    var_name='col_name',        # The old column headers will go here
    value_name='mean'      # The cell values will go here
)

# 6. Decouple the column names into your plot variables
# This splits "1_DK" into "1" (ph_nr) and "DK" (trg_var) based on the underscore
df_long[['ph_nr', 'trg_var']] = df_long['col_name'].str.split('_', n=1, expand=True)

df_long['trg_var'] = df_long['trg_var'].str.split('_', expand=True)[1]

# Important: Convert the phase number back to an integer so the X-axis scales correctly
df_long['ph_nr'] = pd.to_numeric(df_long['ph_nr'])

# df_long['gd'] = 'gda' in df_long[group_key]

# 7. Plotting
sns.set_theme(style="whitegrid")
g = sns.relplot(
    data=df_long,
    kind="line",
    x="ph_nr",             # The extracted phase number (1, 2, 3)
    y="mean",         # The aggregated values
    hue=group_key,          # Grouping comb_key
    style=group_key,        
    col="trg_var",         # The extracted target variable (DK, GDA)
    col_order=["DK", "GDA", "SYNT"],
    markers=True,
    dashes=False,
    # height=4,
    # aspect=1.2,
)

# Formatting
g.set(ylim=(0.55, 0.85))
g.set_axis_labels("phase", "IoU")
g.set_titles("{col_name}")
# g.despine(left=True, bottom=False)

# sns.move_legend(
#     g, "lower center",
#     # g, "lower",
#     # bbox_to_anchor=(0.5, -0.15), 
#     ncol=6,                 
#     title=None, frameon=False
# )

# plt.tight_layout()
plt.show()
