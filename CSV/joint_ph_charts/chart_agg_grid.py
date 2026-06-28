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

cols = df.columns
cols = list(set(cols) & set(IOU_COLS)) + ['comb_key']

df = df[cols]

# 4. Group by the new 'comb_key' and aggregate all the numeric columns (e.g., taking the mean)
# df_agg = df.groupby('comb_key').max().reset_index()
# df_agg = df.groupby('comb_key').median().reset_index()
# df_agg = df.groupby('comb_key').mean().reset_index()
df_agg = df.groupby('comb_key').quantile(0.75).reset_index()

# 4. Melt and split column names
df_long = df_agg.melt(id_vars=['comb_key'], var_name='col_name', value_name='stat_mean')
df_long[['ph_nr', 'trg_var']] = df_long['col_name'].str.split('_', n=1, expand=True)
df_long['trg_var'] = df_long['trg_var'].str.split('_', expand=True)[1]
df_long['ph_nr'] = pd.to_numeric(df_long['ph_nr'])
df_long['gd'] = df_long['comb_key'].str.count('gda')

# 5. Plotting a Grid Setup
sns.set_theme(style="whitegrid")

# g = sns.relplot( # org grid
#     data=df_long,
#     kind="line",
#     x="ph_nr",             
#     y="stat_mean",         
#     # col="trg_var",         # Creates a column for each target (DK, GDA)
#     # col_order=["DK", "GDA", "SYNT"], # Optional: force column order
#     row="comb_key",          # Creates a row for each key
#     col="comb_key",
#     hue="trg_var",
#     color="b",             # A uniform color since keys are separated by row
#     markers=True,
#     dashes=False,
#     height=2.5,            # Reduced height since there will be many rows
#     aspect=1.5,
# )

# g = sns.relplot( # gda lvl
#     data=df_long,
#     kind="line",
#     x="ph_nr",             
#     y="stat_mean",         
#     col="trg_var",
#     col_order=["DK", "GDA", "SYNT"],
#     row="gd",
#     markers=True,
#     dashes=False,
#     height=2.5,
#     aspect=1.5,
# )

# g = sns.catplot( # gda lvl cat
#     data=df_long,
#     kind="bar",
#     x="ph_nr",             
#     y="stat_mean",         
#     col="trg_var",
#     col_order=["DK", "GDA", "SYNT"],
#     row="gd",
#     height=2.5,
#     aspect=1.5,
# )

# g = sns.catplot( # gda lvl cat hue
#     data=df_long,
#     kind="bar",
#     x="ph_nr",             
#     y="stat_mean",         
#     col="trg_var",
#     col_order=["DK", "GDA", "SYNT"],
#     hue="gd",
#     height=2.5,
#     aspect=1.5,
# )

# g = sns.relplot( # gda lvl T
#     data=df_long,
#     kind="line",
#     x="ph_nr",             
#     y="stat_mean",         
#     col="gd",
#     row_order=["DK", "GDA", "SYNT"],
#     row="trg_var",
#     markers=True,
#     dashes=False,
#     height=2.5,
#     aspect=1.5,
# )


# g = sns.catplot( # gda lvl T cat
#     data=df_long,
#     kind="bar",
#     x="ph_nr",             
#     y="stat_mean",         
#     col="gd",
#     row="trg_var",
#     row_order=["DK", "GDA", "SYNT"],
#     height=2.5,
#     aspect=1.5,
# )


# g = sns.catplot( # gda lvl T cat hue
#     data=df_long,
#     kind="bar",
#     x="ph_nr",             
#     y="stat_mean",         
#     col="gd",
#     hue="trg_var",
#     # row_order=["DK", "GDA", "SYNT"],
#     height=2.5,
#     aspect=1.5,
# )


# 6. Formatting
g.set(ylim=(0.55, 0.85))
g.set_axis_labels("Phase", "IoU")

# Clean up titles so they don't redundantly say "comb_key=" or "trg_var="
g.set_titles(row_template="{row_name}", col_template="{col_name}")

# g.set(title='IoUs by amount of gda in setups')

# Adjust spacing so the titles don't overlap the charts above them
plt.subplots_adjust(hspace=0.4, wspace=0.1)

plt.show()
