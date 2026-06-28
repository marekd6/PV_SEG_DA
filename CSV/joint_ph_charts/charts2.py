import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
# df = pd.read_csv('dff.csv')
# df_long = pd.read_csv('df_long_all_SYNT.csv')
# df_long = pd.read_csv('df_long_all_GDADK.csv')
# df_long = pd.read_csv('df_long_all_DK.csv')
# df_long = pd.read_csv('df_long_all_total.csv')
# df_long = pd.read_csv('df_long_SYNT.csv')
# df_long = pd.read_csv('df_long_GDADK.csv')
# df_long = pd.read_csv('df_long_DK.csv')
# df_long = pd.read_csv('df_long_total.csv')
df_long = pd.read_csv('CSV/joint_ph_charts/modf/long.csv')

# # 2. Reshape the data: Split the rows by phase (_x, _y, and blank)
# # We pull the specific columns for each phase and rename them so they stack perfectly
# df_x = df[['comb_key', 'trg_var', 'ph_nr_x', 'stat_mean_x']].copy()
# df_x.columns = ['comb_key', 'trg_var', 'ph_nr', 'stat_mean']

# df_y = df[['comb_key', 'trg_var', 'ph_nr_y', 'stat_mean_y']].copy()
# df_y.columns = ['comb_key', 'trg_var', 'ph_nr', 'stat_mean']

# df_blank = df[['comb_key', 'trg_var', 'ph_nr', 'stat_mean']].copy()

# # Combine (stack) them all together into one long format dataframe
# df_long = pd.concat([df_x, df_y, df_blank], ignore_index=True)

# # 3. Clean the comb_key to create the "series"
# # .str.rsplit('_', n=1).str[0] splits the string at the very last underscore and removes it (e.g., dropping _DK)
# df_long['series'] = df_long['comb_key'].str.rsplit('_', n=1).str[0]


# df_long.to_csv('df_long.csv')

# 4. Plot the newly formatted data
sns.set_theme(style="whitegrid")

g = sns.relplot(
    data=df_long,
    kind="line",
    x="ph", # "ph_nr",             # The phase numbers (now all nicely in one column: 1, 2, 3)
    y="meanGDA", # "stat_mean",         # The values
    # hue="series",          # Colors by your cleaned comb_key
    hue="comb_key",          # Colors by your cleaned comb_key
    style="comb_key",        # Automatically gives each series a unique marker shape
    # style="series",        # Automatically gives each series a unique marker shape
    # col="trg_var",         # Creates the DK / GDA panels
    markers=True,
    dashes=False,          # Keeps lines solid
    height=4,
    aspect=1.2,
)

# 5. Visual Cleanup
# g.set(ylim=(0.4, 0.85))
g.set_axis_labels("", "")
g.set_titles("{col_name}")
g.despine(left=True, bottom=False)

# Move legend to bottom
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(0.5, -0.15), 
    ncol=3,                # You can adjust ncol depending on how wide your legend text is
    title=None, frameon=False
)

plt.tight_layout()
plt.show()
