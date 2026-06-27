import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the raw data
df = pd.read_csv('prt.csv')

# 2. Extract the base series name (dropping the _DK, _GDA, etc. at the end)
df['series'] = df['comb_key'].str.rsplit('_', n=1).str[0]

# 3. Group by the new 'series' and 'trg_var', and aggregate
# (Even if rows are unique, this safely handles duplicates and extracts only the columns we need)
df_agg = df.groupby(['series', 'trg_var']).agg({
    'ph_nr_x': 'first', 'stat_mean_x': 'mean',
    'ph_nr_y': 'first', 'stat_mean_y': 'mean',
    'ph_nr': 'first',   'stat_mean': 'mean'
}).reset_index()

# 4. Standardize suffixes so Pandas knows which columns belong together
df_agg = df_agg.rename(columns={
    'ph_nr_x': 'ph_nr_1', 'stat_mean_x': 'stat_mean_1',
    'ph_nr_y': 'ph_nr_2', 'stat_mean_y': 'stat_mean_2',
    'ph_nr': 'ph_nr_3',   'stat_mean': 'stat_mean_3'
})

# 5. Decouple columns into rows (Unpivot / Wide-to-Long)
# This automatically grabs everything ending in _1, _2, _3 and collapses them into 'ph_nr' and 'stat_mean' rows
df_long = pd.wide_to_long(
    df_agg, 
    stubnames=['ph_nr', 'stat_mean'], # The root names of the columns to extract
    i=['series', 'trg_var'],          # The columns to keep as identifiers (grouping variables)
    j='phase',                        # The new column that will hold 1, 2, or 3
    sep='_'                           # The separator before the suffix
).reset_index()

# 6. Plotting
sns.set_theme(style="whitegrid")
g = sns.relplot(
    data=df_long,
    kind="line",
    x="ph_nr",             # Extracted x-axis (1, 2, 3)
    y="stat_mean",         # Aggregated values
    hue="series",          # Grouping series
    style="series",        
    col="trg_var",         # Extracted column variable (DK, GDA)
    markers=True,
    dashes=False,
    height=4,
    aspect=1.2,
)

# Formatting
g.set(ylim=(0, 0.9))
g.set_axis_labels("", "")
g.set_titles("{col_name}")
g.despine(left=True, bottom=False)

sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(0.5, -0.15), 
    ncol=3,                 
    title=None, frameon=False
)

plt.tight_layout()
plt.show()
