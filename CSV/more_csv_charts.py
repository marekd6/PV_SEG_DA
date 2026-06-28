import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your actual data
df = pd.read_csv('absa.csv', sep=';', decimal=',')

# --- MOCK DATA (Just so you can run and test this code immediately) ---
# data = {
#     'Main_Category': ['DK']*6 + ['GDA']*6 + ['SYNT']*6,
#     'Sub_Category':  [1, 1, 2, 2, 3, 3] * 3,
#     'Series':        ['subs_16A', 'subs_m'] * 9,
#     'Value':         [0.5, 0.78, 0.76, 0.8, 0.8, 0.8,  # DK
#                       0.58, 0.7, 0.72, 0.74, 0.74, 0.74, # GDA
#                       0.8, 0.65, 0.78, 0.8, 0.8, 0.8]  # SYNT
# }
# df = pd.DataFrame(data)
# ----------------------------------------------------------------------

# 2. Set the visual style (similar to the clean Excel look)
sns.set_theme(style="whitegrid")

# 3. Create the point plot
# kind="point" draws the markers AND connects them with lines across the X-axis
g = sns.relplot(
    data=df,
    kind="line",           # <--- This is the magic parameter for connecting dots!
    x="nr",       # The 1, 2, 3 axis
    # y="stat_mean",              # The 0 to 0.9 axis
    y=" stat_mean ",              # The 0 to 0.9 axis
    hue="source_file",           # Your different data lines (ph2_dk_dk, etc.)
    style="source_file",           # Your different data lines (ph2_dk_dk, etc.)
    col="trg",    # Splits the chart into DK, GDA, SYNT panels
    markers=True,
    dashes=False, #["o", "^", "s", "D", "X", "v", "P", "*", "h", "+", "x", "d"], # Different shapes
    # linestyles="-",         # Solid lines joining the points
    height=4,               
    aspect=1.2,             
    # palette="tab10"         # A good color palette for many series
)

# 4. Customize the Y-axis to match your image (0 to 0.9)
g.set(ylim=(0, 0.9))

# 5. Clean up labels and titles
g.set_axis_labels("", "")               # Removes default X/Y axis titles for a cleaner look
g.set_titles("{col_name}")              # Sets the panel titles to just "DK", "GDA", "SYNT"
g.despine(left=True, bottom=False)

# 6. Move the legend to the bottom (like in your screenshot)
# sns.move_legend(
#     g, "lower center",
#     bbox_to_anchor=(0.5, -0.15), 
#     ncol=6,                 # Spreads the legend into multiple columns
#     title=None, frameon=False
# )

# 7. Adjust layout so the legend isn't cut off
plt.tight_layout()

# 8. Show the plot
plt.show()
