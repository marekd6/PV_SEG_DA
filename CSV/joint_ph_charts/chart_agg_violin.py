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

df = pd.read_csv('csv_all_jjj/modf/ph123b.csv')

cols = df.columns
cols = list(set(cols) & set(IOU_COLS)) + ['comb_key']

df = df[cols]

# 2. Melt the raw dataframe directly
df_raw_long = df.melt(id_vars=['comb_key'], var_name='col_name', value_name='stat_mean')
df_raw_long[['ph_nr', 'trg_var']] = df_raw_long['col_name'].str.split('_', n=1, expand=True)
df_raw_long['trg_var'] = df_raw_long['trg_var'].str.split('_', expand=True)[1]
df_raw_long['ph_nr'] = pd.to_numeric(df_raw_long['ph_nr'])
df_raw_long['gd'] = df_raw_long['comb_key'].str.count('gda')


# 3. Draw the Violin Grid
sns.set_theme(style="whitegrid")
# g = sns.catplot( # org grid - gda lvl
#     data=df_raw_long,
#     kind="violin", 
#     x="ph_nr",             
#     y="stat_mean",         
#     col="trg_var",         
#     # row="comb_key",          
#     row="gd",          
#     col_order=["DK", "GDA", "SYNT"],
#     color="skyblue",
#     inner="quartile",
#     height=2.5,            
#     aspect=1.5,
# )

g = sns.catplot( # gda lvl T
    data=df_raw_long,
    kind="violin",
    x="ph_nr",             
    y="stat_mean",         
    row="trg_var",         
    col="gd",          
    row_order=["DK", "GDA", "SYNT"],
    color="skyblue",
    inner="quartile",
    height=2.5,            
    aspect=1.5,
)

# g.set(ylim=(0.4, 0.9))
g.set_axis_labels("Phase", "IoU")
g.set_titles(row_template="{row_name}", col_template="{col_name}")
plt.subplots_adjust(hspace=0.4, wspace=0.1)
plt.show()
