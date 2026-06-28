import pandas as pd
import matplotlib
matplotlib.use('Agg') # Forces non-interactive file-rendering backend
import matplotlib.pyplot as plt
import seaborn as sns

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK

SAVE_DIR = 'CSV/joint_ph_charts/modf'

FILES = {
    'joint': 'CSV/joint_ph_charts/modf/ph123b.csv',
    'ph1': 'CSV/joint_ph_charts/modf/proc_ph1.csv',
    # 'ph2': 'CSV/joint_ph_charts/modf/proc_ph2.csv',
    # 'ph3': 'CSV/joint_ph_charts/modf/proc_ph3.csv',
    'concat': 'CSV/joint_ph_charts/modf/cnc123b.csv',
}

STATS = ['mean', 'max', 'median', '75']

KEYS = ['comb_key', 'tr_val']


def rel_1(df_long, group_key):
    return sns.relplot(
        data=df_long,
        kind="line",
        x="ph_nr",
        y="stat_mean",
        hue=group_key,
        style=group_key,        
        col="trg_var",
        col_order=["DK", "GDA", "SYNT"],
        markers=True,
        dashes=False,
        # height=4,
        # aspect=1.2,
    )


def rel_grid(df_long, group_key):
    return sns.relplot( # org grid
        data=df_long,
        kind="line",
        x="ph_nr",             
        y="stat_mean",         
        row=group_key,
        col=group_key,
        hue="trg_var",
        color="b",
        markers=True,
        dashes=False,
        # height=2.5,
        # aspect=1.5,
    )


def gd_grid(df_long, group_key):
    return sns.catplot( # gda lvl cat
        data=df_long,
        kind="bar",
        x="ph_nr",             
        y="stat_mean",         
        col="trg_var",
        col_order=["DK", "GDA", "SYNT"],
        row="gd",
        height=2.5,
        aspect=1.5,
    )


def gd_grid_hue(df_long, group_key):
    return sns.catplot( # gda lvl cat
        data=df_long,
        kind="bar",
        x="ph_nr",             
        y="stat_mean",         
        col="trg_var",
        col_order=["DK", "GDA", "SYNT"],
        hue="gd",
        height=2.5,
        aspect=1.5,
    )


def gd_grid_T(df_long, group_key):
    return sns.relplot( # gda lvl T
        data=df_long,
        kind="line",
        x="ph_nr",             
        y="stat_mean",         
        col="gd",
        row_order=["DK", "GDA", "SYNT"],
        row="trg_var",
        markers=True,
        dashes=False,
        height=2.5,
        aspect=1.5,
    )


def gd_grid_T_bar(df_long, group_key):
    return sns.catplot( # gda lvl T cat
        data=df_long,
        kind="bar",
        x="ph_nr",             
        y="stat_mean",         
        col="gd",
        row="trg_var",
        row_order=["DK", "GDA", "SYNT"],
        height=2.5,
        aspect=1.5,
    )


def gd_grid_T_bar_hue(df_long, group_key):
    return sns.catplot( # gda lvl T cat hue
        data=df_long,
        kind="bar",
        x="ph_nr",             
        y="stat_mean",         
        col="gd",
        hue="trg_var",
        height=2.5,
        aspect=1.5,
    )


def cnc_T_bar_hue(df_long, group_key):
    return sns.catplot(
        data=df_long,
        kind="bar",
        x="ph_nr",             
        y="stat_mean",         
        hue=group_key,
        col="trg_var",
        height=2.5,
        aspect=1.5,
    )


def ph1_T_bar_hue(df_long, group_key):
    return sns.catplot(
        data=df_long,
        kind="bar",
        x=group_key,             
        y="stat_mean",         
        hue=group_key,
        col="trg_var",
        height=2.5,
        aspect=1.5,
    )


def ph123_T_bar_hue(df_long, group_key):
    return sns.catplot(
        data=df_long,
        kind="bar",
        x=group_key,             
        y="stat_mean",         
        hue=group_key,
        col="trg_var",
        row='ph_nr',
        height=2.5,
        aspect=1.5,
    )


CHARTS = {
    'cnc_T_bar_hue': cnc_T_bar_hue,
    'rel': rel_1,
    # 'rel_grid': rel_grid,
    'gd_grid': gd_grid,
    'gd_grid_hue': gd_grid_hue,
    'gd_grid_T': gd_grid_T,
    'gd_grid_T_bar': gd_grid_T_bar,
    'gd_grid_T_bar_hue': gd_grid_T_bar_hue,
}


CHARTS = {
    'ph1_T_bar_hue': ph1_T_bar_hue,
    'ph123_T_bar_hue': ph123_T_bar_hue,
}


def agg_df(df: pd.DataFrame, stat, group_key):
    if stat == 'mean':
        return df.groupby(group_key).mean().reset_index()
    elif stat == 'max':
        return df.groupby(group_key).max().reset_index()
    elif stat == 'median':
        return df.groupby(group_key).median().reset_index()
    elif stat == '75':
        return df.groupby(group_key).quantile(0.75).reset_index()
    return df
  

def plot_file(f, stat='mean', group_key = 'comb_key', chart='rel'):
    pth = FILES[f]
    df = pd.read_csv(pth)
    cols = df.columns
    cols = list(set(cols) & set(IOU_COLS)) + [group_key]
    df = df[cols]

    df_agg = agg_df(df, stat, group_key)

    df_long = df_agg.melt(id_vars=[group_key], var_name='col_name', value_name='stat_mean')
    df_long[['ph_nr', 'trg_var']] = df_long['col_name'].str.split('_', n=1, expand=True)
    df_long['trg_var'] = df_long['trg_var'].str.split('_', expand=True)[1]
    df_long['ph_nr'] = pd.to_numeric(df_long['ph_nr'])
    df_long['gd'] = df_long[group_key].str.count('gda')

    sns.set_theme(style="whitegrid")
    g = CHARTS[chart](df_long, group_key)

    # g.set(ylim=(0.55, 0.85))
    g.set_axis_labels("Phase", "IoU")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    # g.set(title='IoUs by amount of gda in setups')

    if group_key == 'tr_val':
        try:
            sns.move_legend(
                g, "lower center",
                # g, "lower",
                bbox_to_anchor=(0.5, -0.15), 
                ncol=6,                 
                title=None, frameon=False
            )
        except Exception as e:
            print(e)

    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    # plt.show()

    fn = '_'.join([f, stat, chart, group_key])
    plt.savefig(f'{SAVE_DIR}/{fn}')
    print('saved', fn)
    plt.close()


if __name__ == '__main__':
    for f, pth in FILES.items():
        for p in CHARTS.keys():
            for st in STATS:
                plot_file(f, st, 'tr_val', p)
