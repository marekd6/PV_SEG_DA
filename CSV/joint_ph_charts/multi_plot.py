import pandas as pd
# import matplotlib
# matplotlib.use('Agg') # Forces non-interactive file-rendering backend
import matplotlib.pyplot as plt
import seaborn as sns

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK

# SAVE_DIR = 'CSV/joint_ph_charts/modf'
SAVE_DIR = 'CSV/joint_ph_charts/modf_rntg'

FILES = {
    'joint': f'{SAVE_DIR}/ph123b.csv',
    # 'ph1': f'{SAVE_DIR}/proc_ph1.csv',
    # 'ph2': f'{SAVE_DIR}/proc_ph2.csv',
    # 'ph3': f'{SAVE_DIR}/proc_ph3.csv',
    # 'concat': f'{SAVE_DIR}/cnc123b.csv',
}

STATS = ['mean', 'max', 'median', '75']
STATS = ['mean', '75']
STATS = ['mean']

KEYS = ['comb_key', 'tr_val']
KEYS = ['comb_key']


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
        dashes=False, # TODO: ph_nr as int cat, CI for mean
        # height=4,
        # aspect=1.2,
    )


def rel_2(df_long, group_key):
    return sns.catplot(
        data=df_long,
        # kind="point",
        kind="bar",
        x="ph_nr",
        y="stat_mean",
        hue=group_key,
        # style=group_key,        
        col="trg_var",
        col_order=["DK", "GDA", "SYNT"],
        # markers=True,
        # dashes=False, # TODO: ph_nr as int cat, CI for mean
        # height=4,
        # aspect=1.2,
    )


def rel_2_pnt(df_long, group_key):
    return sns.catplot(
        data=df_long,
        kind="point",
        # kind="bar",
        x="ph_nr",
        y="stat_mean",
        hue=group_key,
        # style=group_key,        
        col="trg_var",
        col_order=["DK", "GDA", "SYNT"],
        markers=True,
        dashes=False, # TODO: ph_nr as int cat, CI for mean
        # height=4,
        # aspect=1.2,
    )


def rel_2_T(df_long, group_key):
    return sns.catplot(
        data=df_long,
        kind="bar",
        x="trg_var",
        y="stat_mean",
        hue=group_key,
        # style=group_key,        
        col="ph_nr",
        # col_order=["DK", "GDA", "SYNT"],
        # markers=True,
        # dashes=False, # TODO: ph_nr as int cat, CI for mean
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
        y=group_key,             
        x="stat_mean",         
        hue=group_key,
        row="trg_var",
        col='ph_nr',
        height=2.5,
        aspect=1.5,
    )
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

def rel_1_x(df_long, group_key, bs=[0.617, 0.359, 0.71]): # GDA, SYNT, DK
    g = rel_1(df_long, group_key) \
        .map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
    g = rel_1(df_long, group_key)
    g.set(ylim=(0.35, 1.0))
    for ax, b in zip(g.axes.flatten(), bs):
        ax.axhline(b, ls='--')
    return g


def rel_2_x(df_long, group_key, bs=[0.71, 0.617, 0.359]): # GDA, SYNT, DK "DK", "GDA", "SYNT"
    # g = rel_2(df_long, group_key) \
    #     .map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
    g = rel_2(df_long, group_key)
    g.set(ylim=(0.35, 1.0))
    for ax, b in zip(g.axes.flatten(), bs):
        ax.axhline(b, ls='--')
    return g


def rel_2_x_T(df_long, group_key, bs=[0.71, 0.617, 0.359]): # GDA, SYNT, DK "DK", "GDA", "SYNT"
    g = rel_2_T(df_long, group_key)
    g.set(ylim=(0.35, 1.0))
    # for ax, b in zip(g.axes.flatten(), bs):
        # ax.axhline(b, ls='--')
    return g


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

CHARTS1 = {
    'rel': rel_1,
    # 'rel_2_pnt': rel_2_pnt,
    'rel_2': rel_2,
    'rel_1_x': rel_1_x,
    'rel_2_x': rel_2_x,
    'rel_2_x_T': rel_2_x_T,
    'ph123_T_bar_hue': ph123_T_bar_hue,
}

CHARTS2 = {
    # 'rel_2_pnt': rel_2_pnt,
    'rel_2': rel_2,
    'rel_1_x': rel_1_x,
    'rel_2_x': rel_2_x,
    'rel_2_x_T': rel_2_x_T,
    'ph123_T_bar_hue': ph123_T_bar_hue,
}

CHARTS = CHARTS1.copy()
CHARTS.update(CHARTS2)


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
    df_long['ph_nr'] = pd.to_numeric(df_long['ph_nr'], downcast='integer')
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
    plt.show()

    # fn = '_'.join([f, stat, chart, group_key])
    # plt.savefig(f'{SAVE_DIR}/{fn}')
    # print('saved', fn)
    # plt.close()
  

def plot_file_raw(f, stat='mean', group_key = 'comb_key', chart='rel'):
    pth = FILES[f]
    df = pd.read_csv(pth)
    cols = df.columns
    print(cols)
    cols = list(set(cols) & set(IOU_COLS)) + [group_key]
    df = df[cols]

    df_agg = df # agg_df(df, stat, group_key)
    print(df_agg.columns)
    print(df_agg.size)
    print(df.head())

    df_long = df_agg.melt(id_vars=[group_key], var_name='col_name', value_name='stat_mean')
    df_long[['ph_nr', 'trg_var']] = df_long['col_name'].str.split('_', n=1, expand=True)
    df_long['trg_var'] = df_long['trg_var'].str.split('_', expand=True)[1]
    df_long['ph_nr'] = pd.to_numeric(df_long['ph_nr'], downcast='integer')
    # df_long['gd'] = df_long[group_key].str.count('gda')
    if group_key == 'tr_val':
        df_long['gd'] = True
    else:
        df_long['gdstr'] = df_long[group_key].str.split('|', expand=True)[0]
        # df_long['gd'] = 'gda' in df_long['gdstr']
        df_long['gd'] = ['gda' in x for x in df_long['gdstr']]
        # df_long['gd'] = df_long[group_key].str.split('|')[0].count('gda')
        # df_long['gd1'] = False
        # df_long.loc['gda' in df_long[group_key].str.split('|')[0], 'gd1'] = True
    # print(df_long.columns)

    df_long_org = df_long.copy()

    for t in [True, False]:
        df_long = df_long_org[df_long_org['gd'] == t]
        
        # print(df_long.size)
        # print(df_long.head())

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
        plt.show()

        # fn = '_'.join([f, stat, chart, group_key])
        # plt.savefig(f'{SAVE_DIR}/{fn}')
        # print('saved', fn)
        # plt.close()


def p1():
    for f, pth in FILES.items():
        for p in CHARTS1.keys():
            for st in STATS:
                for k in KEYS:
                    print(f, st, k, p)
                    plot_file(f, st, k, p)

def p2():
    KEYS = ['comb_key', 'tr_val']
    for f, pth in FILES.items():
        for p in CHARTS2.keys():
            for st in STATS:
                for k in KEYS:
                    print(f, st, k, p)
                    plot_file_raw(f, st, k, p)

if __name__ == '__main__':
    p1()
    p2()                  
