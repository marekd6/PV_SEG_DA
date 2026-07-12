'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


import pandas as pd
import numpy as np
import seaborn as sns
import re
import itertools
import math

SAVING = True
SAVING = False

if SAVING:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK

DIR = 'CSV/joint_ph_charts/modf_rntg2'
SAVEDIR = 'CSV/joint_ph_charts/selected'

FILES = {
    'joint': f'{DIR}/ph123b.csv',
    'ph1': f'{DIR}/proc_ph1.csv',
    'ph2': f'{DIR}/proc_ph2.csv',
    'ph3': f'{DIR}/proc_ph3.csv',
    'concat': f'{DIR}/cnc123b.csv',
}

DS_SIZES_TR = { # TODO
    's': 123,
    'dk': 789,
    'gda': 18,
}

DS_SIZES_VAL = { # TODO
    's': 123,
    'dk': 789,
    'gda': 20,
}

S_SUB_SIZES = { # TODO
    'mix': 456,
    'composite': 546
}

DS_DOM_SIM = {
    's': 0.75,
    'sub': 0.75,
    'gda': 1,
    'dk': 0,
    'm': 0.25,
    'subm': 0.5,
}

DS_REAL = {
    's': 0,
    'sub': 0,
    'gda': 1,
    'dk': 1,
    'm': 0.5,
    'subm': 0.25,
}

DS_DOM_REAL = {
    's': [0.75, 0],
    'sub': [0.75, 0],
    'gda': [1, 1],
    'dk': [0, 1],
    'm': [0.25, 0.5],
    'subm': [0.5, 0.25],
}

RNTMS = ['1_cum_Runtime', '2_cum_Runtime', '3_cum_Runtime']


def _clean_df_(df: pd.DataFrame, h=['comb_key']):
    cols = df.columns
    cols = list(set(cols) & set(IOU_COLS)) + h
    return df[cols]


def process_runtime(df: pd.DataFrame, h):
    '''
    cols
    '''
    def rename_column(col):
        m = re.match(r"(\d+)_test_([A-Z]+)_iou$", col) # Match: phase_test_SET_t
        if m:
            phase, set_name = m.groups()
            return f"iou_{set_name}_{phase}"
        m = re.match(r"(\d+)_cum_Runtime$", col) # Match: phase_cum_m
        if m:
            phase = m.group(1)
            return f"Runtime_{phase}"
        return col
    
    df = df.rename(columns=rename_column)
    df = df.drop(columns=['Runtime'])
    df = df.reset_index(names="row_id")
    return (pd.wide_to_long(df, stubnames=['iou_SYNT', 'iou_GDA', 'iou_DK', 'Runtime'], 
                            i=['row_id', h], j='phase', sep='_', suffix=r"\d+")).reset_index()


def process_workload(df: pd.DataFrame):
    workload_factor = 0.1
    df['workload'] = 0
    pass


def data_diversity_space():
    for pt in DS_DOM_REAL.values():
        print(math.dist(pt, [0 ,0]))
    fff = list(zip(DS_DOM_REAL.values(), DS_DOM_REAL.keys()))
    print(fff[0])
    df = pd.DataFrame(DS_DOM_REAL.values(), columns=['domain', 'real'])
    df['key'] = list(DS_DOM_REAL.keys())
    df['d'] = [math.dist(pt, [0,0]) for pt in DS_DOM_REAL.values()]
    df.set_index('key').sort_index()
    print(df.head())
    g = sns.scatterplot(df, x='real', y='domain', hue='key', size='d')
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    # sns.scatterplot(list(zip(DS_DOM_REAL.values(), DS_DOM_REAL.keys()))) # TODO
    # # sns.scatterplot(data=list(DS_DOM_REAL.values()))
    # plt.grid(visible=True, which='major')
    plt.show()


# def data_diversity_space():
#     d = zip(DS_DOM_SIM.values(), DS_REAL.values())
#     print(list(d))
#     # d = itertools.product(DS_DOM_SIM.values(), DS_REAL.values())
#     for pt in d:
#         print(math.dist(pt, (0, 0)))
#     d = zip(DS_DOM_SIM.values(), DS_REAL.values())
#     # sns.scatterplot(data=zip(*d), hue=DS_DOM_SIM.keys())
#     # sns.scatterplot(data=list(zip(*d, DS_DOM_SIM.keys())))
#     sns.scatterplot(data=list(zip(*d)))
#     plt.grid(visible=True, which='major')
#     plt.show()


def process_diversity(df: pd.DataFrame):
    zero = np.array([0, 0])
    df[['d_dom1', 'd_real1']] = df['train_x'].map(DS_DOM_REAL).tolist() # val_x
    df[['d_dom2', 'd_real2']] = df['train_y'].map(DS_DOM_REAL).tolist()
    df[['d_dom3', 'd_real3']] = df['train'].map(DS_DOM_REAL).tolist()
    df['d123inner'] = np.sqrt((df['d_dom1'] + df['d_dom2'] + df['d_dom3'] - zero[0])**2 + 
                              (df['d_real1'] + df['d_real2'] + df['d_real3'] - zero[1])**2)
    df['d123outer'] = np.sum([np.sqrt((df['d_dom1']- zero[0])**2 + (df['d_real1'] - zero[1])**2), 
                          np.sqrt((df['d_dom2']- zero[0])**2 + (df['d_real2'] - zero[1])**2),
                          np.sqrt((df['d_dom3']- zero[0])**2 + (df['d_real3'] - zero[1])**2)], axis=0)
    df['ddiff'] = (df['d123outer'] - df['d123inner']) / df['d123inner'] * 100
    print(df[['d123inner', 'd123outer', 'ddiff']].head())


def widen_runtime(df: pd.DataFrame, h):
    df = pd.melt(df, id_vars=[h, 's_lvl', 'phase', 'Runtime'], # TODO s_lvl
                      value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], 
                      var_name='test data', value_name='iou')
    df['test data'] = df['test data'].str.replace('iou_', '')
    ho = df[h].unique()
    ho.sort()
    # print(len(ho), 'hos')
    plot_df_iou = df.groupby(by=[h, 'phase', 'test data', 's_lvl'], as_index=False).agg(iou=('iou', 'mean')) # IoU by ph, data, key
    plot_df_runtime = df.groupby(by=[h, 'phase', 's_lvl'], as_index=False).agg(Runtime=('Runtime', 'mean')) # time by ph, key
    plot_df = pd.merge(left=plot_df_iou, right=plot_df_runtime, on=[h, 'phase', 's_lvl'])
    return plot_df, ho


def widen_phases(df: pd.DataFrame, x='phase', y='IoU', h='tr_val', f='test set'):
    df = _clean_df_(df, [h])
    df = df.melt(id_vars=[h], var_name='col_name', value_name=y) # 3_test_GDA_iou
    df[[x, f]] = df['col_name'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
    df[f] = df[f].str.split('_', expand=True)[1] # GDA
    df[x] = pd.to_numeric(df[x], downcast='integer')
    df = df.drop(columns=['col_name'])
    return df


def plot_4d(df: pd.DataFrame):
    '''
    IoU, ph/t/hyperparam, test set, key
    '''
    pass


def plot_3d(df: pd.DataFrame):
    '''
    IoU, hyperparam, test set
    '''
    pass


def dist_4d_cont(df: pd.DataFrame, x, y, h, f, r):
    pass


def rels(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line'):
    '''
    all rel plots
    ph123 by cont
    '''
    return sns.relplot(
        data=df, # df.sort_values(by='phase'),
        kind=ch,
        x=x,
        y=y,
        hue=h,
        hue_order=h_ord, # sorted(ho, key=lambda x: str(x).count('s')),
        # style='phase',
        style=s, # h,
        row=r, # "test data",
        row_order=r_ord, # ["DK", "GDA", "SYNT"],
        col=c, # 's_lvl',
        col_order=c_ord,
        markers=True,
        sort=True,
        palette=sns.color_palette(), # TODO
    )


def cats(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, ch='box'):
    '''
    all cat plots
    '''
    return sns.catplot(
        data=df,
        kind=ch,
        x=x,
        y=y,
        hue=h,
        col=c,
        col_order=c_ord,
        row=r,
        row_order=r_ord,
    )


def plot_prod(g, x, y, h, t='', bs=None): # TODO title, labels
    '''
    labels, base lines
    '''
    if bs:
        # g.set(ylim=(0.35, 0.85))
        for ax, b in zip(g.axes.flatten(), bs):
            ax.axhline(b, ls='--')
    # g.set_titles(row_template="{row_name}", col_template="{col_name}")
    # g.set(title=t)
    # g.set(title='IoUs by amount of gda in setups')
    # g.set_titles("{col_name}")
    # g.set_axis_labels(x, y)
    return g


def joint_phases(df: pd.DataFrame): # TODO not by comb_key, new one
    '''
    x = 'phase'
    y = 'IoU'
    h = 'comb_key'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    h = 'comb_key'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    df = widen_phases(df, x, y, h, col)
    g = cats(df, x, y, h, col, col_order, ch='point')
    g = plot_prod(g, x, y, h)
    return g, df


def wall_time(df: pd.DataFrame):
    '''
    x = 'Walltime'
    y = 'IoU'
    h = 'comb_key'
    col = 'test set'
    '''
    x = 'Walltime'
    y = 'IoU'
    h = 'comb_key'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    df = process_runtime(df, h)
    df, h_ord = widen_runtime(df, h)
    g = rels(df, x, y, h, col, col_order, h_ord=h_ord, s=h)
    g = plot_prod(g, x, y, h)
    return g, df


def sngl_ph_tr_val_bar(df: pd.DataFrame, t=''):
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    ch = 'bar'
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    ch='bar'
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, h, col)
    g = cats(df, x, y, h, col, col_order, ch=ch)
    g = plot_prod(g, x, y, h, t=t, bs=bs)
    return g, df


def sngl_ph_tr_val_box(df: pd.DataFrame, t=''):
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    ch = 'box'
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    ch='box'
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, h, col)
    g = cats(df, x, y, h, col, col_order, ch=ch)
    g = plot_prod(g, x, y, h, t=t, bs=bs)
    return g, df
    


def sngl_ph_tr_val_violin(df: pd.DataFrame, t=''):
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    ch = 'violin'
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    ch='violin'
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, h, col)
    g = cats(df, x, y, h, col, col_order, ch=ch)
    g = plot_prod(g, x, y, h, t=t, bs=bs)
    return g, df
    


def run_catch(df: pd.DataFrame, pth_df, pth_plt, fff=wall_time):
    g, df = fff(df)
    if SAVING:
        df.to_csv(SAVEDIR / pth_df)
        g.savefig(SAVEDIR / pth_plt)
        g.close()
    else:
        print(df.columns)
        print(df.head())
        plt.show()


CHARTS_JOINT = {
    # 'wall_time': wall_time, # TODO data not ready yet; C
    # 'joint_workload': wall_time, # TODO rel workload - ious - test sets; D
    'joint_phases': joint_phases, # TODO h=diversity; B''
}

CHARTS_SNGL = {
    'sngl_ph_tr_val_bar': sngl_ph_tr_val_bar, # almost done: tr_val, no s100; A
    'sngl_ph_tr_val_box': sngl_ph_tr_val_box, # almost done: tr_val, no s100; A
    'sngl_ph_tr_val_violin': sngl_ph_tr_val_violin, # almost done: tr_val, no s100; A
    # 'sngl_workload_bar_viol_box': wall_time, # TODO cat workload - ious - test sets; D
}

# CHARTS = CHARTS_JOINT
# CHARTS.update(CHARTS_SNGL)


def main():
    ph123 = pd.read_csv(FILES['joint'])
    ph1 = pd.read_csv(FILES['ph1'])
    # ph2 = pd.read_csv(FILES['ph2'])
    # ph3 = pd.read_csv(FILES['ph3'])
    process_diversity(ph123)
    data_diversity_space()
    # for ch, ff in CHARTS_JOINT.items(): # TODO match FILE with CHARTS
    #     run_catch(ph123, ch+'.csv', ch+'.png', ff) # TODO fname, title
    # for ch, ff in CHARTS_SNGL.items(): # TODO match FILE with CHARTS
    #     run_catch(ph1, ch+'.csv', ch+'.png', ff) # TODO fname, title


if __name__ == '__main__':
    main()
    # data_diversity_space()
