'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


import pandas as pd
import seaborn as sns
import re

SAVING = True

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


DS_SIZES_TR = {
    's': 123,
    'dk': 789,
    'gda': 18,
}

DS_SIZES_VAL = {
    's': 123,
    'dk': 789,
    'gda': 20,
}

S_SUB_SIZES = {
    'mix': 456,
    'composite': 546
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


def process_diversity(df: pd.DataFrame):
    pass


def widen_runtime(df: pd.DataFrame, h):
    df = pd.melt(df, id_vars=[h, 's_lvl', 'phase', 'Runtime'], 
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


# def bars(df: pd.DataFrame, x, y, h, f):
#     '''
#     sngl ph
#     '''
#     pass


# def lines_cat(df: pd.DataFrame, x, y, h, f):
#     '''
#     ph123 by ph
#     '''
#     pass


def dist_4d_cont(df: pd.DataFrame, x, y, h, f, r):
    pass


# def dist_box_viol(df: pd.DataFrame, x, y, h, f, ch='box'):
#     return sns.catplot(
#         data=df,
#         x=x,
#         y=y,
#         hue=h,
#         col=f,
#         kind=ch,
#         # col_wrap=col_wrap,
#     )


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
        markers=True,
        sort=True,
        palette=sns.color_palette(), # 
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


def plot_prod(g, x, y, h, bs=None):
    '''
    labels, base lines
    '''
    if bs:
        g.set(ylim=(0.35, 0.85))
        for ax, b in zip(g.axes.flatten(), bs):
            ax.axhline(b, ls='--')
    return g


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


def sngl_ph_tr_val(df: pd.DataFrame):
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    h = 'tr_val'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, h, col)
    g = cats(df, x, y, h, col, col_order, None, None, 'bar')
    g = plot_prod(g, x, y, h, bs)
    return g, df
    


def run_catch(df: pd.DataFrame, pth_df, pth_plt, fff=wall_time):
    g, df = fff(df)
    df.to_csv(pth_df)
    if SAVING:
        g.savefig(pth_plt)
        g.close()
    else:
        g.show()


CHARTS_JOINT = {
    'wall_time': wall_time,
    'sngl_ph_tr_val': sngl_ph_tr_val,
}


def main():
    ph123 = pd.read_csv(FILES['joint'])
    # ph1 = pd.read_csv(FILES['ph1'])
    # ph2 = pd.read_csv(FILES['ph2'])
    # ph3 = pd.read_csv(FILES['ph3'])

    for ch, ff in CHARTS_JOINT.items():
        run_catch(ph123, ch, ch, ff)


if __name__ == '__main__':
    main()
