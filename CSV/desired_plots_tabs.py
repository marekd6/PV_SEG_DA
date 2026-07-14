'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


import pandas as pd
import seaborn as sns
import re


SAVING = True
# SAVING = False

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

DIR = 'CSV/joint_ph_charts/modf_rnt_wrk_div'
# DIR = 'CSV/joint_ph_charts/modf_rnt_wrk_div_tr'
SAVEDIR = 'CSV/joint_ph_charts/selected/tr_val_derivs'
# SAVEDIR = 'CSV/joint_ph_charts/selected/tr_only_derivs'

FILES = {
    'joint': f'{DIR}/ph123b.csv',
    'ph1': f'{DIR}/proc_ph1.csv',
    'ph2': f'{DIR}/proc_ph2.csv',
    'ph3': f'{DIR}/proc_ph3.csv',
    'concat': f'{DIR}/cnc123b.csv',
}

# DS: workload (Ssize*factor) =====================
# sub = 0.1 * |S| * sub
# dk = 1 * |dk|
# m = dk + sub
# DS: diversity (form/content - realism/domain) Sszie
# real(s) == real(sub)
# s = 0
# dk = 1
# sub = 0
# m = [real(sub)*|s|*sub + real(dk)*|dk|] / (|s|*sub + |dk|)
#
# DS: workload, diversity - train, val; per file
# DS: workload, diversity - agg as D = Sd or D = d(Sreal, Sdom); W = Sw
# DS: count uniqe - comb_key only


RNTMS = ['1_cum_Runtime', '2_cum_Runtime', '3_cum_Runtime']


def _clean_df_(df: pd.DataFrame, h=['comb_key']):
    cols = df.columns
    # cols = list(set(cols) & set(IOU_COLS)) + h
    cols = list(set(cols) & set(IOU_COLS+h)) # TODO ?
    return df[cols]


def process_runtime(df: pd.DataFrame, h):
    '''
    cols
    '''
    def rename_column(col):
        m = re.match(r"(\d+)_test_([A-Z]+)_iou$", col) # Match: phase_test_SET_iou
        if m:
            phase, set_name = m.groups()
            return f"iou_{set_name}_{phase}"
        m = re.match(r"(\d+)_cum_Runtime$", col) # Match: phase_cum_Runtime
        if m:
            phase = m.group(1)
            return f"Runtime_{phase}"
        return col
    
    df = df.rename(columns=rename_column)
    df = df.drop(columns=['Runtime'])
    df = df.reset_index(names="row_id")
    return (pd.wide_to_long(df, stubnames=['iou_SYNT', 'iou_GDA', 'iou_DK', 'Runtime'], 
                            i=['row_id', h], j='phase', sep='_', suffix=r"\d+")).reset_index()


def process_diversity_workload(df: pd.DataFrame):
    '''
    directly with 3-phase runs
    '''
    df['Sworkload'] = df['workload_x'] + df['workload_y'] + df['workload'] # sum

    df['cnt_ds'] = df['comb_key'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) # number of DSs

    df['s_lvl'] = [x.count('s') for x in df['comb_key']] # how many SYNTs: s, sub, subm
    df['s_lvl'] += [x.count('_m') for x in df['comb_key']] # plus how many MIXs: _m
    df['dk_lvl'] = [x.count('dk') for x in df['comb_key']] # how many DKs: dk
    df['dk_lvl'] += [x.count('m') for x in df['comb_key']] # plus how many MIXs: m
    df['gda_lvl'] = [x.count('gda') for x in df['comb_key']] # how many GDAs
    # df['s_lvl'] = [x.count('s') for x in df['trains']] # how many SYNTs in train only
    # df['gda_lvl'] = [x.count('gda') for x in df['trains']] # how many GDAs in train only

    df['DS_scores_sum'] = df['dist_x'] + df['dist_y'] + df['dist'] # sum of 1, 2, 3 scores
    # df['DS_scores_sum'] = df['DS_scores_sum'] / 3

    df['Sdom_raw'] = df['dom_x'] + df['dom_y'] + df['dom'] # sum of dim
    df['Sreal_raw'] = df['real_x'] + df['real_y'] + df['real'] # sum of dim    
    # df['Sdom_raw'] = df['Sdom_raw'] / 3
    # df['Sreal_raw'] = df['Sreal_raw'] / 3
    df['DS_score_tot_raw'] = ((df['Sdom_raw'])**2 + (df['Sreal_raw'])**2)**0.5 # score by summed dims
    df['DS_score_tot'] = df['DS_score_tot_raw'].round()
    df['Sdom'] = df['Sdom_raw'].round()
    df['Sreal'] = df['Sreal_raw'].round()

    df['DS_score_raw'] = ( # score by 1, 2, 3 dims
        (df['dom_x'])**2 + (df['dom_y'])**2 + (df['dom'])**2 +
        (df['real_x'])**2 + (df['real_y'])**2 + (df['real'])**2
        )**0.5
    df['DS_score'] = df['DS_score_raw'].round()
    
    df['ddiff'] = (df['DS_scores_sum'] - df['DS_score_tot_raw']) / df['DS_score_tot_raw'] * 100
    # print(df[['DS_score_tot', 'DS_scores_sum', 'ddiff', 'cnt_ds', 'Sworkload']].head())

    if not SAVING and False:
        sns.scatterplot(data=df, x='Sdom', y='Sreal', hue='DS_score_tot', size='DS_score_tot')
        # sns.scatterplot(data=df, x='Sdom', y='Sreal')
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.grid(visible=True, which='major')
        plt.show()
        
        # sns.scatterplot(data=df, x='Sdom_raw', y='Sreal_raw', hue='DS_score_tot', size='DS_score_tot_raw')
        sns.scatterplot(data=df, x='Sdom_raw', y='Sreal_raw', hue='DS_score', size='DS_score_raw')
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.grid(visible=True, which='major')
        plt.show()

        # sns.scatterplot(data=df, x='DS_score_tot', y='DS_scores_sum')
        # sns.scatterplot(data=df, x='DS_score_tot', y='DS_score')
        sns.scatterplot(data=df, x='DS_score_tot_raw', y='DS_score_raw')
        plt.show()

        # sns.catplot(data=df, x='cnt_ds', y='DS_score_tot')
        # plt.show()

        # sns.scatterplot(data=df, x='DS_score_tot', y='Sworkload', hue='DS_score_tot', size='DS_score_tot')
        # plt.grid(visible=True, which='major')
        # plt.show()

    return df


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


def widen_cont_diversity(df: pd.DataFrame, h=['s_lvl']):
    df = pd.melt(df, id_vars=h+['phase'],
                      value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], 
                      var_name='test data', value_name='iou')
    df['test data'] = df['test data'].str.replace('iou_', '')
    # plot_df_iou = df.groupby(by=h+['phase', 'test data'], as_index=False).agg(iou=('iou', 'mean')) # IoU by ph, data, key
    # plot_df_runtime = df.groupby(by=[h, 'phase', 's_lvl'], as_index=False).agg(Runtime=('Runtime', 'mean')) # time by ph, key
    # plot_df = pd.merge(left=plot_df_iou, right=plot_df_runtime, on=[h, 'phase', 's_lvl'])
    return df


# def widen_phases(df: pd.DataFrame, x='phase', y='IoU', h='tr_val', f='test set'):
#     df = _clean_df_(df, [h])
#     df = df.melt(id_vars=[h], var_name='col_name', value_name=y) # 3_test_GDA_iou
#     df[[x, f]] = df['col_name'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
#     df[f] = df[f].str.split('_', expand=True)[1] # GDA
#     df[x] = pd.to_numeric(df[x], downcast='integer')
#     df = df.drop(columns=['col_name'])
#     return df


def widen_phases(df: pd.DataFrame, x='phase', y='IoU', h=['tr_val'], f='test set'):
    df = _clean_df_(df, h)
    df = df.melt(id_vars=h, var_name='col_name', value_name=y) # 3_test_GDA_iou
    df[[x, f]] = df['col_name'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
    df[f] = df[f].str.split('_', expand=True)[1] # GDA
    df[x] = pd.to_numeric(df[x], downcast='integer')
    df = df.drop(columns=['col_name'])
    print(df.columns)
    return df


def dist_4d_cont(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='scatter'):
    pass


def rels(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None):
    '''
    all rel plots
    ph123 by cont
    '''
    # print(df.head())
    return sns.relplot(
        data=df, # df.sort_values(by='phase'),
        kind=ch,
        x=x,
        y=y,
        hue=h,
        hue_order=h_ord, # sorted(ho, key=lambda x: str(x).count('s')),
        # style='phase',
        style=s, # h,
        size=size,
        row=r, # "test data",
        row_order=r_ord, # ["DK", "GDA", "SYNT"],
        col=c, # 's_lvl',
        col_order=c_ord,
        markers=True,
        # palette=sns.color_palette(), # TODO
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
        palette=sns.color_palette(),
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


def joint_phases(df: pd.DataFrame, h):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    df = widen_phases(df, x, y, [h], col)
    return df


def joint_phases_pt_line(df: pd.DataFrame, h):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    df = joint_phases(df, h)
    g = cats(df, x, y, h, col, col_order, ch='point')
    g = plot_prod(g, x, y, h)
    return g, df


def joint_phases_bar(df: pd.DataFrame, h):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    bs = [0.71, 0.617, 0.359]
    df = joint_phases(df, h)
    g = cats(df, x, y, h, col, col_order, ch='bar')
    g = plot_prod(g, x, y, h, bs=bs)
    return g, df


def joint_phases_box(df: pd.DataFrame, h):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    bs = [0.71, 0.617, 0.359]
    df = joint_phases(df, h)
    g = cats(df, x, y, h, col, col_order, ch='box')
    g = plot_prod(g, x, y, h, bs=bs)
    return g, df


def joint_phases_violin(df: pd.DataFrame, h):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    bs = [0.71, 0.617, 0.359]
    df = joint_phases(df, h)
    g = cats(df, x, y, h, col, col_order, ch='violin')
    g = plot_prod(g, x, y, h, bs=bs)
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


def derived_keys_scatter(df: pd.DataFrame, x='DS_score_raw', h=None): # h='DS_score'
    '''
    y = 'IoU'
    col = 'test set'
    row = 'phase'
    '''
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    row = 'phase'
    df = widen_phases(df, x=row, y=y, h=[x], f=col) # [x, h]
    g = rels(df, x, y, h, col, col_order, row, [1, 2, 3], ch='scatter', size=h)
    g = plot_prod(g, x, y, h)
    return g, df


def derived_keys_violin(df: pd.DataFrame, x='DS_score_raw', h=None): # h='DS_score'
    '''
    y = 'IoU'
    col = 'test set'
    row = 'phase'
    '''
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    row = 'phase'
    df = widen_phases(df, x=row, y=y, h=[x], f=col) # [x, h]
    g = cats(df, x, y, h, col, col_order, row, [1, 2, 3], ch='violin')
    g = plot_prod(g, x, y, h)
    return g, df


def derived_keys_box(df: pd.DataFrame, x='DS_score_raw', h=None): # h='DS_score'
    '''
    y = 'IoU'
    col = 'test set'
    row = 'phase'
    '''
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    row = 'phase'
    df = widen_phases(df, x=row, y=y, h=[x], f=col) # [x, h]
    g = cats(df, x, y, h, col, col_order, row, [1, 2, 3], ch='box')
    g = plot_prod(g, x, y, h)
    return g, df


def derived_keys_bar(df: pd.DataFrame, x='DS_score_raw', h=None): # h='DS_score'
    '''
    y = 'IoU'
    col = 'test set'
    row = 'phase'
    '''
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    row = 'phase'
    df = widen_phases(df, x=row, y=y, h=[x], f=col) # [x, h]
    g = cats(df, x, y, h, col, col_order, row, [1, 2, 3], ch='bar')
    g = plot_prod(g, x, y, h)
    return g, df


# scatters/rels: r, c, y, x
# cats: r, c, y, x==h (h agg)
def workloads_4d_violin(df: pd.DataFrame, x='DS_score_raw', h=None): # h='DS_score' # TODO scatter, viol.
    '''
    y = 'IoU'
    col = 'test set'
    row = 'phase'
    '''
    pass
    # y = 'IoU'
    # col = 'test set'
    # col_order = ["DK", "GDA", "SYNT"]
    # row = 'phase'
    # df = widen_phases(df, x=row, y=y, h=[x], f=col) # [x, h]
    # g = cats(df, x, y, h, col, col_order, row, [1, 2, 3], ch='violin')
    # g = plot_prod(g, x, y, h)
    # return g, df


def sngl_ph_tr_val_bar(df: pd.DataFrame, h='tr_val', t=''):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    ch = 'bar'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    ch='bar'
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, [h], col)
    g = cats(df, x, y, h, col, col_order, ch=ch)
    g = plot_prod(g, x, y, h, t=t, bs=bs)
    return g, df


def sngl_ph_tr_val_box(df: pd.DataFrame, h='tr_val', t=''):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    ch = 'box'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    ch='box'
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, [h], col)
    g = cats(df, x, y, h, col, col_order, ch=ch)
    g = plot_prod(g, x, y, h, t=t, bs=bs)
    return g, df


def sngl_ph_tr_val_violin(df: pd.DataFrame, h='tr_val', t=''):
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    ch = 'violin'
    '''
    x = 'phase'
    y = 'IoU'
    col = 'test set'
    col_order = ["DK", "GDA", "SYNT"]
    ch='violin'
    bs = [0.71, 0.617, 0.359]
    df = widen_phases(df, x, y, [h], col)
    g = cats(df, x, y, h, col, col_order, ch=ch)
    g = plot_prod(g, x, y, h, t=t, bs=bs)
    return g, df
    

def run_catch(df: pd.DataFrame, pth_df, pth_plt, fff=wall_time):
    catt = ['cnt_ds', 's_lvl', 'gda_lvl'] # , 'comb_key']
    rell = ['DS_score_tot_raw', 'DS_score_raw', 'Sdom_raw', 'Sreal_raw']
    df_org = df.copy()
    if fff in CHARTS_JOINT.values():
        ks = rell if fff == derived_keys_scatter else catt
        for h in ks:
            print(h)
            g, df = fff(df_org, h)
            if SAVING:
                ppp = pth_plt.replace('.png', f'_{h}.png')
                ppdf = pth_df.replace('.csv', f'_{h}.csv')
                df.to_csv(f'{SAVEDIR}/{ppdf}')
                g.savefig(f'{SAVEDIR}/{ppp}')
            else:
                print(df.columns)
                print(df.head())
                plt.show()
            print()
    else: # SNGL
        for h in ['tr_val', 'dist', 'real', 'dom']: # , 'comb_key']:
            print(h)
            g, df = fff(df_org, h)
            if SAVING:
                ppp = pth_plt.replace('.png', f'_{h}.png')
                ppdf = pth_df.replace('.csv', f'_{h}.csv')
                df.to_csv(f'{SAVEDIR}/{ppdf}')
                g.savefig(f'{SAVEDIR}/{ppp}')
            else:
                print(df.columns)
                print(df.head())
                plt.show()
            print()


CHARTS_JOINT = {
    # 'wall_time': wall_time, # TODO data not ready yet; C
    # 'joint_workload': wall_time, # TODO rel workload - ious - test sets; D
    'joint_phases_pt_line': joint_phases_pt_line,
    'joint_phases_bar': joint_phases_bar,
    'joint_phases_box': joint_phases_box,
    'joint_phases_violin': joint_phases_violin,
    'derived_keys_scatter': derived_keys_scatter,
    'derived_keys_bar': derived_keys_bar,
    'derived_keys_box': derived_keys_box,
    'derived_keys_violin': derived_keys_violin,
}

# CHARTS_JOINT = { # these should work for ph3 from joint - final effect TODO where phase == 3 after proc; r=None
#     'derived_keys': derived_keys,
#     'derived_keys_bar': derived_keys_bar,
#     'derived_keys_box': derived_keys_box,
#     'derived_keys_violin': derived_keys_violin,
# }

CHARTS_SNGL = { # 1st phase; TODO rel with dist_raw and with workload
    'sngl_ph_tr_val_bar': sngl_ph_tr_val_bar, # almost done: tr_val, no s100; A
    'sngl_ph_tr_val_box': sngl_ph_tr_val_box, # almost done: tr_val, no s100; A
    'sngl_ph_tr_val_violin': sngl_ph_tr_val_violin, # almost done: tr_val, no s100; A
    # 'sngl_workload_bar_viol_box': wall_time, # TODO cat workload - ious - test sets; D
}


def main():
    ph123 = pd.read_csv(FILES['joint'])
    ph1 = pd.read_csv(FILES['ph1'])
    # ph2 = pd.read_csv(FILES['ph2'])
    # ph3 = pd.read_csv(FILES['ph3'])

    ph123 = process_diversity_workload(ph123)
    # x = 'DS_score_raw'
    # y = 'IoU'
    # h = 'DS_score'
    # col = 'test set'
    # df = widen_phases(ph123, h=[h, x])
    # df[x] = df[x].astype(float)
    # g = rels(df, x, y, h, col, c_ord=["DK", "GDA", "SYNT"], r='phase', r_ord=[1, 2, 3], ch='line', s=h)
    # # g = rels(df, x, y, h, col, ["DK", "GDA", "SYNT"], ch='scatter')
    # plt.show()

    for ch, ff in CHARTS_JOINT.items():
        run_catch(ph123, ch+'.csv', ch+'.png', ff)
    for ch, ff in CHARTS_SNGL.items():
        run_catch(ph1, ch+'.csv', ch+'.png', ff)


if __name__ == '__main__':
    main()
