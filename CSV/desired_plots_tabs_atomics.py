'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


import pandas as pd
import seaborn as sns
import seaborn.objects as so
import re
from numpy import mean


SAVING = True
SAVING = False

LMT = False

if SAVING:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
from matplotlib import collections

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK

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


def _clean_df_(df: pd.DataFrame, h=['comb_key']):
    cols = df.columns
    cols = list(set(cols) & set(IOU_COLS+h))
    return df[cols]


def process_runtime(df: pd.DataFrame, h):
    '''
    add cumulative runtime (walltime), widen phases, rename
    '''
    df['1_cum_Runtime'] = df['Runtime_x']
    df['2_cum_Runtime'] = df['Runtime_y'] + df['1_cum_Runtime']
    df['3_cum_Runtime'] = df['Runtime'] + df['2_cum_Runtime']
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


def widen_runtime_agg(df: pd.DataFrame, h):
    df = process_runtime(df, h)
    df = pd.melt(df, id_vars=[h, 'phase', 'Runtime'],
                      value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], 
                      var_name='test data', value_name='iou')
    df['test set'] = df['test data'].str.replace('iou_', '')
    df = df.drop(columns=['test data'])
    ho = df[h].unique()
    ho.sort()
    plot_df_iou = df.groupby(by=[h, 'phase', 'test set'], as_index=False).agg(IoU=('iou', 'mean')) # IoU by ph, set, key
    plot_df_runtime = df.groupby(by=[h, 'phase'], as_index=False).agg(Walltime=('Runtime', 'mean')) # time by ph, key
    plot_df = pd.merge(left=plot_df_iou, right=plot_df_runtime, on=[h, 'phase'])
    return plot_df, ho


def widen_runtime_no_agg(df: pd.DataFrame, h):
    df = process_runtime(df, h)
    df = pd.melt(df, id_vars=[h, 'phase', 'Runtime'],
                      value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], 
                      var_name='test data', value_name='iou')
    df['test set'] = df['test data'].str.replace('iou_', '')
    df = df.drop(columns=['test data'])
    df = df.rename(columns={'Runtime': 'Walltime', 'iou': 'IoU'})
    ho = df[h].unique()
    ho.sort()
    if not SAVING:
        print(df.head(20))
    return df, ho


def process_workload(df: pd.DataFrame, h):
    '''
    add cumulative workload, widen phases, rename
    '''
    df['1_cum_workload'] = df['workload_x']
    df['2_cum_workload'] = df['workload_y'] + df['1_cum_workload']
    df['3_cum_workload'] = df['workload'] + df['2_cum_workload']
    def rename_column(col):
        m = re.match(r"(\d+)_test_([A-Z]+)_iou$", col) # Match: phase_test_SET_iou
        if m:
            phase, set_name = m.groups()
            return f"iou_{set_name}_{phase}"
        m = re.match(r"(\d+)_cum_workload$", col) # Match: phase_cum_workload
        if m:
            phase = m.group(1)
            return f"workload_{phase}"
        return col
    
    df = df.rename(columns=rename_column)
    df = df.drop(columns=['workload'])
    df = df.reset_index(names="row_id")
    return (pd.wide_to_long(df, stubnames=['iou_SYNT', 'iou_GDA', 'iou_DK', 'workload'], 
                            i=['row_id', h], j='phase', sep='_', suffix=r"\d+")).reset_index()


def widen_workload_agg(df: pd.DataFrame, h):
    df = process_workload(df, h)
    df = pd.melt(df, id_vars=[h, 'phase', 'workload'],
                      value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], 
                      var_name='test data', value_name='iou')
    df['test set'] = df['test data'].str.replace('iou_', '')
    df = df.drop(columns=['test data'])
    ho = df[h].unique()
    ho.sort()
    plot_df_iou = df.groupby(by=[h, 'phase', 'test set'], as_index=False).agg(IoU=('iou', 'mean')) # IoU by ph, set, key
    plot_df_runtime = df.groupby(by=[h, 'phase'], as_index=False).agg(Workload=('workload', 'mean')) # time by ph, key
    plot_df = pd.merge(left=plot_df_iou, right=plot_df_runtime, on=[h, 'phase'])
    return plot_df, ho


def widen_workload_no_agg(df: pd.DataFrame, h):
    df = process_workload(df, h)
    df = pd.melt(df, id_vars=[h, 'phase', 'workload'],
                      value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], 
                      var_name='test data', value_name='iou')
    df['test set'] = df['test data'].str.replace('iou_', '')
    df = df.drop(columns=['test data'])
    df = df.rename(columns={'workload': 'Workload', 'iou': 'IoU'})
    ho = df[h].unique()
    ho.sort()
    if not SAVING:
        print(df.head(1))
    return df, ho


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


def round_sngl_ph(df: pd.DataFrame, endecja=0):
    df['dom'] = df['dom'].round(endecja)
    df['real'] = df['real'].round(endecja)
    df['dist'] = df['dist'].round(endecja)
    return df


def limit_to_successful(df: pd.DataFrame, cnc=False, off=not LMT):
    if off or cnc:
        return df
    iou = list(set(df.columns) & set(sorted(IOU_COLS_GDA)))
    print(sorted(iou))
    iou = sorted(iou)[0]
    return df[df[iou] > 0.617]


def widen_phases(df: pd.DataFrame, x='phase', y='IoU', h=['tr_val'], f='test set'):
    df = _clean_df_(df, h)
    df = df.melt(id_vars=h, var_name='col_name', value_name=y) # 3_test_GDA_iou
    df[[x, f]] = df['col_name'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
    df[f] = df[f].str.split('_', expand=True)[1] # GDA
    df[x] = pd.to_numeric(df[x], downcast='integer')
    df = df.drop(columns=['col_name'])
    if not SAVING:
        print(df.columns)
    return df


def widen_phases_id(df: pd.DataFrame, x='phase', y='IoU', h=['ID'], f='test set'):
    return widen_phases(df, x, y, ['ID'], f)

def widen_phases_h(df: pd.DataFrame, x='phase', y='IoU', h=['ID'], f='test set'):
    return widen_phases(df, x, y, h, f)


def widen_phases_h2(df: pd.DataFrame, x='phase', y='IoU', h=['ID'], f='test set'):
    df = _clean_df_(df, h)
    # df = df.dropna(subset=h)
    # df[h[0]] = df[h[0]].fillna('100')
    df = df.fillna({h[0]: '100'}) # TODO nans
    df[h[0]] = df[h[0]].fillna('100')

    
    df = df.melt(id_vars=h, var_name='col_name', value_name=y) # 3_test_GDA_iou
    print(df.head(3))
    df[[x, f]] = df['col_name'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
    df[f] = df[f].str.split('_', expand=True)[1] # GDA
    df[x] = pd.to_numeric(df[x], downcast='integer')
    print(df.head(2))
    print(df.count()-df.size)
    df = df.drop(columns=['col_name'])    
    if not SAVING:
        print(df.columns)
        eee = df.groupby(['phase', f])[h[0]].unique()
        print(eee.head())
    df[h[0]] = pd.Categorical(
        df[h[0]],
        categories=['mix', 'composite', '15', '25', '35', '45', '55', '65', '75', '100'],
        ordered=True,
    )
    return df


def widen_phases_h3(df: pd.DataFrame, x='phase', y='IoU', h=['ID'], f='test set'):
    df = widen_res_all_h(df)
    # df = df.dropna(subset=h)
    # df[h[0]] = df[h[0]].fillna('100')
    df = df.fillna({h[0]: '100'}) # TODO nans
    df[h[0]] = df[h[0]].fillna('100')

    df[[x, f]] = df['phase_set'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
    df[f] = df[f].str.split('_', expand=True)[1] # GDA
    df[x] = pd.to_numeric(df[x], downcast='integer')
    print(df.head(2))
    print(df.count()-df.size)
    # df = df.drop(columns=['col_name'])    
    if not SAVING:
        print(df.columns)
        eee = df.groupby(['phase', f])[h[0]].unique()
        print(eee.head())
    df[h[0]] = pd.Categorical(
        df[h[0]],
        categories=['mix', 'composite', '15', '25', '35', '45', '55', '65', '75', '100'],
        ordered=True,
    )
    return df


def widen_res_all_h(df: pd.DataFrame, measurements=IOU_COLS):
    '''
    best to be used with a single-phase df or a concat, not joint

    :param list measurements: cols with results
    '''
    print(len(df.columns))
    measurements = list(set(df.columns) & set(measurements))
    hs = list(set(df.columns) - set(measurements))
    print(df.head())
    print('tak----------------------------------------------------------------------------------')
    df = pd.melt(df, id_vars=hs, value_vars=measurements, var_name='phase_set', value_name='IoU')
    print(df.columns)
    print(len(df.columns))
    print('na pewno00000000000000000000000000000000000000000')
    print(df.head())
    # value_vars = [c for c in df.columns if re.match(r'.*(_x|_y)$', c)] # - ['phase_set']
    # value_vars.remove('phase_set')
    # value_vars.remove('IoU')
    return df


# def the_major_col_cleaning(df: pd.DataFrame):
#     cols = df.columns

#     # group 1: suffix-based (_x, _y, or none)
#     suffix_cols = [c for c in cols if c.endswith('_x') or c.endswith('_y') or not any(c.endswith(s) for s in ['_x','_y'])]

#     # group 2: prefix-based (1_, 2_, 3_)
#     prefix_cols = [c for c in cols if c.startswith(('1_', '2_', '3_'))]

#     # group 3: untouched columns
#     id_cols = ['id', 'date', 'category']   # example; adjust to your real identifiers

#     clean_suffix = {}
#     for c in suffix_cols:
#         if c.endswith('_x') or c.endswith('_y'):
#             clean_suffix[c] = c
#         else:
#             clean_suffix[c] = f"{c}_base"


#     clean_prefix = {}
#     for c in prefix_cols:
#         num, name = c.split('_', 1)
#         clean_prefix[c] = f"{name}_{num}"


#     rename_map = {**clean_suffix, **clean_prefix}
#     return df.rename(columns=rename_map)


def the_major_widening(df: pd.DataFrame):
    # df = the_major_col_cleaning(df)
    # df = pd.wide_to_long(df, stubnames=['var1'], i='id', j='suffix', sep='_')
    # return pd.wide_to_long(df, stubnames=['varA'], i='id', j='num', sep='_')
    rename_map = {}

    for col in df.columns:
        # untouched columns
        if col in ['id', 'date', 'category']:
            continue

        # suffix-based: _x, _y, blank
        if col.endswith('_x'):
            base = col[:-2]
            num = 1
        elif col.endswith('_y'):
            base = col[:-2]
            num = 2
        elif re.match(r'.*_[0-9]+$', col):
            # already numeric suffix
            base, num = col.rsplit('_', 1)
            num = int(num)
        # elif col.endswith('_base'):
        #     base = col[:-5]
        #     num = 3
        elif col.endswith('_'):
            # avoid accidental trailing underscores
            base = col.rstrip('_')
            num = 3
        else:
            # blank suffix → 3
            base = col
            num = 3

        # prefix-based: 1_, 2_, 3_
        m = re.match(r'([1-3])_(.+)', col)
        if m:
            num = int(m.group(1))
            base = m.group(2)

        rename_map[col] = f"{base}_{num}"

    df = df.rename(columns=rename_map)

    df = pd.wide_to_long(
        df,
        stubnames=['var1', 'varA'],   # all base names
        i=['id'],                     # identifiers
        j='phase',                      # extracted number
        sep='_'
    ).reset_index() # test_SYNT_iou; 2

    ious = ['test_SYNT_iou', 'test_GDA_iou', 'test_DK_iou'] # list(set(df.columns) & set(IOU_COLS))
    hs = list(set(df.columns) - set(ious))
    df = pd.melt(df, id_vars=hs, value_vars=ious, var_name='phase_set', value_name='IoU')
    df['test set'] = df['phase_set'].str.split('_', n=1, expand=True)[0]
    df = df.drop(columns=['phase_set'])
    return df


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


def cats(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None):
    '''
    all cat plots
    '''
    return sns.catplot(
        data=df,
        kind=ch,
        x=x,
        y=y,
        hue=h,
        hue_order=h_ord,
        col=c,
        col_order=c_ord,
        row=r,
        row_order=r_ord,
        palette=sns.color_palette(),
    )


def cats_endlabs(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None):
    g = cats(df, x, y, h, c, c_ord, r, r_ord, h_ord, s, ch, size)
    # for x in range(len(h_ord)):
    #     g.text(x, -1, va='bottom')
    #     g.text(x, 1, va='top')
    # for ax in g.axes.flat:
    #     violins = [c for c in ax.collections if isinstance(c, matplotlib.collections.PolyCollection)]
    #     for v in violins:
    #         verts = v.get_paths()[0].vertices
    #         y_vals = verts[:, 1]
    #         ymin, ymax = y_vals.min(), y_vals.max()
    #         x_centre = 1
    #         ax.text(x_centre, ymin, va=bottom)
    #         ax.text(x_centre, ymax, va=top)
    # for ax in g.axes.flat:
    #     violins = [c for c in ax.collections if isinstance(c, collections.PolyCollection)]
        
    #     for v in violins:
    #         verts = v.get_paths()[0].vertices
    #         y_vals = verts[:, 1]
    #         ymin, ymax = y_vals.min(), y_vals.max()
            
    #         # x-position of the violin center
    #         x_center = mean(verts[:, 0])
            
    #         ax.text(x_center, ymin, f"{ymin:.1f}", ha="center", va="top")
    #         ax.text(x_center, ymax, f"{ymax:.1f}", ha="center", va="bottom")
    return g



def ucats(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None): # TODO more line plts?
    df = df[df['test set'] == 'GDA']
    return sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=h,
        hue_order=h_ord,
        units=h,
        estimator=None,
        palette=sns.color_palette(),
    )
    return sns.catplot(
        data=df,
        kind=ch,
        x=x,
        y=y,
        hue=h,
        hue_order=h_ord,
        units=h,
        col=c,
        col_order=c_ord,
        row=r,
        row_order=r_ord,
        palette=sns.color_palette(),
    )


# def lineplt_ci(df: pd.DataFrame, x, y, h, c='test set', c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None):
#     summary = (
#         df.groupby([c, "phase"])
#         .agg(
#             mean_x=(x, "mean"),
#             mean_y=(y, "mean"),
#             y_low=(y, lambda v: percentile(v, 2.5)),
#             y_high=(y, lambda v: percentile(v, 97.5))
#         )
#         .reset_index()
#     )
#     df = df.sort_values([c, h, "phase"])
#     summary = summary.sort_values([c, "phase"])

#     # Create facet grid
#     g = sns.FacetGrid(df, col=c, hue=h)

#     def draw_trajectories(data, color, **kwargs):
#         # Draw individual trajectories
#         plt.plot(
#             data[x], data[y],
#             marker="o",
#             # markersize=4,
#             # linewidth=1.2,
#             color=color,
#             alpha=0.7
#         )

#     # Draw individual trajectories
#     g.map_dataframe(draw_trajectories)

#     # Draw confidence ribbon + mean trajectory inside each facet
#     for ax, (col_val, sub) in zip(g.axes.flat, summary.groupby(c)):
#         ax.fill_between(
#             sub["mean_x"], sub["y_low"], sub["y_high"],
#             color="black", alpha=0.15
#         )
#         ax.plot(
#             sub["mean_x"], sub["mean_y"],
#             color="black", linewidth=2.5, label="Mean trajectory"
#         )

#     return g


def plot_prod(g, x, y, h, t='', bs=None, xl=''): # TODO title, labels
    '''
    labels, base lines
    '''
    if bs:
        # g.set(ylim=(0.35, 0.85))
        for ax, b in zip(g.axes.flatten(), bs):
            ax.axhline(b, ls='--')
    if xl != '':
        g.set_xlabels(x+' '+xl)
        for ax in g.axes.flatten():
            ax.set_xscale('log')
    if t != '':
        g.set(title=t)
    # g.set_titles(row_template="{row_name}", col_template="{col_name}")
    # g.set(title=t)
    # g.set(title='IoUs by amount of gda in setups')
    # g.set_titles("{col_name}")
    # g.set_axis_labels(x, y)
    for ax in g.axes.flat:
        violins = [c for c in ax.collections if isinstance(c, collections.PolyCollection)]
        
        for v in violins:
            verts = v.get_paths()[0].vertices
            y_vals = verts[:, 1]
            ymin, ymax = y_vals.min(), y_vals.max()
            
            # x-position of the violin center
            x_center = mean(verts[:, 0])
            
            ax.text(x_center, ymin, f"{ymin:.3f}", ha="center", va="top")
            ax.text(x_center, ymax, f"{ymax:.3f}", ha="center", va="bottom")
    return g
