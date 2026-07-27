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
from matplotlib import collections, axes

IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
RAW_IOU_COLS = ['test_SYNT_iou', 'test_GDA_iou', 'test_DK_iou']


HPARAM_COLS_BASE = ['wd', 'epochs',  'ID', 'epochs_done', 'loss', 'val', 'lrdec', 'ema', 'epoch', 
                    'src', 'batch_size', 'Sweep', 'warmup_epochs', 'train', 'sub',  
                    'tr_val', 'Runtime', 'fn', 'lrenc']
CALC_COLS_BASE = ['workload', 're_t', 'do_t', 're_v', 'real', 'w_v', 'w_t', 'dom', 'do_v', 'sub_mult', 'effective workload', 'dist']

SNGL_COLS_BASES = CALC_COLS_BASE + HPARAM_COLS_BASE + RAW_IOU_COLS
JOINT_COLS_BASES = SNGL_COLS_BASES + ['Walltime', 'Workload', 'cumul. SYNT use', 'cumul. DK use', 'cumul. GDA use', 'cumul. no. unique DS'] # the cums

SNGL_FIXED_COLS = ['entry_id']
GLOB_FIXED_COLS = ['comb_key', 'trains', 'Sworkload', 'total no. unique DS', 'total SYNT use', 'total DK use', 'total GDA use', 
             'DS_scores_sum', 'Sdom_raw', 'Sreal_raw', 'DS_score_tot_raw', 'DS_score_tot',
             'Sdom', 'Sreal', 'DS_score_raw', 'DS_score', 'ddiff'] + SNGL_FIXED_COLS


def total_df_treatment(pth: str, limit=False, round=False, joint=False, cnc=False, off=not LMT, endecja=0):
    '''
    many operations, mainly df widening
    '''
    df = pd.read_csv(pth)
    df = df.rename(columns={'Unnamed: 0': 'entry_id'})
    st = SNGL_COLS_BASES
    if limit:
        df = limit_to_successful(df, cnc, off)
    if joint:
        df = process_comb_cum_calcs(df)
        st = JOINT_COLS_BASES
    if round:
        df = round_sngl_ph(df, endecja)
    df = the_major_widening(df, st)
    df = df.fillna({'ema': False, 'sub': '100'}) # TODO map composite, mix to numerics 1/12 and cast to numerics, plot
    df = make_categorical(df, ['phase', 'ema', 'sub', 'comb_key', 'trains', 'tr_val', 'cumul. no. unique DS',
                               'cumul. SYNT use', 'cumul. DK use', 'cumul. GDA use', 'loss', 'val', 'fn',
                               'total no. unique DS', 'total SYNT use', 'total DK use', 'total GDA use', 'train', 'src', 'test set'])

    print(df.columns)
    print(df.head())
    return df


def make_categorical(df: pd.DataFrame, vars=['phase']):
    for x in vars:
        if x in df.columns:
            # dt = int if x in ['phase', 'SYNT use', 'DK use', 'GDA use', 'cnt_ds', 's_lvl', 'dk_lvl', 'gda_lvl',] else pd.CategoricalDtype()
            df[x] = pd.Categorical(df[x])
    return df


def process_comb_cum_calcs(df: pd.DataFrame):
    '''
    operates on the joint df

    add walltime, workload, levels, cnts
    '''
    df['1_Walltime'] = df['Runtime_x']
    df['2_Walltime'] = df['Runtime_y'] + df['1_Walltime']
    df['3_Walltime'] = df['Runtime'] + df['2_Walltime']

    df['1_Workload'] = df['workload_x']
    df['2_Workload'] = df['workload_y'] + df['1_Workload']
    df['3_Workload'] = df['workload'] + df['2_Workload']

    df['1_cumul. SYNT use'] = [x.count('s') for x in df['tr_val_x']] # how many SYNTs: s, sub, subm
    df['1_cumul. SYNT use'] += [x.count('_m') for x in df['tr_val_x']] # plus how many MIXs: _m
    df['1_cumul. DK use'] = [x.count('dk') for x in df['tr_val_x']] # how many DKs: dk
    df['1_cumul. DK use'] += [x.count('m') for x in df['tr_val_x']] # plus how many MIXs: m
    df['1_cumul. GDA use'] = [x.count('gda') for x in df['tr_val_x']] # how many GDAs

    df['2_cumul. SYNT use'] = [x.count('s') for x in df['tr_val_y']] # how many SYNTs: s, sub, subm
    df['2_cumul. SYNT use'] += [x.count('_m') for x in df['tr_val_y']] # plus how many MIXs: _m
    df['2_cumul. SYNT use'] += df['1_cumul. SYNT use']
    df['2_cumul. DK use'] = [x.count('dk') for x in df['tr_val_y']] # how many DKs: dk
    df['2_cumul. DK use'] += [x.count('m') for x in df['tr_val_y']] # plus how many MIXs: m
    df['2_cumul. DK use'] += df['1_cumul. DK use']
    df['2_cumul. GDA use'] = [x.count('gda') for x in df['tr_val_y']] # how many GDAs
    df['2_cumul. GDA use'] += df['1_cumul. GDA use']

    df['3_cumul. SYNT use'] = [x.count('s') for x in df['tr_val']] # how many SYNTs: s, sub, subm
    df['3_cumul. SYNT use'] += [x.count('_m') for x in df['tr_val']] # plus how many MIXs: _m
    df['3_cumul. SYNT use'] += df['2_cumul. SYNT use']
    df['3_cumul. DK use'] = [x.count('dk') for x in df['tr_val']] # how many DKs: dk
    df['3_cumul. DK use'] += [x.count('m') for x in df['tr_val']] # plus how many MIXs: m
    df['3_cumul. DK use'] += df['2_cumul. DK use']
    df['3_cumul. GDA use'] = [x.count('gda') for x in df['tr_val']] # how many GDAs
    df['3_cumul. GDA use'] += df['2_cumul. GDA use']

    df['Sworkload'] = df['3_Workload'] # sum

    df['total no. unique DS'] = df['comb_key'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) # number of DSs
    df['1_cumul. no. unique DS'] = df['tr_val_x'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst))))
    df['2_cumul. no. unique DS'] = df['tr_val_y'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) + df['1_cumul. no. unique DS']
    df['3_cumul. no. unique DS'] = df['tr_val'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) + df['2_cumul. no. unique DS']

    df['total SYNT use'] = [x.count('s') for x in df['comb_key']] # how many SYNTs: s, sub, subm
    df['total SYNT use'] += [x.count('_m') for x in df['comb_key']] # plus how many MIXs: _m
    df['total DK use'] = [x.count('dk') for x in df['comb_key']] # how many DKs: dk
    df['total DK use'] += [x.count('m') for x in df['comb_key']] # plus how many MIXs: m
    df['total GDA use'] = [x.count('gda') for x in df['comb_key']] # how many GDAs
    # df['total SYNT use'] = [x.count('s') for x in df['trains']] # how many SYNTs in train only
    # df['total GDA use'] = [x.count('gda') for x in df['trains']] # how many GDAs in train only

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


def the_major_widening(df: pd.DataFrame, stubs=SNGL_COLS_BASES):
    '''
    unify naming convention of cols, widen params, widen IoUs
    '''
    rename_map = {}
    for col in df.columns:
        # untouched columns
        if col in GLOB_FIXED_COLS:
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
    # pd.set_option('display.max_columns', None)
    # print(df.head(1).T.to_string())
    # print(stubs)

    df = pd.wide_to_long(
        df,
        stubnames=stubs,
        i=SNGL_FIXED_COLS,
        j='phase',
        sep='_'
    ).reset_index() # test_SYNT_iou; 2
    # print(df.head(1).T.to_string())

    hs = list(set(df.columns) - set(RAW_IOU_COLS))
    df = pd.melt(df, id_vars=hs, value_vars=RAW_IOU_COLS, var_name='phase_set', value_name='IoU')
    df['test set'] = df['phase_set'].str.split('_', n=2, expand=True)[1]
    df = df.drop(columns=['phase_set'])
    print(df.shape)
    return df


def rels(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None):
    '''
    line/scatter
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
        size=size,
        row=r, # "test data",
        row_order=r_ord, # ["DK", "GDA", "SYNT"],
        col=c, # 's_lvl',
        col_order=c_ord,
        markers=True,
        # palette=sns.color_palette(), # TODO
    )


def cats(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='bar', size=None):
    '''
    box/viol/bar/point/count/boxen/strip/swarm
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


def line(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None):
    '''
    rel: line
    '''
    g = sns.FacetGrid(
        data=df,
        row=r,
        row_order=r_ord,
        col=c,
        col_order=c_ord,
        hue=h,
    )
    g.map_dataframe(sns.lineplot, x=x, y=y)
    if h:
        g.add_legend()
    g.set_axis_labels(x, y)
    return g
    # return sns.lineplot(
    #     data=df,
    #     x=x,
    #     y=y,
    #     hue=h,
    #     hue_order=h_ord,
    #     # units=h,
    #     # estimator=None,
    #     palette=sns.color_palette(),
    # )


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


def plot_prod(g, x, y, h, t='', bs=None, xl='', add_viol_labs=False): # TODO title, labels
    '''
    labels, base lines
    '''
    if bs:
        # g.set(ylim=(0.35, 0.85))
        for ax, b in zip(g.axes.flatten(), bs):
            ax.axhline(b, ls='--')
    if xl != '':
        if isinstance(g, axes.Axes):
            g.set_xlabel(x+' '+xl)
            g.set_xscale('log')
        else:
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

    if add_viol_labs:
        for ax in g.axes.flat: # TODO plot vals are 'fake' - extract from df, pass as param
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
