import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


SAVING = True
DIR = 'CSV/joint_ph_charts/modf_rntg2'


def _melt_for_facets(df: pd.DataFrame, x_var: str, y_vars: list, hue_var: str) -> pd.DataFrame:
    id_vars = [x_var]
    if hue_var:
        id_vars.append(hue_var)
        
    # Melt the dataframe: y_vars become 'facet_var', their values become 'y_value'
    return pd.melt(
        df,
        id_vars=id_vars,
        value_vars=y_vars,
        var_name='facet_var',
        value_name='y_value'
    )


def plot_grouped_boxplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, col_wrap: int = 3):
    long_df = _melt_for_facets(df, x_var, y_vars, hue_var)
    
    g = sns.catplot(
        data=long_df,
        x=x_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='box',
        col_wrap=col_wrap,
        # sharey=False,  
        # height=4,
        # aspect=0.5
        # legend='full',
        # legend_out=True,
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(x_var, "IoU")
    return g


def plot_grouped_violinplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var: str, col_wrap: int = 3, split: bool = False):
    long_df = _melt_for_facets(df, x_var, y_vars, hue_var)
    
    g = sns.catplot(
        data=long_df,
        x=x_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='violin',
        split=split,           # binary hue var
        # inner='quartile',  
        col_wrap=col_wrap,
        # sharey=False,
        # height=4,
        # aspect=0.1,
        legend='full',
        legend_out=True,
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(x_var, "Value")
    return g


def plot_grouped_barplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, col_wrap: int = 3):
    long_df = _melt_for_facets(df, x_var, y_vars, hue_var)
    
    g = sns.catplot(
        data=long_df,
        x=x_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='bar',
        capsize=0.1,           # Adds caps to the error bars for readability
        col_wrap=col_wrap,
        # sharey=False,
        # height=4,
        # aspect=1.2
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(x_var, "Mean Value")
    return g


def _prepare_and_melt(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int) -> tuple:
    data = df.copy()
    
    # Create the binned X variable
    binned_x_name = f"{x_var}_binned"
    data[binned_x_name] = pd.cut(data[x_var], bins=bins)
    
    id_vars = [binned_x_name]
    if hue_var:
        id_vars.append(hue_var)
        
    # Melt the dataframe
    long_df = pd.melt(
        data,
        id_vars=id_vars,
        value_vars=y_vars,
        var_name='facet_var',
        value_name='y_value'
    )
    
    return long_df, binned_x_name


def plot_binned_grouped_boxplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int = 5, col_wrap: int = 3):
    long_df, x_plot_var = _prepare_and_melt(df, x_var, y_vars, hue_var, bins)
    
    g = sns.catplot(
        data=long_df,
        x=x_plot_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='box',
        col_wrap=col_wrap,
        # sharey=False,
        height=4,
        aspect=1.2
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(f"{x_var} (Binned)", "Value")
    g.set_xticklabels(rotation=45, ha='right')
    return g


def plot_binned_grouped_violinplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int = 5, col_wrap: int = 3, split: bool = False):
    long_df, x_plot_var = _prepare_and_melt(df, x_var, y_vars, hue_var, bins)
    
    g = sns.catplot(
        data=long_df,
        x=x_plot_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='violin',
        split=split,
        inner='quartile',
        col_wrap=col_wrap,
        # sharey=False,
        height=4,
        aspect=1.2
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(f"{x_var} (Binned)", "Value")
    g.set_xticklabels(rotation=45, ha='right')
    return g


def plot_binned_grouped_barplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int = 5, col_wrap: int = 3):
    long_df, x_plot_var = _prepare_and_melt(df, x_var, y_vars, hue_var, bins)
    
    g = sns.catplot(
        data=long_df,
        x=x_plot_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='bar',
        capsize=0.1,
        col_wrap=col_wrap,
        # sharey=False,
        # height=4,
        # aspect=1.2
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(f"{x_var} (Binned)", "Mean Value")
    # g.set_xticklabels(rotation=45, ha='right')
    return g


def plot_grouped_stripp(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, col_wrap: int = 3):
    long_df = _melt_for_facets(df, x_var, y_vars, hue_var)
    
    g = sns.stripplot(
        data=long_df,
        x=x_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        # kind='box',
        col_wrap=col_wrap,
        # sharey=False,          # Set to False to allow independent Y-axis scales
        # height=4,
        # aspect=1.2
    )
    
    # g.set_titles("{col_name}")
    # g.set_axis_labels(x_var, "IoU")
    return g


IOU_COLS_GDA = sorted(["3_test_GDA_iou", "1_test_GDA_iou", "2_test_GDA_iou"])
IOU_COLS_SYNT = sorted(["3_test_SYNT_iou", "1_test_SYNT_iou", "2_test_SYNT_iou"])
IOU_COLS_DK = sorted(["3_test_DK_iou", "1_test_DK_iou", "2_test_DK_iou"])

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK
IOU_COLS_SO = sorted(IOU_COLS)
IOU_COLS1 = IOU_COLS_SO[:3]
IOU_COLS2 = IOU_COLS_SO[3:6]
IOU_COLS3 = IOU_COLS_SO[6:]
IOU123 = [IOU_COLS1, IOU_COLS2, IOU_COLS3]


def by_ph_scen_ema_trval(df):
    plot_grouped_barplot(
        df=df,
        x_var='ema',
        y_vars=IOU_COLS1,
        hue_var='tr_val'
    )
    plot_grouped_violinplot(
        df=df,
        x_var='ema',
        y_vars=IOU_COLS2,
        hue_var='tr_val'
        # hue_var=None
    )
    plot_grouped_boxplot(
        df=df,
        x_var='ema',
        y_vars=IOU_COLS3,
        hue_var='tr_val'
    )


def by_ph_scen_trval_ema(df):
    plot_grouped_barplot(
        df=df,
        x_var='tr_val',
        y_vars=IOU_COLS1,
        hue_var='ema'
    )
    plot_grouped_violinplot(
        df=df,
        x_var='tr_val',
        y_vars=IOU_COLS2,
        hue_var='ema'
        # hue_var=None
    )
    plot_grouped_boxplot(
        df=df,
        x_var='tr_val',
        y_vars=IOU_COLS3,
        hue_var='ema'
    )


def by_ph_scen_trval_bs(df):
    plot_grouped_barplot(
        df=df,
        x_var='batch_size',
        y_vars=IOU_COLS1,
        hue_var='tr_val'
    )
    plot_grouped_violinplot(
        df=df,
        x_var='batch_size',
        y_vars=IOU_COLS2,
        hue_var='tr_val'
        # hue_var=None
    )
    plot_grouped_boxplot(
        df=df,
        x_var='batch_size',
        y_vars=IOU_COLS3,
        hue_var='tr_val'
    )


def q(df, x, h, ious=IOU123, fff=plot_grouped_barplot, t='all'):
    for iou in ious:
        g = fff(df, x, iou, h)
        if SAVING:
            fn = '_'.join(['joint', x, str(h), str(iou), f's_lvl={t}', str(fff.__name__)])
            g.savefig(f'{DIR}/{fn}')
            print('saved', fn)
            plt.close()
        else:
            plt.show()


def hists(df: pd.DataFrame, ious=IOU_COLS, h='tr_val'):
    df_long = df.melt(id_vars=[h], value_vars=ious, var_name='ph_trg', value_name='iou') # 3_test_GDA_iou
    df_long[['phase', 'test data']] = df_long['ph_trg'].str.split('_', n=1, expand=True)
    df_long['test data'] = df_long['test data'].str.split('_', expand=True)[1]
    df_long['phase'] = pd.to_numeric(df_long['phase'], downcast='integer')
    # sns.displot(df_long,
    #             x='iou',
    #             col='phase',
    #             row='test data',
    #             row_order=["DK", "GDA", "SYNT"],
    #             # stat="density",
    #             stat="probability",
    #             # kind='kde',
    #             common_norm=False,
    #             hue='tr_val',
    # )
    sns.displot(df_long,
                x='iou',
                col='phase',
                row='test data',
                row_order=["DK", "GDA", "SYNT"],
                kind='kde',
                common_norm=False,
                hue=h,
    )
    # sns.displot(df_long,
    #             x='iou',
    #             row='phase',
    #             col='test data',
    #             col_order=["DK", "GDA", "SYNT"],
    #             stat="density",
    #             common_norm=False
    # )
    plt.show()


def hists2d(df: pd.DataFrame, the_snd, ious=IOU_COLS, h: str | None ='tr_val'):
    idv = [the_snd]
    if h:
        idv.append(h)
    df_long = df.melt(id_vars=idv, value_vars=ious, var_name='ph_trg', value_name='iou') # 3_test_GDA_iou
    df_long[['phase', 'test data']] = df_long['ph_trg'].str.split('_', n=1, expand=True)
    df_long['test data'] = df_long['test data'].str.split('_', expand=True)[1]
    df_long['phase'] = pd.to_numeric(df_long['phase'], downcast='integer')
    # sns.displot(df_long,
    #             x='iou',
    #             y=the_snd,
    #             col='phase',
    #             row='test data',
    #             row_order=["DK", "GDA", "SYNT"],
    #             stat="probability",
    #             common_norm=False,
    #             hue=h,
    # )
    sns.displot(df_long,
                x='iou',
                y=the_snd,
                row='phase',
                col='test data',
                col_order=["DK", "GDA", "SYNT"],
                stat="density",
                common_norm=False,
                hue=h,
                log_scale=(False, True)
    )
    plt.show()


def valid(df, t):
    # q(df, 'ema', 'tr_val', t=t) # saved
    q(df, 'ema', 'tr_val', fff=plot_grouped_boxplot, t=t) # err
    q(df, 'ema', 'tr_val', fff=plot_grouped_violinplot, t=t)
    # hists(df)
    # hists2d(df, 'Runtime')
    # hists2d(df, 'Runtime', h=None)


def mn():
    fff = f'{DIR}/cnc123b.csv'
    df = pd.read_csv(fff)
    valid(df, 'all')
    # TODO splt by gda lvl or sth else (gda, dk, s)
    df['s_lvl'] = ['s' in x for x in df['train']]
    for t in [True, False]:
        valid(df[df['s_lvl'] == t], t)

    # by_ph_scen_trval_ema(df) # ok, deprec by q
    # by_ph_scen_ema_trval(df)
    # by_ph_scen_trval_bs(df)
    # q(df, 'ema', 'tr_val')
    # hists(df)
    # hists2d(df, 'Runtime')
    # hists2d(df, 'Runtime', h=None)


    # plot_grouped_boxplot(
    #     df=df,
    #     x_var='loss',
    #     y_vars=IOU_COLS_GDA,
    #     # hue_var='tr_val'
    #     hue_var=None
    # )

    # plot_grouped_barplot(
    #     df=df,
    #     x_var='loss',
    #     y_vars=IOU_COLS,
    #     hue_var='tr_val'
    # )
    # plot_grouped_violinplot(
    #     df=df,
    #     x_var='ema',
    #     y_vars=IOU_COLS,
    #     hue_var='tr_val'
        # hue_var=None
    # )
 
    # plot_grouped_barplot(
    #     df=df,
    #     x_var='lrdec',
    #     y_vars=IOU_COLS,
    #     # hue_var='tr_val'
    #     hue_var=None
    # )

    # plot_binned_grouped_barplot(
    #     df=df,
    #     x_var='Runtime',
    #     y_vars=IOU_COLS,
    #     hue_var='tr_val'
    #     # hue_var=None
    # )

    # sns.stripplot(
    #     data=df,
    #     x='Runtime',
    #     y=IOU_COLS_GDA[0],
    #     hue='tr_val',
    #     # jitter=False,
    #     # dodge=True,
    # )
    # plt.show()

    # plot_grouped_boxplot(
    #     df=df,
    #     x_var='Runtime',
    #     y_vars=IOU_COLS,
    #     hue_var='tr_val'
    #     # hue_var=None
    # )


import numpy as np
from scipy import stats
from typing import Optional, Literal

def relplot_aggregated_shared_x(
    df: pd.DataFrame,
    x: str,
    y: str,
    col: str,
    key: str,
    hue: Optional[str] = None,
    ci: float = 95,
    ci_method: Literal["t", "bootstrap"] = "t",
    n_boot: int = 1000,
    palette: Optional[dict] = None,
    marker: str = "o",
    linewidth: float = 2.0,
    alpha_ci: float = 0.25,
    figsize=(12, 4),
    random_state: Optional[int] = None,
):
    """
    Use seaborn.relplot to draw aggregated mean +/- CI lines where:
      - mean_x is computed per (hue, key) and shared across columns
      - mean_y and CI are computed per (col, hue, key)
    Returns the FacetGrid and the aggregated DataFrame used for plotting.
    """
    rng = np.random.default_rng(random_state)

    # 1. Clean and required columns
    cols_needed = [x, y, col, key] + ([hue] if hue is not None else [])
    dfc = df[cols_needed].dropna(subset=[x, y, col, key]).copy()

    # 2. Compute mean_x per (hue, key) or (key)
    agg_x_group = [key] if hue is None else [hue, key]
    mean_x = dfc.groupby(agg_x_group, as_index=False)[x].mean().rename(columns={x: "mean_x"})

    # 3. Compute mean_y and CI per (col, hue, key)
    group_cols = [col, key] + ([hue] if hue is not None else [])
    rows = []
    def t_ci(arr, conf=ci):
        arr = np.asarray(arr)
        n = arr.size
        if n <= 1:
            return np.nan, np.nan
        m = np.nanmean(arr)
        se = stats.sem(arr, nan_policy="omit")
        alpha = 1 - conf / 100.0
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        return m - tcrit * se, m + tcrit * se

    def boot_ci(arr, conf=ci, n_iter=n_boot):
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan, np.nan
        boot = rng.choice(arr, size=(n_iter, arr.size), replace=True)
        means = boot.mean(axis=1)
        lower = np.percentile(means, (100 - conf) / 2)
        upper = np.percentile(means, 100 - (100 - conf) / 2)
        return lower, upper

    for name, g in dfc.groupby(group_cols):
        vals = g[y].dropna().values
        mean_y = np.nan if vals.size == 0 else vals.mean()
        if ci_method == "t":
            lo, hi = t_ci(vals)
        else:
            lo, hi = boot_ci(vals)
        row = dict(zip(group_cols, name if isinstance(name, tuple) else (name,)))
        row.update({"mean_y": mean_y, "ci_lower": lo, "ci_upper": hi, "n": vals.size})
        rows.append(row)
    agg_y = pd.DataFrame(rows)

    # 4. Merge mean_x into agg_y so each (col,hue,key) has the shared mean_x
    merge_on = [key] + ([hue] if hue is not None else [])
    plot_df = agg_y.merge(mean_x, how="left", on=merge_on)
    plot_df = plot_df.dropna(subset=["mean_x", "mean_y"]).copy()

    # 5. Plot with seaborn.relplot using precomputed points
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        x="mean_x",
        y="mean_y",
        hue=hue,
        col=col,
        kind="line",
        estimator=None,      # use the precomputed means
        marker=marker,
        linewidth=linewidth,
        palette=palette,
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        legend="brief",
    )

    # 6. Overlay CI bands per facet and per hue
    # For each axis (facet) draw fill_between for each hue level present in that facet
    for ax in g.axes.flat:
        # determine the column value from the axis title
        title = ax.get_title()
        # seaborn titles are like "col = value" or just "value" depending on version
        if "=" in title:
            col_val = title.split(" = ", 1)[1].strip()
        else:
            col_val = title.strip()
        facet_df = plot_df[plot_df[col].astype(str) == str(col_val)]
        if facet_df.empty:
            continue
        hue_levels = facet_df[hue].unique() if hue is not None else [None]
        for h in hue_levels:
            sub = facet_df if h is None else facet_df[facet_df[hue] == h]
            if sub.empty:
                continue
            sub = sub.sort_values("mean_x")
            xvals = sub["mean_x"].values
            lower = sub["ci_lower"].values
            upper = sub["ci_upper"].values
            # choose color consistent with the line color
            if hue is not None:
                # get color from legend or palette
                try:
                    color = g._legend.get_lines()[list(hue_levels).index(h)].get_color()
                except Exception:
                    color = None
            else:
                color = None
            ax.fill_between(xvals, lower, upper, color=color, alpha=alpha_ci, linewidth=0)

    # 7. Labels and layout
    g.set_axis_labels(f"mean {x}", f"mean {y} (±{ci}% CI)")
    plt.tight_layout()
    return g, plot_df


RNTMS = ['1_cum_Runtime', '2_cum_Runtime', '3_cum_Runtime']

def rename_column(col):
    # Match: phase_test_SET_t
    m = re.match(r"(\d+)_test_([A-Z]+)_iou$", col)
    if m:
        phase, set_name = m.groups()
        return f"iou_{set_name}_{phase}"

    # Match: phase_cum_m
    m = re.match(r"(\d+)_cum_Runtime$", col)
    if m:
        phase = m.group(1)
        return f"Runtime_{phase}"

    # Leave other columns unchanged (e.g. key)
    return col


def plot_pnt_line_joint_rnt(df: pd.DataFrame, q=1, x='Runtime', h='comb_key', ious=IOU_COLS, rntms=RNTMS):
    df = df.rename(columns=rename_column)
    df = df.drop(columns=['Runtime'])
    df = df.reset_index(names="row_id")
    # print(df.columns)
    df_long = (pd.wide_to_long(df, stubnames=['iou_SYNT', 'iou_GDA', 'iou_DK', 'Runtime'], i=['row_id', h], j='phase', sep='_', suffix=r"\d+")).reset_index()
    # df_long = (pd.wide_to_long(df, stubnames=['iou_SYNT', 'iou_GDA', 'iou_DK', 'Runtime'], i=[h], j='ph', sep='_', suffix=r"\d+")).reset_index()
    # print(df_long.columns)
    # print(df_long.head(10))
    df_long = pd.melt(df_long, id_vars=[h, 's_lvl', 'phase', 'Runtime'], value_vars=['iou_SYNT', 'iou_GDA', 'iou_DK'], var_name='test data', value_name='iou')
    df_long['test data'] = df_long['test data'].str.replace('iou_', '')
    # print(df_long.columns)
    # print(df_long.head(10))
    # df_long = df_long.head(444)
    if q == 1:
        return sns.displot(
            data=df_long,
            x="Runtime",
            y="iou",
            hue=h,
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            log_scale=(True, False),
        )
    df_long = df_long.reset_index()
    if q == 2:
        return sns.catplot(
            data=df_long,
            kind="point",
            x="Runtime",
            y="iou",
            hue=h,
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            estimator='mean',
        )
    if q == 3:
        return sns.relplot(
            data=df_long,
            x="Runtime",
            y="iou",
            hue=h,
            style='phase',
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
        )
    ho = df_long[h].unique()
    ho.sort()
    print(len(ho), 'hos')
    if q == 4:
        return sns.relplot(
            data=df_long.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            style=h,
            hue_order=ho.sort(),
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            markers=True,
            estimator="mean",
            errorbar=("ci", 95),
            sort=True,
        )
    plot_df = df_long.groupby(by=[h, 'test data', 'phase'], as_index=False).agg(iou=('iou', 'mean'), Runtime=('Runtime', 'mean')) # TODO these go to new file
    if q == 5:
        return sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=ho.sort(),
            style=h,
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            markers=True,
            sort=True,
        )
    # plot_df = df_long.groupby(by=[h, 'phase'], as_index=False).agg(iou=('iou', 'mean'), Runtime=('Runtime', 'mean'))
    # plot_df_iou = df_long.groupby(by=[h, 'phase', 'test data'], as_index=False).agg(iou=('iou', 'mean')) # IoU by ph, data, key
    # plot_df_runtime = df_long.groupby(by=[h, 'phase'], as_index=False).agg(Runtime=('Runtime', 'mean')) # time by ph, key
    plot_df_iou = df_long.groupby(by=[h, 'phase', 'test data', 's_lvl'], as_index=False).agg(iou=('iou', 'mean')) # IoU by ph, data, key
    plot_df_runtime = df_long.groupby(by=[h, 'phase', 's_lvl'], as_index=False).agg(Runtime=('Runtime', 'mean')) # time by ph, key
    # print(plot_df_iou.columns)
    # print(plot_df_runtime.columns)
    # print(plot_df_iou.head())
    # print(plot_df_runtime.head())
    # plot_df = pd.merge(left=plot_df_iou, right=plot_df_runtime, on=[h, 'phase'])
    plot_df = pd.merge(left=plot_df_iou, right=plot_df_runtime, on=[h, 'phase', 's_lvl'])
    # print(plot_df.columns)
    # print(plot_df.head())
    if q == 6: # only this one makes sense!!!!!!!!!!!!!!!!!!!
        return sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=ho.sort(),
            style=h,
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            markers=True,
            sort=True,
        )
    if q == 7: # only this one makes sense!!!!!!!!!!!!!!!!!!!
        return sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=ho.sort(),
            style=h,
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            row='s_lvl',
            markers=True,
            sort=True,
        )
    if q == 9: # exp 8,9
        return sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=ho.sort(),
            # style='phase',
            style=h,
            col="test data",
            col_order=["DK", "GDA", "SYNT"],
            row='s_lvl',
            markers=True,
            sort=True,
            palette=sns.color_palette(),
        )
    if q == 10: # exp swap r-c
        return sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=ho.sort(),
            # style='phase',
            style=h,
            row="test data",
            row_order=["DK", "GDA", "SYNT"],
            col='s_lvl',
            markers=True,
            sort=True,
            palette=sns.color_palette(),
        )
    if q == 11: # exp swap r-c, ho
        return sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=sorted(ho, key=lambda x: str(x).count('s')),
            # style='phase',
            style=h,
            row="test data",
            row_order=["DK", "GDA", "SYNT"],
            col='s_lvl',
            markers=True,
            sort=True,
            palette=sns.color_palette(),
        )
    if q == 12: # exp swap r-c, ho, leg
        g = sns.relplot(
            data=plot_df.sort_values(by='phase'),
            kind="line",
            x="Runtime",
            y="iou",
            hue=h,
            hue_order=sorted(ho, key=lambda x: str(x).count('s')),
            # style='phase',
            style=h,
            row="test data",
            row_order=["DK", "GDA", "SYNT"],
            col='s_lvl',
            markers=True,
            sort=True,
            palette=sns.color_palette(),
        )
        for (i, j, k), dt in g.facet_data():
            # (g.axes[i, j]).add_legend()
            sns.move_legend(g.axes[i, j], loc='right')
        return g
    return sns.relplot(
        data=plot_df.sort_values(by='phase'),
        kind="line",
        x="Runtime",
        y="iou",
        hue=h,
        hue_order=ho,
        style=h,
        col="test data",
        col_order=["DK", "GDA", "SYNT"],
        markers=True,
        sort=True,
    )


def mn_joint():
    fff = f'{DIR}/ph123b.csv'
    df = pd.read_csv(fff)
    # TODO splt by gda lvl or sth else (gda, dk, s)
    # df['s_lvl'] = ['s' in x for x in df['train']]
    df['s_lvl'] = [x.count('s') for x in df['comb_key']]

    df['1_cum_Runtime'] = df['Runtime_x']
    df['2_cum_Runtime'] = df['Runtime_y'] + df['1_cum_Runtime']
    df['3_cum_Runtime'] = df['Runtime'] + df['2_cum_Runtime']
    # bar plot Runtime - phase - comb_key
    df_org = df.copy()
    # for t in range(0, 6):
    #     df = df_org[df_org['s_lvl'] == t]
    #     if df.empty:
    #         print(t, 'empty')
    #         continue
    #     print('s in comb_key', t)
    #     # for qqq in range(6):
    #     for qqq in [6]:
    #         g = plot_pnt_line_joint_rnt(df, qqq)
    #         # g.limi
    #         if SAVING:
    #             fn = '_'.join(['joint', str(qqq), 'comb_key', f's_lvl={t}'])
    #             g.savefig(f'{DIR}/{fn}')
    #             print('saved', fn)
    #             plt.close()
    #         else:
    #             plt.show()
    for t in ['all']:
        for qqq in [12]:
            g = plot_pnt_line_joint_rnt(df_org, qqq)
            if SAVING:
                fn = '_'.join(['joint', str(qqq), 'comb_key', f's_lvl={t}'])
                g.savefig(f'{DIR}/{fn}')
                print('saved', fn)
                plt.close()
            else:
                plt.show()


if __name__ == '__main__':
    # mn()
    mn_joint()
