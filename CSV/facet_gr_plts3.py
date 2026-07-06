import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _melt_for_facets(df: pd.DataFrame, x_var: str, y_vars: list, hue_var: str) -> pd.DataFrame:
    """
    Helper function to transform multiple Y variables into a long format 
    suitable for Seaborn faceting.
    """
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
    """
    Draws multi-facet grouped bivariate box plots to show distributions.
    """
    long_df = _melt_for_facets(df, x_var, y_vars, hue_var)
    
    g = sns.catplot(
        data=long_df,
        x=x_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='box',
        col_wrap=col_wrap,
        # sharey=False,          # Set to False to allow independent Y-axis scales
        # height=4,
        # aspect=0.5
        legend='full',
        legend_out=True,
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(x_var, "IoU")
    # plt.tight_layout()
    plt.show()
    # return g

def plot_grouped_violinplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var: str, col_wrap: int = 3, split: bool = False):
    """
    Draws multi-facet grouped bivariate violin plots to show density distributions.
    Set split=True if hue_var has exactly two levels to draw split violins.
    """
    long_df = _melt_for_facets(df, x_var, y_vars, hue_var)
    
    g = sns.catplot(
        data=long_df,
        x=x_var,
        y='y_value',
        hue=hue_var,
        col='facet_var',
        kind='violin',
        split=split,           # Useful for binary hue variables
        # inner='quartile',      # Shows quartiles inside the violin
        col_wrap=col_wrap,
        # sharey=False,
        # height=4,
        # aspect=0.1,
        legend='full',
        legend_out=True,
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels(x_var, "Value")
    # plt.tight_layout()
    plt.show()
    return g

def plot_grouped_barplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, col_wrap: int = 3):
    """
    Draws multi-facet grouped bivariate bar plots to show the mean and 95% CI.
    """
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
    # plt.tight_layout()
    plt.show()
    return g


def _prepare_and_melt(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int) -> tuple:
    """
    Bins the continuous X variable and melts multiple Y variables into long format.
    Returns the melted DataFrame and the name of the new binned X column.
    """
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
    """Draws multi-facet grouped box plots across binned X values."""
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
    g.set_xticklabels(rotation=45, ha='right') # Essential for binned labels
    plt.tight_layout()
    plt.show()
    return g


def plot_binned_grouped_violinplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int = 5, col_wrap: int = 3, split: bool = False):
    """Draws multi-facet grouped violin plots across binned X values."""
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
    plt.tight_layout()
    plt.show()
    return g


def plot_binned_grouped_barplot(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, bins: int = 5, col_wrap: int = 3):
    """Draws multi-facet grouped bar plots (Mean + 95% CI) across binned X values."""
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
    plt.tight_layout()
    plt.show()
    return g


def plot_grouped_stripp(df: pd.DataFrame, x_var: str, y_vars: list, hue_var, col_wrap: int = 3):
    """
    Draws multi-facet grouped bivariate box plots to show distributions.
    """
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
    plt.tight_layout()
    plt.show()
    # return g

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


def q(df, x, h, ious=IOU123, fff=plot_grouped_barplot):
    for iou in ious:
        fff(df, x, iou, h)


def hists(df: pd.DataFrame, ious=IOU_COLS, h='tr_val'):
    # df = pd.melt(
    #     df,
    #     # id_vars=id_vars,
    #     value_vars=ious,
    #     var_name='facet_var',
    #     value_name='y_value'
    # )
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


def valid(df):
    # q(df, 'ema', 'tr_val')
    # q(df, 'ema', 'tr_val', fff=plot_grouped_boxplot)
    q(df, 'ema', 'tr_val', fff=plot_grouped_violinplot)
    # hists(df)
    # hists2d(df, 'Runtime')
    # hists2d(df, 'Runtime', h=None)


def mn():
    fff = 'CSV/joint_ph_charts/modf_rntg2/cnc123b.csv'

    df = pd.read_csv(fff)
    # valid(df)
    # TODO splt by gda lvl or sth else (gda, dk, s)
    df['s_lvl'] = ['s' in x for x in df['train']]
    for t in [True, False]:
        valid(df[df['s_lvl'] == t])

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

RNTMS = ['1_cum_Runtime', '2_cum_Runtime', '3_cum_Runtime']

def plot_pnt_line_joint_rnt(df: pd.DataFrame, x='Runtime', h='comb_key', ious=IOU_COLS, rntms=RNTMS):
    df_long = df.melt(id_vars=[h]+RNTMS, value_vars=ious, var_name='ph_trg', value_name='iou') # 3_test_GDA_iou
    print(df_long.columns)
    print(df_long.head())
    # df_long = df_long.melt(id_vars=[h, 'ph_trg', 'iou'], value_vars=rntms, var_name='ph_rnt', value_name='rnt') # 3_cum_Runtime
    df_long[['phase', 'test data']] = df_long['ph_trg'].str.split('_', n=1, expand=True) # 3, test_GDA_iou
    df_long['test data'] = df_long['test data'].str.split('_', expand=True)[1] # GDA
    df_long = df_long.melt(id_vars=[h, 'phase', 'iou', 'test data'], value_vars=rntms, var_name='ph_rnt', value_name='runtime') # 3_cum_Runtime
    df_long[['phase', 'rrr']] = df_long['ph_rnt'].str.split('_', n=1, expand=True) # 3, cum_Runtime
    # df_long['test data'] = df_long['test data'].str.split('_', expand=True)[1] # GDA
    # df_long['phase'] = pd.to_numeric(df_long['phase'], downcast='integer')
    print(df_long.columns)
    print(df_long.head())

    sns.catplot(
    # sns.displot(
        data=df_long,
        kind="point",
        # x="phase",
        x="runtime",
        y="iou",
        hue=h,
        col="test data",
        col_order=["DK", "GDA", "SYNT"],
        # aspect=0.5,
        # sharey=shx,
    )
    plt.plot()


def mn_joint():
    fff = 'CSV/joint_ph_charts/modf_rntg2/ph123b.csv'
    df = pd.read_csv(fff)
    # TODO splt by gda lvl or sth else (gda, dk, s)
    df['s_lvl'] = ['s' in x for x in df['train']]
    # for t in [True, False]:
    #     valid(df[df['s_lvl'] == t])
    df['1_cum_Runtime'] = df['Runtime_x']
    df['2_cum_Runtime'] = df['Runtime_y'] + df['1_cum_Runtime']
    df['3_cum_Runtime'] = df['Runtime'] + df['2_cum_Runtime']
    # bar plot Runtime - phase - comb_key
    plot_pnt_line_joint_rnt(df)


if __name__ == '__main__':
    # mn()
    mn_joint()
