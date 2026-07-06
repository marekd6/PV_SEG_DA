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


def mn():
    fff = 'CSV/joint_ph_charts/modf_rnt/cnc123b.csv'
    fff = 'CSV/joint_ph_charts/modf_rntg2/cnc123b.csv'
    # fff = 'CSV/joint_ph_charts/modf_rnt/ph123b.csv'

    df = pd.read_csv(fff)
    # splt by gda lvl or sth else (gda, dk, s)

    # by_ph_scen_trval_ema(df)
    # by_ph_scen_ema_trval(df)
    # by_ph_scen_trval_bs(df)
    q(df, 'ema', 'tr_val')

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
    #     jitter=False,
    #     dodge=True,
    # )
    # plt.show()

    # plot_grouped_boxplot(
    #     df=df,
    #     x_var='Runtime',
    #     y_vars=IOU_COLS,
    #     hue_var='tr_val'
    #     # hue_var=None
    # )
    

if __name__ == '__main__':
    mn()
