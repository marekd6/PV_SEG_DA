'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


from os import makedirs
from desired_plots_tabs_atomics import *


DIR = 'CSV/joint_ph_charts/res_dfs3'
SAVEDIR = 'CSV/joint_ph_charts/selected3/a'

FILES = {
    'joint': f'{DIR}/ph123b.csv',
    'ph1': f'{DIR}/proc_ph1.csv',
    'subs_16': f'{DIR}/subs_16.csv',
    'ph2': f'{DIR}/proc_ph2.csv',
    'ph3': f'{DIR}/proc_ph3.csv',
    'concat': f'{DIR}/cnc123b.csv',
}


HS_SNGL = ['tr_val', 'dom', 'real', 'dist']
HS_JOINT = ['cnt_ds', 's_lvl', 'gda_lvl', 'dk_lvl'] # comb/joint
HS_CUM = ['cnt_ds_cum', 's_lvl_cum', 'gda_lvl_cum', 'dk_lvl_cum'] # TODO by cum aggs - 9 grids
HS_ALL = HS_SNGL + HS_JOINT

CH_BBV = ['bar', 'box', 'violin']
CH_CAT = CH_BBV + ['line']


def save_plt_df(df: pd.DataFrame, g, fu_name: str, chart: str, keyy: str, sv_df=False, swap_dir_ord=False):
    if SAVING:
        d = f'{SAVEDIR}/{keyy}/{fu_name}'
        if swap_dir_ord:
            d = f'{SAVEDIR}/{fu_name}/{keyy}'
            ppp = f'{d}/{chart}_{keyy}_{fu_name}.png'
        makedirs(d, exist_ok=True)
        ppp = f'{d}/{keyy}_{fu_name}_{chart}.png'
        ppdf = ppp.replace('png', 'csv')
        if sv_df:
            df.to_csv(ppdf)
        else:
            g.savefig(ppp)
            plt.close()
    elif not sv_df:
        plt.show()


def rep_plts(df, x, y, h, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu):
    for ch in chs:
        if not SAVING:
            print(df.head(1))
        g = ch_fu(df, x, y, h, col, col_order, row, row_ord, ch=ch, h_ord=h_ord, s=s, size=s)
        g = plot_prod(g, x, y, h, bs=bs, t=tit, xl=xl, add_viol_labs=(ch == 'violin'))
        save_plt_df(df, g, fu, ch, h) # g


def repeat_plot(df: pd.DataFrame, fu: str, x='phase', y='IoU', hs=HS_JOINT, h_ord=None, 
                col='test set', col_order=["DK", "GDA", "SYNT"], row='phase', row_ord=[1, 2, 3], 
                chs=CH_BBV, ch_fu=cats, s=None, bs=None, tit='', xl=''):
    '''
    plot & agg, save
    '''
    if row == None:
        row_ord = None
    if len(hs) == 0:
        rep_plts(df, x, y, None, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu)
        save_plt_df(df, None, fu, '', '', True) # df
        # save_plt_df(df.groupby(by=h).agg('mean'), g, f'{fu}_agg', ch, h, True) # df TODO save wide/agg df
    for h in hs:
        rep_plts(df, x, y, h, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu)
        save_plt_df(df, None, fu, '', h, True) # df
        # save_plt_df(df.groupby(by=h).agg('mean'), g, f'{fu}_agg', ch, h, True) # df TODO save wide/agg df


def joint_plts_auto_agg_4D(df: pd.DataFrame, fl: str='joint'): # OK - add more
    '''x=phase, y=IoU, col=set, h=HS_JOINT; bar (viol, box, pt-line - może ten lineplot, nie catplot)'''
    # repeat_plot(df, f'{fl}_phase', row=None, row_ord=None, bs=[0.71, 0.617, 0.359], ch_fu=ucats, chs=['line'])
    repeat_plot(df, f'{fl}_phase', row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=['phase'], row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=[], row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV)


def joint_plts_auto_agg_5D(df: pd.DataFrame, fl: str='joint'): # prawie
    '''x=h, y=IoU, col=set, h=HS_CUM, row=phase; bar (viol, box)'''
    repeat_plot(df, f'{fl}_phase', hs=HS_CUM, row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV) # TODO o, ten
    repeat_plot(df, f'{fl}_phase', row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=['phase'], row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=[], row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV)


def joint_plts_manual_agg_4D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=Workload/Walltime, y=IoU, col=set, h=HS_JOINT; pt-line'''
    for h in HS_JOINT:
        plot_df_iou = df.groupby(by=[h, 'phase', 'test set'], as_index=False).agg(IoU=('IoU', 'mean')) # IoU by ph, set, key
        plot_df_walltime = df.groupby(by=[h, 'phase'], as_index=False).agg(Walltime=('Walltime', 'mean')) # Walltime by ph, key
        plot_df_workload = df.groupby(by=[h, 'phase'], as_index=False).agg(Workload=('Workload', 'mean')) # Workload by ph, key
        plot_df_walltime = pd.merge(left=plot_df_iou, right=plot_df_walltime, on=[h, 'phase'])
        plot_df_workload = pd.merge(left=plot_df_iou, right=plot_df_workload, on=[h, 'phase'])
        # print(plot_df_walltime.head())
        repeat_plot(plot_df_walltime, f'{fl}_Walltime', 'Walltime', hs=[h], s=h, row=None, chs=['line'], ch_fu=rels, xl='[s]')
        # repeat_plot(plot_df_walltime, f'{fl}_Walltime', 'Walltime', hs=[h], s=h, row=None, chs=['line'], ch_fu=line, xl='[s]') # misleading CI per IoU only
        repeat_plot(plot_df_workload, f'{fl}_Workload', 'Workload', hs=[h], s=h, row=None, chs=['line'], ch_fu=rels)
        # repeat_plot(plot_df_workload, f'{fl}_Workload', 'Workload', hs=[h], s=h, row=None, chs=['line'], ch_fu=line) # misleading CI per IoU only


def joint_plts_no_agg_4D(df: pd.DataFrame, fl: str='joint'): # ?
    '''x=Workload/Walltime, y=IoU, col=set, h=HS_JOINT; scatter'''
    repeat_plot(df, f'{fl}_Walltime', 'Walltime', hs=HS_ALL, chs=['scatter'], ch_fu=rels, xl='[s]')
    repeat_plot(df, f'{fl}_Workload', 'Workload', hs=HS_ALL, chs=['scatter'], ch_fu=rels)


def joint_plts_no_agg_5D(df: pd.DataFrame, fl: str='joint'): # ?
    '''x=Workload/Walltime, y=IoU, col=set, h=HS_SNGL, row=phase; scatter'''
    repeat_plot(df, f'{fl}_Walltime', 'Walltime', hs=HS_ALL, chs=['scatter'], ch_fu=rels, xl='[s]')
    repeat_plot(df, f'{fl}_Workload', 'Workload', hs=HS_ALL, chs=['scatter'], ch_fu=rels)


# def concats_5D(df: pd.DataFrame, fl: str):
#     '''x=workload/runtime, y=IoU, h=HS_SNGL, col=set, row=phase; scatter'''
#     repeat_plot(df, f'{fl}_Runtime', 'Runtime', row='phase', hs=HS_SNGL, row_ord=[1, 2, 3], chs=['scatter'], ch_fu=rels, widen_fu=widen_phases, xl='[s]') # 3xWTs
#     repeat_plot(df, f'{fl}_workload', 'workload', row='phase', hs=HS_SNGL, row_ord=[1, 2, 3], chs=['scatter'], ch_fu=rels, widen_fu=widen_phases) # 3xWTs


# def generalised_sngl_ph(df: pd.DataFrame, fl: str):
#     # x=phase, y=IoU, h=HS_SNGL, row=phase, col=phase; bar...
#     # x=h, y=IoU, h=HS_SNGL, row=phase, col=phase; bar...
#     # x=runtime/workload, y=IoU, h=HS_SNGL, row=phase, col=phase; scatter

#     # generalised(df, f'{fl}_phase', hs=HS_SNGL, chs=['bar', 'box', 'violin'], ch_fu=cats, widen_fu=widen_phases, bs=[0.71, 0.617, 0.359], row='phase')
#     # generalised(df, f'{fl}_Runtime', 'Runtime', hs=HS_SNGL, chs=['scatter'], ch_fu=rels, widen_fu=widen_phases, row='phase', xl='[s]')
#     # generalised(df, f'{fl}_workload', 'workload', hs=HS_SNGL, chs=['scatter'], ch_fu=rels, widen_fu=widen_phases, row='phase')
#     if fl in ['subs_16', 'ph1']:
#         # generalised(df, f'{fl}_sub', 'sub', hs=['sub'], row='phase', chs=CH_BBV, ch_fu=cats_endlabs, widen_fu=widen_phases_h2) # only ph1
#         repeat_plot(df, f'{fl}_sub', 'sub', hs=['sub'], row='phase', chs=CH_BBV, ch_fu=cats_endlabs, widen_fu=widen_phases_h3) # only ph1


def ph1_plts_auto_agg_5D(df: pd.DataFrame, fl: str='ph1'): # OK, ale bez sensu
    # x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)
    repeat_plot(df, f'{fl}_phase', hs=HS_SNGL, row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=['phase'], row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=[], row='phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV)


def all_joint123():
    ph123 = total_df_treatment(FILES['joint'], joint=True)
    joint_plts_auto_agg_4D(ph123)
    joint_plts_manual_agg_4D(ph123)
    joint_plts_auto_agg_5D(ph123)
    joint_plts_no_agg_4D(ph123)
    # joint_plts_no_agg_5D(ph123)


def all_concat(): # TODO treatment nie działa
    cnc = total_df_treatment(FILES['concat'], cnc=True)


def all_1st_phase(): # TODO treatment nie działa?
    ph1 = total_df_treatment(FILES['ph1'])
    ph1_plts_auto_agg_5D(ph1)


def main():
    all_joint123()
    # all_concat()
    # all_1st_phase()


if __name__ == '__main__':
    main()
