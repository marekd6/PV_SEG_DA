'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


from os import makedirs
from desired_plots_tabs_atomics import *


DIR = 'CSV/joint_ph_charts/res_dfs4'
SAVEDIR = 'CSV/joint_ph_charts/selected3/f'

FILES = {
    'joint': f'{DIR}/ph123b.csv',
    'ph1': f'{DIR}/proc_ph1.csv',
    'subs_16': f'{DIR}/subs_16.csv',
    'concat': f'{DIR}/cnc123b.csv',
}


HS_SNGL = ['train_val', 'SYNT use', 'DK use', 'GDA use']
HS_JOINT = ['total no. unique DS', 'total SYNT use', 'total DK use', 'total GDA use']
HS_CUM = ['cumul. no. unique DS', 'cumul. SYNT use', 'cumul. DK use', 'cumul. GDA use']
HS_ALL = HS_SNGL + HS_CUM + HS_JOINT

CH_BBV = ['bar', 'box', 'violin']
# CH_BBV = ['bar']
# CH_BBV = ['violin']
CH_CAT = CH_BBV + ['line']


def save_plt_df(df: pd.DataFrame, g, fu_name: str, chart: str, keyy: str, sv_df=False, swap_dir_ord=False, xtra=''):
    if SAVING:
        if xtra != '':
            keyy = str(keyy) + '_' + xtra
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
            print(ppp)
            plt.close()
    elif not sv_df:
        plt.show()


def rep_plts(df, x, y, h, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu, agg, xord=None, xtra=''):
    min_max_labs = agg
    for ch in chs:
        if not SAVING:
            print(df.head(1))
        if ch == 'line2':
            g = line(df, x, y, h, col, col_order, row, row_ord, ch=ch, h_ord=h_ord, s=s, size=s)
        else:
            g = ch_fu(df, x, y, h, col, col_order, row, row_ord, ch=ch, h_ord=h_ord, s=s, size=s, xord=xord)
        g = plot_prod(g, x, y, h, bs=bs, t=tit, xl=xl, min_max_labs=min_max_labs)
        save_plt_df(df, g, fu, ch, h, xtra=xtra) # g


def repeat_plot(df: pd.DataFrame, fu: str, x='phase', y='IoU', hs=HS_JOINT, h_ord=None, 
                col='test set', col_order=["DK", "GDA", "SYNT"], row='phase', row_ord=[1, 2, 3], 
                chs=CH_BBV, ch_fu=cats, s=None, bs=None, tit='', xl=''):
    '''
    plot & agg, save
    '''
    xo = False
    xord = None
    if row == None:
        row_ord = None
    if len(hs) == 0:
        gr = [x, 'test set']
        print('gr by', gr)
        agg = df.groupby(gr)[y].agg(min='min', Q1=lambda x: x.quantile(0.25), mean='mean',
                                    Q3=lambda x: x.quantile(0.75), max='max').reset_index()
        if not SAVING:
            print(agg.head())
        rep_plts(df, x, y, None, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu, agg)
        save_plt_df(agg, None, fu, '', '', True) # df
        # save_plt_df(df.groupby(by=h).agg('mean'), g, f'{fu}_agg', ch, h, True) # df TODO save wide/agg df
    for h in hs:
        xtra = ''
        xx = x
        gr = [x, h, 'test set']
        if x == 'h':
            gr.remove(x)
            x = h
            xx = h
            xo = True
        elif x == 'hh':
            gr.remove(x)
            xx = h
        if xo:
            xord = df[gr].drop_duplicates().sort_values(h)[x].tolist()
            xtra=h
        gr = list(set(gr))
        print('gr by', gr)
        agg = df.groupby(gr)[y].agg(min='min', Q1=lambda x: x.quantile(0.25), mean='mean',
                                    Q3=lambda x: x.quantile(0.75), max='max').reset_index()
        if not SAVING:
            print(agg.head())
        print(agg['max'].size, agg['max'].count(), x, xx, h, gr)
        rep_plts(df, xx, y, h, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu, agg, xord, xtra)
        save_plt_df(agg, None, fu, '', h, True) # df
        # save_plt_df(df.groupby(by=h).agg('mean'), g, f'{fu}_agg', ch, h, True) # df TODO save wide/agg df


def joint_plts_auto_agg_4D(df: pd.DataFrame, fl: str='joint'): # OK - add more
    '''x=phase, y=IoU, col=set, h=HS_JOINT; bar (viol, box, pt-line - może ten lineplot, nie catplot)'''
    # repeat_plot(df, f'{fl}_phase', row=None, row_ord=None, bs=[0.71, 0.617, 0.359], ch_fu=ucats, chs=['line']) # TODO ucats???

    # repeat_plot(df, f'{fl}_phase', hs=HS_ALL, row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV) # -----------------------------------------
    repeat_plot(df, f'{fl}_phase', hs=HS_CUM, row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=['phase'], row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV)
    repeat_plot(df, f'{fl}_phase', hs=[], row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV)


def joint_plts_auto_agg_5D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=h, y=IoU, col=set, h=HS_CUM, row=phase; bar (viol, box)'''
    print(df.dtypes)
    repeat_plot(df, f'{fl}_phase9', hs=HS_CUM, bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 9
    repeat_plot(df, f'{fl}_phase3', hs=HS_CUM, row=None, bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 3
    # repeat_plot(df, f'{fl}_phase', bs=[0.71, 0.617, 0.359], chs=CH_BBV) # not cum
    repeat_plot(df, f'{fl}_phase9', hs=['phase'], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 9
    repeat_plot(df, f'{fl}_phase3', hs=['phase'], row=None,bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 3
    repeat_plot(df, f'{fl}_phase9', hs=[], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 9
    repeat_plot(df, f'{fl}_phase3', hs=[], row=None,bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 3


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


def joint_plts_no_agg_4D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=Workload/Walltime, y=IoU, col=set, h=HS_JOINT; scatter'''
    repeat_plot(df, f'{fl}_Walltime3', 'Walltime', hs=HS_JOINT+HS_CUM, row=None, chs=['scatter'], ch_fu=rels, xl='[s]') # HS_ALL not
    repeat_plot(df, f'{fl}_Workload3', 'Workload', hs=HS_JOINT+HS_CUM, row=None, chs=['scatter'], ch_fu=rels) # HS_ALL not


def joint_plts_no_agg_5D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=Workload/Walltime, y=IoU, col=set, h=HS_SNGL, row=phase; scatter'''
    repeat_plot(df, f'{fl}_Walltime9', 'Walltime', hs=HS_JOINT+HS_CUM, chs=['scatter'], ch_fu=rels, xl='[s]') # HS_ALL not
    repeat_plot(df, f'{fl}_Workload9', 'Workload', hs=HS_JOINT+HS_CUM, chs=['scatter'], ch_fu=rels) # HS_ALL not


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


def ph1_plts_auto_agg_5D(df: pd.DataFrame, fl: str='ph1'): # OK
    '''x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)'''
    repeat_plot(df, f'{fl}_phase', x='h', hs=HS_SNGL, row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # x by h, xord by h
    repeat_plot(df, f'{fl}_phase', x='hh', hs=HS_SNGL, row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # self-self x-h
    repeat_plot(df, f'{fl}_phase', hs=HS_SNGL, row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # redundant but OK
    repeat_plot(df, f'{fl}_phase', row=None, hs=['phase'], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # no h
    repeat_plot(df, f'{fl}_phase', row=None, hs=[], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # no h


def ph1_plts_auto_agg_5D_v2(df: pd.DataFrame, fl: str='ph1'): # TODO Hparams more
    '''x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)'''
    print(df.dtypes)
    # repeat_plot(df, f'{fl}_phase', x='h', hs=['sub', 'val', 'train'], row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # x by h, xord by h
    dfs = df[~df['sub'].isna()]
    print('from', df.count(), 'to', dfs.count())
    repeat_plot(dfs, f'{fl}_phase', x='hh', hs=['sub'], row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # self-self x-h
    repeat_plot(df, f'{fl}_phase', x='hh', hs=['val', 'train'], row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # self-self x-h
    # repeat_plot(df, f'{fl}_phase', hs=HPARAM_COLS_BASE, row_ord=[1], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # redundant but OK


def concats_5D(df: pd.DataFrame, fl: str): # next
    '''x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)'''
    repeat_plot(df, f'{fl}_phase', row=None, hs=['phase'], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 3
    repeat_plot(df, f'{fl}_phase', hs=[], bs=[0.71, 0.617, 0.359], chs=CH_BBV) # 9


def all_joint123():
    ph123 = total_df_treatment(FILES['joint'], joint=True)
    joint_plts_auto_agg_4D(ph123)
    joint_plts_manual_agg_4D(ph123)
    joint_plts_auto_agg_5D(ph123)
    joint_plts_no_agg_4D(ph123)
    joint_plts_no_agg_5D(ph123)


def all_concat(): # TODO treatment nie działa
    cnc = total_df_treatment(FILES['concat'], cnc=True)


def all_1st_phase():
    '''ph1: joit and sngl'''
    ph1 = total_df_treatment(FILES['ph1'], sngl_ph_nr=1)
    ph1_plts_auto_agg_5D(ph1)
    ph1_plts_auto_agg_5D_v2(ph1)

    ph123 = total_df_treatment(FILES['joint'], joint=True)
    print(ph123['ema'].count())
    ph1 = ph123[ph123['phase'] == 1]
    # ph1_plts_auto_agg_5D(ph1, 'ph1_joint') # no
    ph1_plts_auto_agg_5D_v2(ph1, 'ph1_joint')


def main():
    all_joint123()
    # all_concat()
    all_1st_phase()


if __name__ == '__main__':
    main()
