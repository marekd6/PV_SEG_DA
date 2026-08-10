'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


from desired_plots_tabs_atomics import *
import multiprocessing


DIR = 'CSV/joint_ph_charts/res_dfs4'
DIR = 'joint_ph_charts/res_dfs7'

FILES = {
    'joint': f'{DIR}/ph123b.csv',
    'ph1': f'{DIR}/proc_ph1.csv',
    'subs_16': f'{DIR}/subs_16.csv',
    'concat': f'{DIR}/cnc123b.csv',
}


HS_SNGL = ['train_val', 'SYNT use', 'DK use', 'GDA use']
HS_JOINT = ['total no. unique DS', 'total SYNT use', 'total DK use', 'total GDA use']
HS_JOINT.extend(['total SYNT use tr', 'total DK use tr', 'total GDA use tr'])
HS_CUM = ['cumul. no. unique DS', 'cumul. SYNT use', 'cumul. DK use', 'cumul. GDA use']
HS_ALL = HS_SNGL + HS_CUM + HS_JOINT

CH_BBV = ['bar', 'box', 'violin']
if not SAVING:
    CH_BBV = ['bar']
    CH_BBV = ['violin']
CH_CAT = CH_BBV + ['line']

BASELINES = [0.71, 0.617, 0.359]


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


def rep_plts(df, x, y, h, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu, agg, xord=None, xtra='', dg='auto'):
    min_max_labs = agg
    for ch in chs:
        if not SAVING:
            # print(df.head(1))
            print('rep_plts', x, y, h, col, row, s)
        if ch == 'line2':
            g = line(df, x, y, h, col, col_order, row, row_ord, ch=ch, h_ord=h_ord, s=s, size=s)
        else:
            g = ch_fu(df, x, y, h, col, col_order, row, row_ord, ch=ch, h_ord=h_ord, s=s, size=s, xord=xord, dg=dg)
        g = plot_prod(g, x, y, h, col, row, bs=bs, t=tit, xl=xl, min_max_labs=min_max_labs)
        save_plt_df(df, g, fu, ch, h, xtra=xtra) # g


def repeat_plot(df: pd.DataFrame, fu: str, x='phase', y='IoU', hs=HS_JOINT, h_ord=None, 
                col='test set', col_order=["DK", "GDA", "SYNT"], row='phase', row_ord=[1, 2, 3], 
                chs=CH_BBV, ch_fu=cats, s=None, bs=None, tit='', xl='', aggs=True, dg='auto'):
    '''
    plot & agg, save
    '''
    if 'scatter' in chs:
        aggs=False
    xo = False
    xord = None # TODO
    if row == None:
        row_ord = None
    if col == None:
        col_order = None
    if len(hs) == 0:
        gr = list(set([q for q in [x, col, row] if q is not None]))
        if aggs and len(gr) > 0:
            print('gr by', gr)
            agg = df.groupby(gr)[y].agg(min='min', Q1=lambda x: x.quantile(0.25), mean='mean',
                                        Q3=lambda x: x.quantile(0.75), max='max').reset_index()
        else:
            agg = pd.DataFrame([1,1,1])
        rep_plts(df, x, y, None, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu, agg, dg)
        save_plt_df(agg, None, fu, '', '', True) # df
    for h in hs:
        if h not in df.columns:
            continue
        xtra = ''
        xx = x
        gr = list(set([q for q in [x, col, row, h] if q is not None]))
        if x == 'h':
            gr.remove(x)
            x = h
            xx = h
            xo = True # TODO
        elif x == 'hh':
            gr.remove(x)
            xx = h
        if xo:
            xord = df[gr].drop_duplicates().sort_values(h)[x].tolist()
            h_ord = xord # TODO
            xtra=h
        if aggs and len(gr) > 0:
            print('gr by', gr)
            agg = df.groupby(gr)[y].agg(min='min', Q1=lambda x: x.quantile(0.25), mean='mean',
                                        Q3=lambda x: x.quantile(0.75), max='max').reset_index()
        else:
            agg = pd.DataFrame([1,1,1])
        rep_plts(df, xx, y, h, col, col_order, row, row_ord, s, chs, fu, h_ord, bs, tit, xl, ch_fu, agg, xord, xtra, dg)
        save_plt_df(agg, None, fu, '', h, True) # df


def joint_plts_auto_agg_4D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=phase, y=IoU, col=set, h=HS_JOINT; bar (viol, box, pt-line - może ten lineplot, nie catplot)'''
    # repeat_plot(df, f'{fl}_phase', row=None, row_ord=None, bs=BASELINES, ch_fu=ucats, chs=['line']) # TODO ucats???
    # repeat_plot(df, f'{fl}_phase', hs=HS_ALL, row=None, bs=BASELINES) # -----------------------------------------
    repeat_plot(df, f'{fl}_phase', hs=HS_CUM, row=None, bs=BASELINES)
    repeat_plot(df, f'{fl}_phase', hs=HS_JOINT, row=None, bs=BASELINES)
    repeat_plot(df, f'{fl}_phase', hs=['phase'], row=None, bs=BASELINES)
    repeat_plot(df, f'{fl}_phase', hs=[], row=None, bs=BASELINES, dg=None)
    repeat_plot(df, f'{fl}_workload_delta3', y='Workload increase', row=None, hs=HS_CUM, col=None)
    repeat_plot(df, f'{fl}_workload_delta3', y='Workload increase', row=None, hs=['phase'], col=None)
    repeat_plot(df, f'{fl}_workload_delta9', y='Workload increase', hs=HS_CUM, col=None)
    repeat_plot(df, f'{fl}_workload_delta9', y='Workload increase', hs=['phase'], col=None)


def joint_plts_auto_agg_5D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=h, y=IoU, col=set, h=HS_CUM/phase/None, row=phase/None; bar (viol, box)'''
    repeat_plot(df, f'{fl}_phase9', hs=HS_CUM, bs=BASELINES) # 9
    repeat_plot(df, f'{fl}_phase3', hs=HS_CUM, row=None, bs=BASELINES) # 3
    # repeat_plot(df, f'{fl}_phase', bs=BASELINES) # not cum
    repeat_plot(df, f'{fl}_phase9', hs=['phase'], bs=BASELINES) # 9
    repeat_plot(df, f'{fl}_phase3', hs=['phase'], row=None,bs=BASELINES) # 3
    repeat_plot(df, f'{fl}_phase9', hs=[], bs=BASELINES, dg=None) # 9
    repeat_plot(df, f'{fl}_phase3', hs=[], row=None,bs=BASELINES, dg=None) # 3

    
    print(df['sub'].unique())
    dfs = df[~df['sub'].isna()]
    dfs['sub'] = pd.Categorical(dfs['sub'], categories=['mix', 'composite', '15', '25', '35', '100'], ordered=True)
    print(dfs['sub'].unique())
    repeat_plot(dfs, f'{fl}_phase9', x='hh', hs=['sub'], bs=BASELINES, dg=False)
    repeat_plot(df, f'{fl}_phase9b', x='hh', hs=['sub'], bs=BASELINES, dg=False)
    repeat_plot(df, f'{fl}_phase9', x='hh', hs=['sub_mult'], bs=BASELINES, dg=False)
    hg = HPARAM_COLS_BASE_CAT.copy()
    hg.remove('sub')
    repeat_plot(df, f'{fl}_phase9', x='hh', hs=hg, bs=BASELINES, dg=False)


def joint_plts_manual_agg_4D(df: pd.DataFrame, fl: str='joint'): # OK
    '''x=Workload/Walltime, y=IoU, col=set, h=HS_JOINT; pt-line'''
    for h in HS_JOINT:
        plot_df_iou = df.groupby(by=[h, 'phase', 'test set'], as_index=False).agg(IoU=('IoU', 'mean')) # IoU by ph, set, key
        plot_df_walltime = df.groupby(by=[h, 'phase'], as_index=False).agg(Walltime=('Walltime', 'mean')) # Walltime by ph, key
        plot_df_workload = df.groupby(by=[h, 'phase'], as_index=False).agg(Workload=('Workload', 'mean')) # Workload by ph, key
        plot_df_walltime = pd.merge(left=plot_df_iou, right=plot_df_walltime, on=[h, 'phase'])
        plot_df_workload = pd.merge(left=plot_df_iou, right=plot_df_workload, on=[h, 'phase'])
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


def ph1_plts_auto_agg_5D(df: pd.DataFrame, fl: str='ph1'): # OK
    '''x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)'''
    # repeat_plot(df, f'{fl}_phase', x='h', hs=HS_SNGL, row_ord=[1], bs=BASELINES) # x by h, xord by h
    repeat_plot(df, f'{fl}_phase', x='hh', hs=HS_SNGL, row_ord=[1], bs=BASELINES) # self-self x-h
    repeat_plot(df, f'{fl}_phase', hs=HS_SNGL, row_ord=[1], bs=BASELINES) # redundant but OK
    repeat_plot(df, f'{fl}_phase', row=None, hs=['phase'], bs=BASELINES) # no h
    repeat_plot(df, f'{fl}_phase', row=None, hs=[], bs=BASELINES, dg=None) # no h


def ph1_plts_auto_agg_5D_hps(df: pd.DataFrame, fl: str='ph1'): # TODO Hparams more
    '''x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)'''
    # repeat_plot(df, f'{fl}_phase', x='h', hs=['sub', 'val', 'train'], row_ord=[1], bs=BASELINES) # x by h, xord by h
    dfs = df[~df['sub'].isna()]
    if fl != 'ph1':
        dfs['sub'] = pd.Categorical(dfs['sub'], categories=['mix', 'composite', '15', '25', '35', '100'], ordered=True)
    repeat_plot(dfs, f'{fl}_phase', x='hh', hs=['sub'], row_ord=[1], bs=BASELINES, dg=False) # self-self x-h
    repeat_plot(df, f'{fl}_phase', x='hh', hs=['val', 'train'], row_ord=[1], bs=BASELINES, dg=False) # self-self x-h
    repeat_plot(df, f'{fl}_phase', x='hh', hs=HPARAM_COLS_BASE_CAT, row_ord=[1], bs=BASELINES, dg=False) # self-self x-h
    # repeat_plot(df, f'{fl}_phase', hs=HPARAM_COLS_BASE_CAT, row_ord=[1], bs=BASELINES) # redundant but OK


def ph1_plts_no_agg_5D(df: pd.DataFrame, fl: str='ph1'): # OK
    '''x=Workload/Walltime, y=IoU, col=None/h, h=HS_JOINT, row=phase; bar (viol, box)'''
    repeat_plot(df, f'{fl}_Workload', 'Workload', hs=[], col=None, row_ord=[1], ch_fu=rels, chs=['scatter'], aggs=True)
    repeat_plot(df, f'{fl}_Workload', 'Workload', hs=[], row_ord=[1], ch_fu=rels, chs=['scatter'], aggs=True)
    repeat_plot(df, f'{fl}_Workload', 'Workload', hs=HS_SNGL, col=None, row_ord=[1], ch_fu=rels, chs=['scatter'])
    df = df[df['test set'] == 'SYNT'] # no tripling
    hhhh = HS_SNGL if 'SYNT use' in df.columns else HS_JOINT + ['train_val']
    repeat_plot(df, f'{fl}_Workload', y='Workload', hs=hhhh, col=None, row_ord=[1], aggs=True)
    repeat_plot(df, f'{fl}_Workload', x='h', y='Workload', hs=hhhh, col=None, row_ord=[1], aggs=True, dg=False)
    repeat_plot(df, f'{fl}_Workload', x='hh', y='Workload', hs=hhhh, col=None, row_ord=[1], aggs=True, dg=False)


def concats_5D(df: pd.DataFrame, fl: str): # next
    '''x=h, y=IoU, col=set, h=HS_JOINT, row=phase; bar (viol, box)'''
    repeat_plot(df, f'{fl}_phase', row=None, hs=['phase'], bs=BASELINES) # 3
    repeat_plot(df, f'{fl}_phase', hs=[], bs=BASELINES, dg=None) # 9


def all_joint123():
    ph123 = total_df_treatment(FILES['joint'], joint=True)
    ph123.to_csv(f'{SAVEDIR}/ph123.csv')
    print(ph123.columns)
    joint_plts_auto_agg_4D(ph123)
    joint_plts_manual_agg_4D(ph123)
    joint_plts_auto_agg_5D(ph123)
    joint_plts_no_agg_4D(ph123)
    joint_plts_no_agg_5D(ph123)
    print('done joint')


def all_concat(): # TODO treatment nie działa
    cnc = total_df_treatment(FILES['concat'], cnc=True)


def all_1st_phase():
    '''ph1: joit and sngl'''
    ph1 = total_df_treatment(FILES['ph1'], sngl_ph_nr=1)
    ph1.to_csv(f'{SAVEDIR}/ph1.csv')
    ph1_plts_auto_agg_5D(ph1)
    ph1_plts_auto_agg_5D_hps(ph1)
    ph1_plts_no_agg_5D(ph1)

    ph1 = total_df_treatment(FILES['joint'], joint=True, cut_to_ph1=True)
    ph1.to_csv(f'{SAVEDIR}/ph1_joint.csv')
    ph1_plts_auto_agg_5D(ph1, 'ph1_joint')
    ph1_plts_auto_agg_5D_hps(ph1, 'ph1_joint')
    ph1_plts_no_agg_5D(ph1, 'ph1_joint')
    print('done 1st phase')


def main():
    all_joint123()
    # all_concat()
    all_1st_phase()


if __name__ == '__main__':
    # main()
    jnt = multiprocessing.Process(target=all_joint123)
    jnt.start()
    all_1st_phase()
    jnt.join()
    print('done')
