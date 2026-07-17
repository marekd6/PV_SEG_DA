'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


from os import makedirs
from desired_plots_tabs_atomics import *


IOU_COLS_GDA = ["3_test_GDA_iou", "3_test/GDA/iou", "1_test/GDA/iou",  
                "2_test/GDA/iou", "1_test_GDA_iou", "2_test_GDA_iou"]
IOU_COLS_SYNT = ["3_test_SYNT_iou", "1_test/SYNT/iou", "2_test/SYNT/iou", 
            "3_test/SYNT/iou", "1_test_SYNT_iou", "2_test_SYNT_iou"]
IOU_COLS_DK = ["3_test_DK_iou", "1_test/DK/iou", "2_test/DK/iou", 
            "3_test/DK/iou", "1_test_DK_iou", "2_test_DK_iou"]

IOU_COLS = IOU_COLS_GDA + IOU_COLS_SYNT + IOU_COLS_DK

DIR = 'CSV/joint_ph_charts/res_dfs2'
SAVEDIR = 'CSV/joint_ph_charts/selected2/h'

FILES = {
    'joint': f'{DIR}/ph123b.csv',
    'ph1': f'{DIR}/proc_ph1.csv',
    'ph2': f'{DIR}/proc_ph2.csv',
    'ph3': f'{DIR}/proc_ph3.csv',
    'concat': f'{DIR}/cnc123b.csv',
}


HS_SNGL = ['tr_val', 'dom', 'real', 'dist']
HS_JOINT = ['cnt_ds', 's_lvl', 'gda_lvl', 'dk_lvl']
HS_ALL = HS_SNGL + HS_JOINT

# DS: workload (Ssize*factor) ===================== * EPOCHS done
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


def generalised(df: pd.DataFrame, fu: str, x='phase', y='IoU', col='test set', col_order=["DK", "GDA", "SYNT"],
                          row=None, row_ord=None, hs=HS_JOINT, chs=['point', 'bar', 'box', 'violin'], 
                          ch_fu=cats, widen_fu=widen_phases, s=None, bs=None, tit='', xl=''):
    '''
    widen, plot & agg, save

    all `hs` and `chs` for `ch_fu` and `widen_fu`
    '''
    df_org = df.copy()
    for h in hs:
        xx, hh, xy = x, h, x
        s, size, h_ord = None, None, None
        if x == 'hs':
            xx = 'phase'
            hh = None
            xy = h
        if widen_fu in [widen_runtime_agg, widen_runtime_no_agg, widen_workload_agg, widen_workload_no_agg]:
            df, h_ord = widen_fu(df_org, h)
            s = h
        elif x != 'phase' and x != 'hs' and widen_fu == widen_phases:
            df = widen_fu(df_org, h=[h, x])
            s = h
        else:
            df = widen_fu(df_org, xx, y, [h], col)
        for ch in chs:
            print(xy, y, hh, col, row, ch, h_ord)
            if not SAVING:
                print(df.head(1))
            g = ch_fu(df, xy, y, hh, col, col_order, row, row_ord, ch=ch, h_ord=h_ord, s=s, size=size)
            g = plot_prod(g, xy, y, hh, bs=bs, t=tit, xl=xl)
            save_plt_df(df, g, fu, ch, h) # g
        save_plt_df(df, g, fu, ch, h, True) # df
        # save_plt_df(df.groupby(by=h).agg('mean'), g, f'{fu}_agg', ch, h, True) # df TODO save wide/agg df


def generalised_joints_4D(df: pd.DataFrame, fl: str):
    generalised(df, f'{fl}_phase', ch_fu=cats, widen_fu=widen_phases, bs=[0.71, 0.617, 0.359]) # IoU avg+CI by ph, h, set
    generalised(df, f'{fl}_phase', hs=['phase'], ch_fu=cats, widen_fu=widen_phases_id, bs=[0.71, 0.617, 0.359]) # IoU avg+CI by ph, set
    generalised(df, f'{fl}_Walltime', "Walltime", chs=['line'], ch_fu=rels, widen_fu=widen_runtime_agg, xl='[s]') # IoU avg by (WT avg by ph, h, set) | (IoU) by ph, h, set | (WT) by ph, h | 3xWTs
    generalised(df, f'{fl}_Workload', "Workload", chs=['line'], ch_fu=rels, widen_fu=widen_workload_agg) # IoU avg by (WL avg by ph, h, set) | (IoU) by ph, h, set | (WL) by ph, h | 3xWLs
    generalised(df, f'{fl}_Walltime', 'Walltime', row='phase', hs=HS_ALL, row_ord=[1, 2, 3], chs=['scatter'], ch_fu=rels, widen_fu=widen_runtime_no_agg, xl='[s]') # 3xWTs
    generalised(df, f'{fl}_Workload', 'Workload', row='phase', hs=HS_ALL, row_ord=[1, 2, 3], chs=['scatter'], ch_fu=rels, widen_fu=widen_workload_no_agg) # 3xWLs
    generalised(df, f'{fl}_phase9', ch_fu=cats, x='hs', row='phase', row_ord=[1, 2, 3], chs=['bar', 'box', 'violin']) # by key


def generalised_concats_4D(df: pd.DataFrame, fl: str):
    generalised(df, f'{fl}_Runtime', 'Runtime', row='phase', hs=HS_SNGL, row_ord=[1, 2, 3], chs=['scatter'], ch_fu=rels, widen_fu=widen_phases, xl='[s]') # 3xWTs
    generalised(df, f'{fl}_workload', 'workload', row='phase', hs=HS_SNGL, row_ord=[1, 2, 3], chs=['scatter'], ch_fu=rels, widen_fu=widen_phases) # 3xWTs


def generalised_sngl_ph(df: pd.DataFrame, fl: str):
    generalised(df, f'{fl}_phase', hs=HS_SNGL, chs=['bar', 'box', 'violin'], ch_fu=cats, widen_fu=widen_phases, bs=[0.71, 0.617, 0.359], row='phase')
    generalised(df, f'{fl}_Runtime', 'Runtime', hs=HS_SNGL, chs=['scatter'], ch_fu=rels, widen_fu=widen_phases, row='phase', xl='[s]')
    generalised(df, f'{fl}_workload', 'workload', hs=HS_SNGL, chs=['scatter'], ch_fu=rels, widen_fu=widen_phases, row='phase')


def main():
    conc123 = limit_to_successful(round_sngl_ph(pd.read_csv(FILES['concat'])), cnc=True)
    ph123 = pd.read_csv(FILES['joint'])
    ph1 = limit_to_successful(round_sngl_ph(pd.read_csv(FILES['ph1'])))
    ph2 = limit_to_successful(round_sngl_ph(pd.read_csv(FILES['ph2'])))
    ph3 = limit_to_successful(round_sngl_ph(pd.read_csv(FILES['ph3'])))

    
    ph123 = process_diversity_workload(ph123)
    generalised_joints_4D(ph123, 'joint')
    generalised_sngl_ph(ph1, 'ph1')
    generalised_sngl_ph(ph2, 'ph2')
    generalised_sngl_ph(ph3, 'ph3')
    generalised_concats_4D(conc123, 'concat')


if __name__ == '__main__':
    main()
