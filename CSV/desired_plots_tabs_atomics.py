'''
data post-processing
(for display, after joins)

saving tabs & charts
'''


import pandas as pd
import seaborn as sns
import seaborn.objects as so
import re
import numpy as np
from new_workload import add_workload
from os import makedirs

SAVING = True
SAVING = False

SAVEDIR = 'CSV/joint_ph_charts/selected3/g'
SAVEDIR = 'joint_ph_charts/selected3/u2'

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

HPARAM_COLS_BASE_CAT = ['train', 'sub', 'tr_val', 'loss', 'val', 'src', 'batch_size', 'wd', 'lrdec', 'ema', 'warmup_epochs', 'lrenc', 'GPU']
HPARAM_COLS_BASE_REL = ['Runtime', 'workload']
HPARAM_COLS_BASE = ['epochs', 'ID', 'epochs_done', 'epoch', 'Sweep', 'fn', ] + HPARAM_COLS_BASE_CAT + HPARAM_COLS_BASE_REL
CALC_COLS_BASE = ['re_t', 'do_t', 're_v', 'real', 'w_v', 'w_t', 'dom', 'do_v', 'sub_mult', 'dist']

SNGL_COLS_BASES = CALC_COLS_BASE + HPARAM_COLS_BASE + RAW_IOU_COLS + ['SYNT use', 'DK use', 'GDA use']
JOINT_COLS_BASES = SNGL_COLS_BASES + ['Walltime', 'Workload', 'cumul. SYNT use', 'cumul. DK use', 'cumul. GDA use', 'cumul. no. unique DS'] # the cums

SNGL_FIXED_COLS = ['entry_id']
GLOB_FIXED_COLS = ['comb_key', 'ck', 'trains', 'Sworkload', 'total no. unique DS', 'total SYNT use', 'total DK use', 'total GDA use', 
                   'total SYNT use tr', 'total DK use tr', 'total GDA use tr',
             'DS_scores_sum', 'Sdom_raw', 'Sreal_raw', 'DS_score_tot_raw', 'DS_score_tot',
             'Sdom', 'Sreal', 'DS_score_raw', 'DS_score', 'ddiff'] + SNGL_FIXED_COLS


def total_df_treatment(pth: str, limit=False, round=False, joint=False, cnc=False, off=not LMT, endecja=0, sngl_ph_nr=3, cut_to_ph1=False):
    '''
    many operations, mainly col aggregations and df elongation
    '''
    df = pd.read_csv(pth)
    df = df.rename(columns={'Unnamed: 0': 'entry_id'})
    st = SNGL_COLS_BASES
    if limit:
        df = limit_to_successful(df, cnc, off)
    if joint:
        df = process_comb_cum_calcs(df)
        st = JOINT_COLS_BASES
    if not joint and not cnc:
        df = process_sngl_calcs(df)
    if round:
        df = round_sngl_ph(df, endecja)
    df = the_major_elongation(df, st, sngl_ph_nr)
    makedirs(SAVEDIR, exist_ok=True)
    df = df.rename(columns={'tr_val': 'train_val'})
    df = df.fillna({'ema': False}) # TODO map composite, mix to numerics 1/12 and cast to numerics, plot
    df.loc[(df['train'] == 's') & (df['sub'].isna()), 'sub'] = '100'
    if joint:
            df = add_workload(df, 'entry_id', 'phase')
            df = df.sort_values(by=['entry_id', 'phase', 'test set'])
            df[['Nworkload', 'workload_delta']] = df[['Nworkload', 'workload_delta']].bfill()
            if 'Workload' in df.columns:
                df = df.drop(columns=['Workload'])
            df = df.rename(columns={'Nworkload': 'Workload', 'workload_delta': 'Workload increase'})
    if not joint and not cnc and 'workload' in df.columns:
        df = df.rename(columns={'workload': 'Workload'})
    if cut_to_ph1:
        df = df[df['phase'] == 1]
        df = process_sngl_calcs(df, 'train_val')
    df = make_categorical(df, ['phase', 'ema', 'sub', 'comb_key', 'trains', 'train_val', 'cumul. no. unique DS',
                               'cumul. SYNT use', 'cumul. DK use', 'cumul. GDA use', 'loss', 'val', 'fn',
                               'total no. unique DS', 'total SYNT use', 'total DK use', 'total GDA use', 'train', 'src', 'test set'])
    print(df.columns)
    print(df.head())
    return df


def make_categorical(df: pd.DataFrame, vars=['phase']):
    '''
    pandas Categorical type for cols
    '''
    for x in vars:
        if x in df.columns:
            if x == 'sub':
                df[x] = pd.Categorical(df[x], categories=['mix', 'composite', '15', '25', '35', '45', '55', '65', '75', '100'], ordered=True)
            else:
                df[x] = pd.Categorical(df[x])
    return df


def process_sngl_calcs(df: pd.DataFrame, trv='tr_val'):
    '''
    sngl phase (local) aggregations
    '''
    # trv = 'train_val' if 'train_val' in df.columns else 'tr_val_1'
    print('trv', trv)
    print(df[trv])
    df['SYNT use'] = [x.count('s') for x in df[trv]] # how many SYNTs: s, sub, subm
    df['SYNT use'] += [x.count('_m') for x in df[trv]] # plus how many MIXs: _m
    df['DK use'] = [x.count('dk') for x in df[trv]] # how many DKs: dk
    df['DK use'] += [x.count('m') for x in df[trv]] # plus how many MIXs: m
    df['GDA use'] = [x.count('gda') for x in df[trv]] # how many GDAs
    print(df['SYNT use'])
    return df


def standardise_runtime_gpus():
    pass # TODO


# to jednak SYNTH/DK use też tak?? nie - tam zwykła liczność, 
# tu (workload) - ile wygenerowanych (praca) w użyciu (użyteczna praca), nie ile razy użyte


# def _parse_calc_comb_key_workload(comb_key: pd.Series, sub_mult: pd.Series, ph: int):
#     '''
#     :param comb_key: substring (cumulative) to process
#     :param sub_mult: fraction (to map with a dict) for synth tr, else 1
#     '''
#     # already done:
#     # S_SUB_S_FCT = {
#     #     'mix': 100/12,
#     #     'composite': 100/12
#     # }
#     # sub_map = df['sub'].map(S_SUB_S_FCT) # mix
#     # df['sub_mult'] = sub_map.fillna(df['sub']) # numb
#     # df['sub_mult'] = df['sub_mult'].fillna(100) # NaN - no sub
#     # df['sub_mult'] = df['sub_mult'].astype(float) / 100

#     # and also previous solution with mix handling but wrong cumulation later (here separate phase values)
#     # def add_workload(df: pd.DataFrame, fn='') -> pd.DataFrame:
#     # # train
#     # msk = df['train'].isin(['m', 'subm'])
#     # w_t_base = df['train'].map(DS_W_MULT) * df['train'].map(DS_SIZES_TR) * df['sub_mult'] # sub_mult == 1 for non SYNT
#     # w_t_m = (
#     #     DS_W_MULT['s'] * DS_SIZES_TR['s'] * df['sub_mult'] +
#     #     DS_W_MULT['dk'] * DS_SIZES_TR['dk']    
#     # )
#     # df['w_t'] = w_t_base.where(~msk, w_t_m)

#     # # val
#     # if (len(fn.split('_')) > 3): # ph3
#     #     msk = df['val'] == 'gda'
#     #     w_v_base = df['val'].map(DS_W_MULT) * df['val'].map(DS_SIZES_VAL)
#     #     w_v_m = (
#     #         DS_W_MULT['gda'] * DS_SIZES_VAL['gda_v']
#     #     )
#     #     df['w_v'] = w_v_base.where(~msk, w_v_m)
#     # else:
#     #     msk = df['val'].isin(['m', 'subm'])
#     #     w_v_base = df['val'].map(DS_W_MULT) * df['val'].map(DS_SIZES_VAL)
#     #     w_v_m = (
#     #         DS_W_MULT['s'] * DS_SIZES_VAL['s'] +
#     #         DS_W_MULT['dk'] * DS_SIZES_VAL['dk']
#     #     )
#     #     df['w_v'] = w_v_base.where(~msk, w_v_m)

#     # # final
#     # df['workload'] = df['w_t'] + df['w_v']
#     # return df

#     DS_SIZES_TR = {
#         's': 8614,
#         'sub': 8614, # s, later mult
#         'dk': 324,
#         'gda': 18,
#     }
#     DS_SIZES_VAL = {
#         's': 1846,
#         'sub': 1846, # s, later mult
#         'dk': 160,
#         'gda': 20,
#         'gda_v': 2,
#     }
#     DS_W_MULT = {
#         's': 0.1,
#         'sub': 0.1,
#         'gda': 1,
#         'dk': 1,
#     }
#     if ph == 2:
#         p1, p2 = comb_key.str.split('|') # dk_gda|sub_gda
#         t1, v1 = p1.split('_')
#         t2, v2 = p2.split('_')
#         ts = max(t1.map(DS_SIZES_TR)*t1.map(DS_W_MULT)*sub_mult, t2.map(DS_SIZES_TR)*t2.map(DS_W_MULT)*sub_mult) # if t1 & t2 == 's' or 'sub'
#         vs = max(v1.map(DS_SIZES_VAL)*v1.map(DS_W_MULT), v2.map(DS_SIZES_VAL)*v2.map(DS_W_MULT)) # if v1 & v2 == 's' or 'sub'
#         tdk = max(t1.map(DS_SIZES_TR)*t1.map(DS_W_MULT)*sub_mult, t2.map(DS_SIZES_TR)*t2.map(DS_W_MULT)*sub_mult) # if t1 & t2 == 'dk'
#         vdk = max(v1.map(DS_SIZES_VAL)*v1.map(DS_W_MULT), v2.map(DS_SIZES_VAL)*v2.map(DS_W_MULT)) # if v1 & v2 == 'dk'
#         tgda = max(t1.map(DS_SIZES_TR)*t1.map(DS_W_MULT)*sub_mult, t2.map(DS_SIZES_TR)*t2.map(DS_W_MULT)*sub_mult) # if t1 & t2 == 'gda'
#         vgda = max(v1.map(DS_SIZES_VAL)*v1.map(DS_W_MULT), v2.map(DS_SIZES_VAL)*v2.map(DS_W_MULT)) # if v1 & v2 == 'gda'
#         for series in [ts, vs, tdk, ]:
#             series = series.fillna(0)
#         return ts+vs+tdk+vdk+tgda+vgda
#     if ph == 3:
#         p1, p2, p3 = comb_key.str.split('|') # dk_gda|sub_gda|subm_dk
#         t1, v1 = p1.split('_')
#         t2, v2 = p2.split('_')
#         t3, v3 = p3.split('_')
#         ts = max(t1.map(DS_SIZES_TR)*t1.map(DS_W_MULT)*sub_mult, 
#                  t2.map(DS_SIZES_TR)*t2.map(DS_W_MULT)*sub_mult,
#                  t3.map(DS_SIZES_TR)*t3.map(DS_W_MULT)*sub_mult,) # if t1 & t2 == 's' or 'sub'



# def _parse_calc_comb_key_workload(comb_key: pd.Series, sub_mult: pd.Series, ph: int) -> pd.Series:
#     '''
#     Calculates cumulative workload metric across training phases.
#     Takes max of dataset variants across phases to avoid double-counting.
#     Unfolds dataset mixes ('m', 'subm') into their base dataset components.
    
#     :param comb_key: pd.Series of strings (e.g., "dk_gda|subm_gda" or "dk_gda|subm_gda|s_gda")
#     :param sub_mult: pd.Series of floats representing synthetic fraction (already processed)
#     :param ph: int, number of phases (1, 2, or 3)
#     :return: pd.Series of cumulative workloads
#     '''
    
#     DS_SIZES_TR = {
#         's': 8614,
#         'sub': 8614,
#         'dk': 324,
#         'gda': 18,
#     }
#     DS_SIZES_VAL = {
#         's': 1846,
#         'sub': 1846,
#         'dk': 160,
#         'gda': 20,
#         'gda_v': 2,
#     }
#     DS_W_MULT = {
#         's': 0.1,
#         'sub': 0.1,
#         'gda': 1.0,
#         'dk': 1.0,
#     }

#     n_rows = len(comb_key)
    
#     # Initialize accumulators to track the MAXIMUM workload seen for each base component
#     max_tr_synth = np.zeros(n_rows)
#     max_tr_dk = np.zeros(n_rows)
#     max_tr_gda = np.zeros(n_rows)
    
#     max_val_synth = np.zeros(n_rows)
#     max_val_dk = np.zeros(n_rows)
#     max_val_gda = np.zeros(n_rows)
    
#     # Based on the old code: phase 3 experiments use the 'gda_v' size for val
#     val_gda_size = DS_SIZES_VAL['gda_v'] if ph >= 3 else DS_SIZES_VAL['gda']
    
#     # Pre-extract strings to avoid multiple str operations
#     # expand=True creates a DataFrame where columns are phases 0, 1, (and 2)
#     phases_split = comb_key.str.split('|', expand=True)
    
#     # Keep sub_mult as a fast numpy array to avoid Pandas index alignment issues in loops
#     sub_mult_arr = sub_mult.to_numpy()

#     for p in range(ph):
#         if p >= phases_split.shape[1]:
#             break
            
#         # Extract train and val strings for the current phase
#         phase_str = phases_split[p].fillna('_')
#         tv_split = phase_str.str.split('_', expand=True)
        
#         t_col = tv_split[0] if 0 < tv_split.shape[1] else pd.Series(index=comb_key.index, dtype=str).fillna('')
#         v_col = tv_split[1] if 1 < tv_split.shape[1] else pd.Series(index=comb_key.index, dtype=str).fillna('')

#         # Create numpy masks for fast filtering
#         t_is_synth = t_col.isin(['s', 'sub']).to_numpy()
#         t_is_mix   = t_col.isin(['m', 'subm']).to_numpy()
#         t_is_dk    = (t_col == 'dk').to_numpy()
#         t_is_gda   = (t_col == 'gda').to_numpy()

#         v_is_synth = v_col.isin(['s', 'sub']).to_numpy()
#         v_is_mix   = v_col.isin(['m', 'subm']).to_numpy()
#         v_is_dk    = (v_col == 'dk').to_numpy()
#         v_is_gda   = (v_col == 'gda').to_numpy()

#         # ====================
#         # TRAIN CALCULATION
#         # ====================
#         # Note: According to your old 'm' logic, sub_mult should ONLY scale the synthetic ('s') portion.
#         # dk and gda maintain their full size regardless of the synthetic subset factor.
#         curr_tr_synth = np.where(t_is_synth | t_is_mix, DS_SIZES_TR['s'] * DS_W_MULT['s'] * sub_mult_arr, 0.0)
#         curr_tr_dk    = np.where(t_is_dk | t_is_mix, DS_SIZES_TR['dk'] * DS_W_MULT['dk'], 0.0)
#         curr_tr_gda   = np.where(t_is_gda, DS_SIZES_TR['gda'] * DS_W_MULT['gda'], 0.0)

#         max_tr_synth = np.maximum(max_tr_synth, curr_tr_synth)
#         max_tr_dk    = np.maximum(max_tr_dk, curr_tr_dk)
#         max_tr_gda   = np.maximum(max_tr_gda, curr_tr_gda)

#         # ====================
#         # VAL CALCULATION
#         # ====================
#         # Note: Validation never uses sub_mult in your original code.
#         curr_val_synth = np.where(v_is_synth | v_is_mix, DS_SIZES_VAL['s'] * DS_W_MULT['s'], 0.0)
#         curr_val_dk    = np.where(v_is_dk | v_is_mix, DS_SIZES_VAL['dk'] * DS_W_MULT['dk'], 0.0)
#         curr_val_gda   = np.where(v_is_gda, val_gda_size * DS_W_MULT['gda'], 0.0)

#         max_val_synth = np.maximum(max_val_synth, curr_val_synth)
#         max_val_dk    = np.maximum(max_val_dk, curr_val_dk)
#         max_val_gda   = np.maximum(max_val_gda, curr_val_gda)

#     # Sum the highest recorded usages of every base dataset piece across all phases
#     total_workload = (
#         max_tr_synth + max_tr_dk + max_tr_gda +
#         max_val_synth + max_val_dk + max_val_gda
#     )
    
#     return pd.Series(total_workload, index=comb_key.index, name='workload')





def process_comb_cum_calcs(df: pd.DataFrame):
    '''
    operates on the wide joint df

    add walltime, workload, levels, cnts
    '''
    df['1_Walltime'] = df['Runtime_x']
    df['2_Walltime'] = df['Runtime_y'] + df['1_Walltime']
    df['3_Walltime'] = df['Runtime'] + df['2_Walltime']

    # df['1_Workload'] = df['workload_x']
    # df['2_Workload'] = df['workload_y'] + df['1_Workload']
    # df['3_Workload'] = df['workload'] + df['2_Workload']

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

    df['Sworkload'] = 0 # df['3_Workload'] # sum

    # df['total no. unique DS'] = df['comb_key'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) # number of DSs
    df['ck'] = df['comb_key'].str.replace('subm', 'dk_s')
    df['ck'] = df['ck'].str.replace('sub', 's')
    df['ck'] = df['ck'].str.replace('m', 'dk_s')
    df['total no. unique DS'] = df['ck'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) # number of DSs
    df['1_cumul. no. unique DS'] = df['tr_val_x'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst))))
    df['2_cumul. no. unique DS'] = df['tr_val_y'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) + df['1_cumul. no. unique DS']
    df['3_cumul. no. unique DS'] = df['tr_val'].str.split(r'_|\|').apply(lambda lst: len(set(map(str.strip, lst)))) + df['2_cumul. no. unique DS']

    df['total SYNT use'] = [x.count('s') for x in df['comb_key']] # how many SYNTs: s, sub, subm
    df['total SYNT use'] += [x.count('_m') for x in df['comb_key']] # plus how many MIXs: _m
    df['total DK use'] = [x.count('dk') for x in df['comb_key']] # how many DKs: dk
    df['total DK use'] += [x.count('m') for x in df['comb_key']] # plus how many MIXs: m
    df['total GDA use'] = [x.count('gda') for x in df['comb_key']] # how many GDAs
    df['total SYNT use tr'] = [x.count('s') for x in df['trains']] # how many SYNTs in train only
    df['total SYNT use tr'] += [x.count('_m') for x in df['trains']] # plus how many MIXs: _m
    df['total DK use tr'] = [x.count('dk') for x in df['trains']] # how many DKs: dk
    df['total DK use tr'] += [x.count('m') for x in df['trains']] # plus how many MIXs: m
    df['total GDA use tr'] = [x.count('gda') for x in df['trains']] # how many GDAs in train only
    # TODO tak, val only też

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
    if not SAVING:
        print(sorted(iou))
    iou = sorted(iou)[0]
    return df[df[iou] > 0.617]


def the_major_elongation(df: pd.DataFrame, stubs=SNGL_COLS_BASES, sngl_ph_nr=None):
    '''
    unify naming convention of cols, elongate params, elongate IoUs
    '''
    print('entered the_major_elongation')
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
            num = sngl_ph_nr

        # prefix-based: 1_, 2_, 3_
        m = re.match(r'([1-3])_(.+)', col)
        if m:
            num = int(m.group(1))
            base = m.group(2)

        rename_map[col] = f"{base}_{num}"

    if not SAVING:
        print('the rename map is')
        print(rename_map)
    df = df.rename(columns=rename_map)
    # pd.set_option('display.max_columns', None)
    if not SAVING:
        print(df.head(1).T.to_string())
        print(stubs)

    df = pd.wide_to_long(
        df,
        stubnames=stubs,
        i=SNGL_FIXED_COLS,
        j='phase',
        sep='_'
    ).reset_index() # test_SYNT_iou; 2
    if not SAVING:
        print(df.head(1).T.to_string())
    print('done wide_to_long')

    hs = list(set(df.columns) - set(RAW_IOU_COLS))
    if not SAVING:
        print(hs)
        for hc in hs:
            if hc not in df.columns:
                print(hc, 'is not in df')
        print(df.columns)
        print(df.dtypes)
    df = pd.melt(df, id_vars=hs, value_vars=RAW_IOU_COLS, var_name='phase_set', value_name='IoU')
    print('done IoU melt')
    df['test set'] = df['phase_set'].str.split('_', n=2, expand=True)[1]
    df = df.drop(columns=['phase_set'])
    print(df.shape)
    print('done IoU sets and whole the_major_elongation')
    return df


def rels(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='line', size=None, xord=None, dg='auto'):
    '''
    line/scatter
    '''
    if not SAVING:
        print('rels:',x, y, h, c,r,s,ch,size, xord, h_ord)
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


def cats(df: pd.DataFrame, x, y, h, c=None, c_ord=None, r=None, r_ord=None, h_ord=None, s=None, ch='bar', size=None, xord=None, dg='auto'):
    '''
    box/viol/bar/point/count/boxen/strip/swarm
    '''
    xs = x is not None and df[x].nunique() > 4
    hs = h is not None and (df[h].nunique() > 4 or 'tot' in h)
    rc = r is None and c is not None
    print('xs', xs, 'hs', hs, 'rc', rc)
    if xs or hs: # TODO univ ver
        r, r_ord = c, c_ord
        c, c_ord = None, None
    if not SAVING:
        print('cats:', x, y, h, c, r, xord, h_ord)
    if ch == 'violin':
        # if r is None and c is not None or df['phase'].nunique() == 1: # let violins fit
        #     r, r_ord = c, c_ord
        #     c, c_ord = None, None
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
            # cut=1,
            cut=0,
            density_norm='count',
            order=xord,
            aspect=1.8,
            dodge=dg,
            # TODO wspace?
        )
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
        order=xord,
        dodge=dg,
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


def plot_prod(g, x, y, h, col, row, t='', bs=None, xl='', min_max_labs=pd.DataFrame(), plot_labs=True):
    '''
    labels, base lines
    '''
    if not SAVING:
        print('plot prod', x, y, h, col, row, t, xl)
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

    if not min_max_labs.empty and not SAVING:
        print('min_max_labs/agg')
        print(min_max_labs.head(30))
    mrg = 0.003
    if plot_labs: # TODO value from agg, loc from vert, cut=def
        # if 'phase' in min_max_labs.columns and 'test set' in min_max_labs.columns:
            # min_max_labs = min_max_labs.sort_values(by=['phase', 'test set'])
        for ax in g.axes.flat: # correct viol data via cut=0; THIS ONE WORKS
            violins = [c for c in ax.collections if isinstance(c, collections.PolyCollection)]
            if not SAVING:
                print(ax.get_title())
            for v in violins:
                if len(v.get_paths()) > 0:
                    verts = v.get_paths()[0].vertices
                    y_vals = verts[:, 1]
                    ymin, ymax = y_vals.min(), y_vals.max()
                    x_center = np.mean(verts[:, 0])    
                    # ax.text(x_center, ymin-mrg, f"{ymin:.3f}", ha="center", va="top")
                    ax.text(x_center, ymax+mrg, f"{ymax:.3f}", ha="center", va="bottom")
        # i_base = 0
        # for ax in g.axes.flat: # the correct way to put df values onto the plot if test set present
        #     mml = min_max_labs.copy()
        #     # print(mml.dtypes)
        #     axt = ax.get_title()
        #     fj = True
        #     if '|' in axt:
        #         axt = axt.split('|')
        #         phh = axt[0].split(' = ')[1].strip()
        #         mml = mml[mml['phase'] == int(phh)]
        #         print('phase', f'*{phh}*', mml.size, mml.empty)
        #         axt = axt[1]
        #         fj = False
        #     ts = axt.split(' = ')[1]
        #     if ts == 'DK' and fj:
        #         ts = 'SYNT'
        #     elif ts == 'SYNT' and fj:
        #         ts = 'DK'
        #     print(ax.get_title(), i_base, ts, 'g')
        #     mml = mml[mml['test set'] == ts]
        #     print(mml.head(10))
        #     # print(axes.Axes().name)
        #     print(ax.get_title(), ax.title, ax.get_label(), ax.name)
        #     violins = [c for c in ax.collections if isinstance(c, collections.PolyCollection)]
        #     j = 0
        #     print('i_base', i_base, 'j', j)
        #     for i, v in enumerate(violins):
        #         print(v.get_label())
        #         print(v.axes.title, v.axes.name)
        #         if len(v.get_paths()) > 0:
        #             verts = v.get_paths()[0].vertices
        #             y_vals = verts[:, 1]
        #             ymin, ymax = y_vals.min(), y_vals.max()
        #             yminf, ymaxf = f"{ymin:.3f}", f"{ymax:.3f}"
        #             yminfo, ymaxfo = f"{mml.iat[j, 3]:.3f}", f"{mml.iat[j, 7]:.3f}"
        #             # yminfo, ymaxfo = f"{min_max_labs.iat[j+i_base, 3]:.3f}", f"{ min_max_labs.iat[j+i_base, 7]:.3f}"
        #             # yminfo, ymaxfo = f"{min_max_labs['min'].iloc[i+i_base]:.3f}", f"{ min_max_labs['max'].iloc[i+i_base]:.3f}"
        #             if ymaxf != ymaxfo:
        #                 print(i, ymaxf, ymaxfo)
        #             if yminf != yminfo:
        #                 print(i, yminf, yminfo)
        #             x_center = np.mean(verts[:, 0])
        #             ax.text(x_center, ymin-mrg, yminfo, ha="center", va="top")
        #             ax.text(x_center, ymax+mrg, ymaxfo, ha="center", va="bottom")
        #             # ax.text(x_center, ymin-mrg, yminf, ha="center", va="top")
        #             # ax.text(x_center, ymax+mrg, ymaxf, ha="center", va="bottom")
        #             j += 1
        #     i_base += j
    return g
