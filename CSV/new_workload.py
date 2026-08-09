# """
# Long-format cumulative workload: one row per (run_id, phase_num), each
# phase carrying its OWN train/val/sub. Adds a running (expanding) 'workload'
# column that reflects the true cumulative total after each phase, without
# double-counting data reused/nested across phases.
# """
from __future__ import annotations
import pandas as pd

# --------------------------------------------------------------------- #
# Static dataset metadata (unchanged from the existing pipeline)
# --------------------------------------------------------------------- #

DS_SIZES_TR = {
    's':   374,
    'sub': 374,   # same underlying pool as 's', scaled later by sub_mult
    'dk':  324,
    'gda': 18,
}
DS_SIZES_VAL = {
    's':     80,
    'sub':   80,
    'dk':    160,
    'gda':   20,
    'gda_v': 2,     # small supplementary val set, only ever used via 'gda' unfolding in a 3-phase run
}
DS_W_MULT = {
    's':   1/20,
    'sub': 1/20,
    'gda': 1,
    'dk':  1,
}
S_SUB_S_FCT = {
    'mix':       100,
    'composite': 100,
}


def compute_sub_mult_and_category(sub_col: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    From the raw 'sub' column (NaN / 'mix' / 'composite' / a numeric-like
    percentage string or number) derive:
      - sub_mult : float fraction in [0, 1], applied to synthetic TRAIN sizes
      - sub_cat  : bucket flavor label, one of 'pct' / 'mix' / 'composite'
    """
    mapped = sub_col.map(S_SUB_S_FCT)
    sub_mult = mapped.fillna(sub_col)     # plain numeric % passes through untouched
    sub_mult = sub_mult.fillna(100)       # no 'sub' at all -> full 100%
    sub_mult = sub_mult.astype(float) / 100

    # sub_cat = sub_col.where(sub_col.isin(['mix', 'composite']), other='pct')
    sub_cat = pd.Series('pct', index=sub_col.index) # obtaining mix or comp - same effort as 100%
    return sub_mult, sub_cat


# --------------------------------------------------------------------- #
# Per-phase expansion into elementary (bucket -> value) contributions
# --------------------------------------------------------------------- #

def _expand_side(code: str, side: str, sub_mult: float, sub_cat: str, is_ph3: bool) -> dict:
    """
    Decompose ONE phase's train (or val) dataset code into elementary
    workload contributions, keyed by cumulation bucket.

    side: 'train' or 'val'
    is_ph3: whether this phase belongs to a 3-phase combo (changes how a
            'gda' val code unfolds, per the existing pipeline's rule)
    """
    contrib: dict = {}

    def bump(bucket, value):
        contrib[bucket] = contrib.get(bucket, 0.0) + value

    if side == 'train':
        sizes = DS_SIZES_TR
        s_scale = sub_mult                # train subsets ARE scaled
    else:
        sizes = DS_SIZES_VAL
        s_scale = 1.0                     # val is never subset

    # bucket flavor for the synthetic ('s') component only matters on the
    # train side; val always uses the full synthetic val set regardless
    # of what fraction was trained on
    s_cat = sub_cat if side == 'train' else 'pct'

    if code in ('s', 'sub'):
        bump(('s', s_cat), DS_W_MULT['s'] * sizes['s'] * s_scale)

    elif code in ('m', 'subm'):
        # 'm'/'subm' denote a composite dataset: (subset of) synthetic + dk,
        # folded together in a single split -> unfold into both components
        bump(('s', s_cat), DS_W_MULT['s'] * sizes['s'] * s_scale)
        bump(('dk', None), DS_W_MULT['dk'] * sizes['dk'])

    elif code == 'dk':
        bump(('dk', None), DS_W_MULT['dk'] * sizes['dk'])

    elif code == 'gda':
        if side == 'val' and is_ph3:
            # in a 3-phase pipeline, 'gda' on the val side denotes the
            # small supplementary gda_v hold-out, not the base gda val set
            # bump(('gda_v', None), DS_W_MULT['gda'] * DS_SIZES_VAL['gda_v'])
            bump(('gda', None), DS_W_MULT['gda'] * DS_SIZES_VAL['gda_v'])
        else:
            bump(('gda', None), DS_W_MULT['gda'] * sizes['gda'])

    else:
        raise ValueError(f"Unknown dataset code {code!r} on side={side!r}")

    return contrib


# Every possible elementary bucket, fixed and explicit so every phase-row
# gets the same set of (zero-filled) bucket columns regardless of which
# codes that particular phase used.
_TRAIN_BUCKETS = [('s', 'pct'), ('s', 'mix'), ('s', 'composite'), ('dk', None), ('gda', None)]
_VAL_BUCKETS   = [('s', 'pct'), ('dk', None), ('gda', None), ('gda_v', None)]


def _col(side_prefix: str, bucket: tuple) -> str:
    ds, flavor = bucket
    return f"{side_prefix}_{ds}" + (f"_{flavor}" if flavor else "")


_TRAIN_COLS = [_col('t', b) for b in _TRAIN_BUCKETS]
_VAL_COLS   = [_col('v', b) for b in _VAL_BUCKETS]
ALL_BUCKET_COLS = _TRAIN_COLS + _VAL_COLS


def dedupe_phase_rows(df: pd.DataFrame,
                       run_col: str = 'run_id',
                       phase_col: str = 'phase_num',
                       identity_cols: list[str] = None) -> pd.DataFrame:
    """
    Collapse elongated rows back to ONE row per (run_id, phase_num).
 
    Verifies identity_cols (default: ['train', 'val', 'sub']) are actually
    identical across the "duplicate" rows first - raises instead of
    silently dropping real data if that assumption doesn't hold for some
    group.
    """
    identity_cols = identity_cols or ['train', 'val', 'sub']
    dupe_check = df.groupby([run_col, phase_col])[identity_cols].nunique()
    bad = dupe_check[(dupe_check > 1).any(axis=1)]
    if not bad.empty:
        raise ValueError(
            f"{identity_cols} are not identical across the elongated rows "
            f"for these (run, phase) groups - dedup would silently drop "
            f"real data:\n{bad}"
        )
    return df.drop_duplicates(subset=[run_col, phase_col]).reset_index(drop=True)


def add_workload(df: pd.DataFrame,
                 run_col: str = 'run_id',
                 phase_col: str = 'phase_num',
                 train_col: str = 'train',
                 val_col: str = 'val',
                 sub_col: str = 'sub') -> pd.DataFrame:
    """
    df: long format, one row per phase. Must be sortable by (run_col, phase_col).
    Returns df with added bucket columns + a 'workload' column giving the
    TRUE running cumulative total after each phase.
    """
    df_org = df.copy()
    df = dedupe_phase_rows(df, run_col, phase_col)
    df = df.sort_values([run_col, phase_col, 'test set']).copy()

    # total phase count per run - needed for the val=='gda' -> gda_v rule,
    # which depends on the *final* phase count of the run, not just how
    # many phases have happened "so far". If you're computing this while a
    # run is still in progress and don't yet know its final phase count,
    # pass it explicitly via an 'n_phases' column instead of inferring it.
    n_phases = df.groupby([run_col, 'test set'])[phase_col].transform('count')
    # is_ph3 = (n_phases == 3)
    is_ph3 = (df[phase_col] == 3)

    sub_mult, sub_cat = compute_sub_mult_and_category(df[sub_col])

    # expand each phase-row into its elementary contributions, as bucket columns
    bucket_rows = []
    for t_code, v_code, sm, sc, ph3 in zip(df[train_col], df[val_col], sub_mult, sub_cat, is_ph3):
        t_contrib = _expand_side(t_code, 'train', sm, sc, ph3)
        v_contrib = _expand_side(v_code, 'val', sm, sc, ph3)
        row = {c: 0.0 for c in ALL_BUCKET_COLS}
        for bucket, val in t_contrib.items(): # TODO put GDA in the same bucket
            row[_col('t', bucket)] = val
        for bucket, val in v_contrib.items():
            row[_col('v', bucket)] = val
        bucket_rows.append(row)

    bucket_df = pd.DataFrame(bucket_rows, index=df.index)
    df = pd.concat([df, bucket_df], axis=1)

    # running max per bucket WITHIN each run, in phase order -> dedupes
    # reused data and correctly treats nested pct subsets as "already paid for"
    running_max = df.groupby(run_col)[ALL_BUCKET_COLS].cummax()

    # cumulative workload after each phase = sum of buckets "unlocked so far"
    df['Nworkload'] = running_max.sum(axis=1)
    # also expose the per-phase incremental cost (nice for plotting phase-by-phase spend)
    df['workload_delta'] = df.groupby(run_col)['Nworkload'].diff().fillna(df['Nworkload'])

    df = pd.concat([df, df_org[df_org['test set'] != 'SYNT']])

    return df
