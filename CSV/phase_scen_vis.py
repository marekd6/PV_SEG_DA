import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
# import pandas as pd

from desired_plots_tabs_top_lvl import *

TRAIN_WIDTHS = {"SYNT": 15, "DK": 6, "GDA": 3}
VAL_WIDTHS = {"SYNT": 10, "DK": 5, "GDA": 4}

MIX_CODE = {"train": "subm", "val": "m"}
SYNTH_ONLY_CODES = {"s", "sub"}
# maps every known single-category code -> display category, for both train and val
CODE_TO_CATEGORY = {"s": "SYNT", "sub": "SYNT", "dk": "DK", "gda": "GDA"}


def _train_segments(train_code, sub_value):
    """Return [(category, width, label_or_None), ...] for the train bar."""
    sv = sub_value
    ss = sub_value
    if sv in ['mix', 'composite']:
        sv = 9
        ss = '9'
    sv = min(1, float(sv)/100 * 3.75)
    if train_code == MIX_CODE["train"]:
        return [
            ("SYNT", TRAIN_WIDTHS["SYNT"]*sv, ss),
            ("DK", TRAIN_WIDTHS["DK"], None),
        ]
    if train_code in SYNTH_ONLY_CODES:
        return [("SYNT", TRAIN_WIDTHS["SYNT"]*sv, ss)]
    # pass-through single category (e.g. 'dk', 'gda') -- no label, 'sub' doesn't apply
    category = CODE_TO_CATEGORY.get(train_code, train_code.upper())
    return [(category, TRAIN_WIDTHS.get(category, 1), None)]


def _val_segments(val_code):
    """Return [(category, width, label_or_None), ...] for the val bar. No labels on val."""
    if val_code == MIX_CODE["val"]:
        return [
            ("SYNT", VAL_WIDTHS["SYNT"], None),
            ("DK", VAL_WIDTHS["DK"], None),
        ]
    category = CODE_TO_CATEGORY.get(val_code, val_code.upper())
    return [(category, VAL_WIDTHS.get(category, 6), None)]


def _with_starts(segments):
    """Attach cumulative start positions to a list of (category, width, label)."""
    out, cursor = [], 0
    for category, width, label in segments:
        out.append((cursor, width, category, label))
        cursor += width
    return out


def transform(raw_df):
    """
    raw_df columns: entry_id, phase, train_data, val_data, test_set, iou, sub
    9 rows per run (3 phases x 3 test_sets); train_data/val_data/sub are
    constant across the 3 test_set rows within a (entry_id, phase) group.

    Returns:
      bar_df   -> run, phase, bar_name (Train/Val), start, width, category, label
      table_df -> run, phase, <one column per test_set> (IoU), single row per (run, phase)
    """
    bar_rows, table_rows = [], []

    for (entry_id, phase), group in raw_df.groupby(["entry_id", "phase"], sort=False):
        first = group.iloc[0]  # composition is constant within the group
        train_segs = _with_starts(_train_segments(first["train"], first["sub"]))
        val_segs = _with_starts(_val_segments(first["val"]))

        for start, width, category, label in train_segs:
            bar_rows.append({"run": entry_id, "phase": phase, "bar_name": "Train",
                              "start": start, "width": width, "category": category, "label": label})
        for start, width, category, label in val_segs:
            bar_rows.append({"run": entry_id, "phase": phase, "bar_name": "Val",
                              "start": start, "width": width, "category": category, "label": label})

        # pivot the 3 test_set/iou rows into one wide row for the table
        row = {"run": entry_id, "phase": phase, 'bs': first['batch_size'], 'epochs': first['epochs'],
               'lr enc': first['lrenc'], 'lr dec': first['lrdec'], 'wd': first['wd'], 'Workload': first['Workload'], 'Runtime': first['Runtime']}
        for _, r in group.iterrows():
            if r['test set'] == 'GDA' or True:
                row[r["test set"]] = r["IoU"]
        table_rows.append(row)

    return pd.DataFrame(bar_rows), pd.DataFrame(table_rows)

category_colors = {"DK": "#4C72B0", "GDA": "#DD8452", "SYNT": "#55A868"}

def get_segments(bar_df, run, phase, bar_name):
    sub = bar_df[(bar_df.run == run) & (bar_df.phase == phase) & (bar_df.bar_name == bar_name)]
    return list(sub.sort_values("start")[["start", "width", "category", 'label']].itertuples(index=False, name=None))

def get_table_rows(table_df, run, phase):
    """Split the run's params into two (cols, values) tuples: hyperparameters
    and metrics (test-set IoUs), so they can be rendered as two stacked
    header-row tables instead of one."""
    sub = table_df[(table_df.run == run) & (table_df.phase == phase)]
    param_cols = [c for c in table_df.columns if c not in ("run", "phase")]
    tv = sub[param_cols].values[0]
    hparam_ks, hparam_vs = [], []
    metric_ks, metric_vs = [], []
    for k, v in zip(param_cols, tv):
        if k in ['DK', 'GDA', 'SYNT'] + ['Runtime', 'Workload']:
            # kk = 'SYNTHETIC' if k == 'SYNT' else k
            kk = k
            metric_ks.append(kk)
            if kk == 'Runtime':
                metric_vs.append(f"{v:.0f}")
            else:
                metric_vs.append(f"{v:.3f}")
        else:
            vv = f'{v:.0e}' if ('lr' in k or k == 'wd') else f'{v:.0f}'
            hparam_ks.append(k)
            hparam_vs.append(vv)
    return (hparam_ks, [hparam_vs]), (metric_ks, [metric_vs])

# bar/table split WITHIN a phase - kept tight, distinct from the larger
# gap BETWEEN phases (set on the outer GridSpec, see below)
INNER_WSPACE = 0.12
BAR_COL_RATIO = 1.1     # single combined Train/Val bar chart (was 2 columns of 0.55 each)
TABLE_COL_RATIO = 5.2


BAR_ROW_ORDER = ["Val", "Train"]  # bottom-to-top; Train ends up drawn above Val

def draw_phase_block(fig, phase_cell, run, phase, bar_df, table_df, xlim_max,
                      label_min_width=3, bar_thickness=0.6, row_pad=0.5,
                      table_hspace=0.15):
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=phase_cell,
        width_ratios=[BAR_COL_RATIO, TABLE_COL_RATIO],
        wspace=INNER_WSPACE,
    )

    # one horizontal chart per phase, Train and Val as two rows sharing
    # a common x-axis (data scale), instead of two separate bar axes
    ax = fig.add_subplot(inner_gs[0, 0])
    for row, bar_name in enumerate(BAR_ROW_ORDER):
        y0 = row - bar_thickness / 2
        for start, width, category, label in get_segments(bar_df, run, phase, bar_name):
            ax.broken_barh([(start, width)], (y0, bar_thickness),
                           facecolors=category_colors[category], edgecolor="white")
            if pd.notna(label) and width > label_min_width:
                ax.text(start + width / 2, row, f'{label}%',
                        ha="center", va="center", fontsize=8, color="white")
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(-row_pad, len(BAR_ROW_ORDER) - 1 + row_pad)
    ax.set_yticks(range(len(BAR_ROW_ORDER)))
    ax.set_yticklabels(BAR_ROW_ORDER, fontsize=8)
    ax.set_xticks([])
    for s in ["top", "right", "bottom", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

    # two header-row tables stacked vertically in the second cell, each
    # sized to exactly half the cell (bbox=[0,0,1,1] forces a fit rather
    # than letting table.scale() overflow the allotted height)
    table_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=inner_gs[0, 1], hspace=table_hspace,
    )
    ax_hparams = fig.add_subplot(table_gs[0, 0]); ax_hparams.axis("off")
    ax_metrics = fig.add_subplot(table_gs[1, 0]); ax_metrics.axis("off")

    (hparam_cols, hparam_vals), (metric_cols, metric_vals) = get_table_rows(table_df, run, phase)

    t1 = ax_hparams.table(cellText=hparam_vals, colLabels=hparam_cols,
                           loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
    t1.auto_set_font_size(False); t1.set_fontsize(8)

    t2 = ax_metrics.table(cellText=metric_vals, colLabels=metric_cols,
                           loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
    t2.auto_set_font_size(False); t2.set_fontsize(8)

# spacing BETWEEN phases - kept larger than the inner train/val/table
# spacing (INNER_WSPACE, set inside draw_phase_block) so phases read as
# visually distinct groups
OUTER_WSPACE = 0.2
# spacing between the label column and the first phase - independent of
# OUTER_WSPACE since it's controlled on a separate, outer GridSpec
LABEL_WSPACE = 0.05

def _fmt_group_val(v):
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def extract_full_entries(df: pd.DataFrame, level: str = 'total SYNT use'):
    """
    For each level, find the entry_id whose phase=3 and test_set='g' row
    has the highest metric, then return ALL rows (all phases) belonging
    to that entry_id.
    """

    # Step 1: filter to the decisive slice
    filtered = df[(df['phase'] == 3) & (df['test set'] == 'GDA')] # TODO or among all phases - v. interseting

    # Step 2: find the best entry_id per level
    idx = filtered.groupby(level)['IoU'].idxmax()
    winners = filtered.loc[idx, [level, 'entry_id']]

    # Step 3: join back to original df to get full long-format entries
    result = df.merge(winners, on=[level, 'entry_id'], how='inner')

    # Optional: sort nicely
    return result.sort_values([level, 'entry_id', 'phase']).reset_index(drop=True)


def build_phase_scenario_grid(group_var='total SYNT use',
                               entry_col='entry_id', title=None,
                               save_name='top_scens_by_synth.png'):
    """
    Rank distinct entries by `group_var` (descending) and render the
    phase x (train/val/tables) grid for the top `top_n` of them.

    First column shows, per row, "<group_var value> (<entry_id>)", with
    `group_var` itself as the column header.
    """
    cols = HPARAM_COLS_BASE_CAT + ['IoU', 'phase', 'test set', entry_col, group_var, 'epochs'] + ['Runtime', 'Workload']
    ph = total_df_treatment(FILES['joint'], joint=True)
    cols = list(set(cols) & set(ph.columns))
    print(cols, 'cols')
    ph = ph[cols]
    ph = extract_full_entries(ph, group_var)

    if not SAVING:
        print(ph)

    bar_df, table_df = transform(ph)
    if not SAVING:
        print(bar_df)
        print(table_df)
        print(table_df.dtypes)

    records = bar_df["run"].unique().tolist()
    phases = bar_df["phase"].unique().tolist()
    row_labels = {run: f"{_fmt_group_val(ph[ph[entry_col] == run][group_var].iloc[0])} ({run})" for run in records}

    # one shared x-scale for every bar chart in the grid, so a given defined
    # width (e.g. DK=12) is always drawn at the same physical size, instead
    # of each phase-block rescaling to its own local max
    xlim_max = (bar_df.groupby(['run', 'phase', 'bar_name'])['width'].sum().max())

    n_records, n_phases = len(records), len(phases)
    fig = plt.figure(figsize=(11.7, 8.3))

    # root: 2 columns (label | phases-block), spacing between them controlled
    # independently from the spacing BETWEEN phases (set below, on phases_gs)
    root_gs = gridspec.GridSpec(
        n_records + 1, 2,
        width_ratios=[0.1, n_phases],   # label column narrow, phases block wide
        height_ratios=[0.15] + [1] * n_records,
        hspace=0.25,
        wspace=LABEL_WSPACE,
        left=0.01, right=0.98,
        top=0.95, bottom=0.08,
    )
    # nested: phases-block column split into n_phases columns - same row
    # structure (height_ratios/hspace) as root_gs so rows line up with the
    # label column, but its OWN wspace between phases
    phases_gs = gridspec.GridSpecFromSubplotSpec(
        n_records + 1, n_phases, subplot_spec=root_gs[:, 1],
        width_ratios=[1] * n_phases,
        height_ratios=[0.15] + [1] * n_records,
        hspace=0.25,
        wspace=OUTER_WSPACE,
    )

    # first-column header names what the row values below represent
    ax_col_label = fig.add_subplot(root_gs[0, 0]); ax_col_label.axis("off")
    ax_col_label.text(0.9, 0.9, group_var, fontweight="bold", fontsize=7,
                       ha="center", va="center") # , rotation=90)
    ax_col_label.text(0.9, 0.2, '(run)', fontweight="bold", fontsize=7,
                       ha="center", va="center") # , rotation=90)

    for p, phase in enumerate(phases):
        ax = fig.add_subplot(phases_gs[0, p]); ax.axis("off")
        ax.text(0.5, 0.2, f'phase = {phase}', fontweight="bold", ha="center", va="center")

    for r, run in enumerate(records):
        row_idx = r + 1
        ax_label = fig.add_subplot(root_gs[row_idx, 0]); ax_label.axis("off")
        ax_label.text(0.5, 0.5, row_labels[run], fontweight="bold", fontsize=8,
                      ha="center", va="center", rotation=90)
        for p, phase in enumerate(phases):
            draw_phase_block(fig, phases_gs[row_idx, p], run, phase, bar_df, table_df, xlim_max)

    legend_handles = [Patch(facecolor=c, label=cat) for cat, c in category_colors.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(category_colors),
               frameon=False, fontsize=10)
    fig.suptitle(title or f'Top scenarios by {group_var}')

    if SAVING:
        fig.savefig(f'{SAVEDIR}/{save_name}')
        plt.close(fig)
    else:
        plt.show()
    return fig


build_phase_scenario_grid()
