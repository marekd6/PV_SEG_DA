import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
# import pandas as pd

from desired_plots_tabs_top_lvl import *

TRAIN_WIDTHS = {"Synthetic": 20, "DK": 12, "GDA": 8}
VAL_WIDTHS = {"Synthetic": 12, "DK": 7, "GDA": 5}

MIX_CODE = {"train": "subm", "val": "m"}
SYNTH_ONLY_CODES = {"s", "sub"}
# maps every known single-category code -> display category, for both train and val
CODE_TO_CATEGORY = {"s": "Synthetic", "sub": "Synthetic", "dk": "DK", "gda": "GDA"}


def _train_segments(train_code, sub_value):
    """Return [(category, width, label_or_None), ...] for the train bar."""
    if train_code == MIX_CODE["train"]:
        return [
            ("Synthetic", TRAIN_WIDTHS["Synthetic"], str(sub_value)),
            ("DK", TRAIN_WIDTHS["DK"], None),
        ]
    if train_code in SYNTH_ONLY_CODES:
        return [("Synthetic", TRAIN_WIDTHS["Synthetic"], str(sub_value))]
    # pass-through single category (e.g. 'dk', 'gda') -- no label, 'sub' doesn't apply
    category = CODE_TO_CATEGORY.get(train_code, train_code.upper())
    return [(category, TRAIN_WIDTHS.get(category, 10), None)]


def _val_segments(val_code):
    """Return [(category, width, label_or_None), ...] for the val bar. No labels on val."""
    if val_code == MIX_CODE["val"]:
        return [
            ("Synthetic", VAL_WIDTHS["Synthetic"], None),
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
        row = {"run": entry_id, "phase": phase}
        for _, r in group.iterrows():
            row[r["test set"]] = r["IoU"]
        table_rows.append(row)

    return pd.DataFrame(bar_rows), pd.DataFrame(table_rows)

cols = ['IoU', 'phase', 'test set', 'sub', 'train', 'val', 'entry_id']
ph123 = total_df_treatment(FILES['joint'], joint=True)
ph123 = ph123[cols]
ph123 = ph123.sort_values(by=['entry_id', 'phase'])
ph123 = ph123.head(9*3)
print(ph123)

# # Long-format inputs
# bar_df = pd.DataFrame([
#     # run, phase, bar_name, category, start, width
#     ("Run A", "Phase 1", "Train", "synthetic", 0, 20),
#     ("Run A", "Phase 1", "Train", "mixed", 20, 5),
#     ("Run A", "Phase 1", "Val", "gda", 0, 5),
#     ("Run A", "Phase 2", "Train", "synthetic", 0, 20),
#     ("Run A", "Phase 2", "Train", "mixed", 20, 5),
#     ("Run A", "Phase 2", "Val", "mixed", 0, 15),
#     ("Run A", "Phase 3", "Train", "synthetic", 0, 20),
#     ("Run A", "Phase 3", "Train", "gda", 20, 5),
#     ("Run A", "Phase 3", "Val", "gda", 0, 5),

#     ("Run B", "Phase 1", "Train", "synthetic", 0, 20),
#     ("Run B", "Phase 1", "Train", "mixed", 20, 5),
#     ("Run B", "Phase 1", "Val", "gda", 0, 5),
#     ("Run B", "Phase 2", "Train", "synthetic", 0, 20),
#     ("Run B", "Phase 2", "Train", "mixed", 20, 5),
#     ("Run B", "Phase 2", "Val", "mixed", 0, 15),
#     ("Run B", "Phase 2", "Val", "gda", 15, 10),
#     ("Run B", "Phase 3", "Train", "synthetic", 0, 20),
#     ("Run B", "Phase 3", "Train", "gda", 20, 5),
#     ("Run B", "Phase 3", "Val", "gda", 0, 5),
# ], columns=["run", "phase", "bar_name", "category", "start", "width"])
# # run, phase, tr/val type, DS part, start, width

# table_df = pd.DataFrame([
#     # run, phase, <param columns...>
#     ("Run A", "Phase 1", "3e-4", "32", "20"),
#     ("Run A", "Phase 2", "3e-4", "32", "20"),
#     ("Run A", "Phase 3", "3e-4", "32", "20"),

#     ("Run B", "Phase 1", "3e-4", "32", "20"),
#     ("Run B", "Phase 2", "3e-4", "32", "20"),
#     ("Run B", "Phase 3", "3e-4", "32", "20"),
# ], columns=["run", "phase", "lr", "batch", "epochs"])
# # run, phase, params1,2,3, IoUa,b,c

bar_df, table_df = transform(ph123)
print(bar_df)
print(table_df)
print(table_df.dtypes)

category_colors = {"Synthetic": "#4C72B0", "DK": "#DD8452", "GDA": "#55A868"}
records = bar_df["run"].unique().tolist()
phases = bar_df["phase"].unique().tolist()

def get_segments(run, phase, bar_name):
    sub = bar_df[(bar_df.run == run) & (bar_df.phase == phase) & (bar_df.bar_name == bar_name)]
    return list(sub.sort_values("start")[["start", "width", "category", 'label']].itertuples(index=False, name=None))

def get_table_row(run, phase):
    sub = table_df[(table_df.run == run) & (table_df.phase == phase)]
    param_cols = [c for c in table_df.columns if c not in ("run", "phase")]
    # return sub[param_cols].values.tolist(), param_cols
    return [[f"{x:.3f}" for x in row] for row in sub[param_cols].values.tolist()], param_cols

def draw_phase_block(fig, gs, row_idx, col_start, run, phase, label_min_width=3, bar_height=2):
    for b, bar_name in enumerate(["Train", "Val"]):
        ax = fig.add_subplot(gs[row_idx, col_start + b])
        bar_width = 0
        for start, width, category, label in get_segments(run, phase, bar_name):
            bar_width += width
            ax.broken_barh([(start, width)], (0, bar_height),
                           facecolors=category_colors[category], edgecolor="white")
            if width > label_min_width:
                ax.text(start + width / 2, bar_height / 2, label,
                        ha="center", va="center", fontsize=8, color="white")
        ax.set_ylim(0, bar_height)
        ax.set_yticks([])
        ax.set_yticklabels('')
        ax.set_xticks([bar_width / 2])
        ax.set_xticklabels([bar_name], fontsize=8)
        for s in ["top", "right", "bottom", "left"]: ax.spines[s].set_visible(False)
        ax.tick_params(left=False)

    ax_table = fig.add_subplot(gs[row_idx, col_start + 2])
    ax_table.axis("off")
    row_values, param_cols = get_table_row(run, phase)
    print(row_values, param_cols)
    table = ax_table.table(cellText=row_values, colLabels=param_cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)

n_records, n_phases = len(records), len(phases)
# fig = plt.figure(figsize=(5 * n_phases + 1.5, 2.0 * n_records + 1.5))
fig = plt.figure()
gs = gridspec.GridSpec(
    n_records + 1, 1 + n_phases * 3,
    # width_ratios=[0.2] + [1.6, 1.6, 1.4] * n_phases,
    width_ratios=[0.2] + [1, 1, 1] * n_phases,
    height_ratios=[1] + [1] * n_records,
    hspace=0.1, wspace=0.7,
)

for p, phase in enumerate(phases):
    col_start = 1 + p * 3
    ax = fig.add_subplot(gs[0, col_start:col_start + 3]); #ax.axis("off")
    ax.text(0.5, 0.2, f'phase = {phase}', fontsize=13, fontweight="bold", ha="center", va="center")

for r, run in enumerate(records):
    row_idx = r + 1
    ax_label = fig.add_subplot(gs[row_idx, 0]); #ax_label.axis("off")
    ax_label.text(0.5, 0.5, run, fontsize=11, fontweight="bold",
                  ha="center", va="center", rotation=90)
    for p, phase in enumerate(phases):
        draw_phase_block(fig, gs, row_idx, 1 + p * 3, run, phase)

legend_handles = [Patch(facecolor=c, label=cat) for cat, c in category_colors.items()]
fig.legend(handles=legend_handles, loc="lower center", ncol=len(category_colors),
           frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)
# fig.suptitle("Per-phase training summary", fontsize=15, y=0.99)
# plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.show()
