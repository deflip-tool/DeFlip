#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
# =========================================================
# COST / MARKERS / SIZES
# =========================================================

# ----- JIT-SDP (bottom row) -----
COST_AND_OFFSETS_JIT = {
    "CF":         ("DeFlip",     0.00, (0.04, 0)),   # our method
    "LIME-HPO":   ("LIME-HPO",   0.95, (0.03, 0)),
    "LIME":       ("LIME",       0.90, (0.03, -4)),
    "PyExplainer":   ("PyExplainer",   0.40, (0.03, 0)),
    "CfExplainer": ("CfExplainer", 0.85, (0.03, 0)),
}

# ----- SDP (top row) -----
COST_AND_OFFSETS_ACT = {
    "CF":         ("DeFlip",     0.00, (0.04, 0)),   # our method
    "LIME-HPO":   ("LIME-HPO",   0.95, (0.03, 0)),
    "LIME":       ("LIME",       0.90, (0.03, -4)),
    "TimeLIME":   ("TimeLIME",   0.40, (0.03, 0)),
    "SQAPlanner": ("SQAPlanner", 0.85, (0.03, 0)),
}

# global marker shapes by *display* name
MARKERS_BY_TOOL = {
    "DeFlip":     "o",
    "LIME":       "^",
    "LIME-HPO":   "s",
    "TimeLIME":   "D",
    "SQAPlanner": "P",
    "PyExplainer": "X",   # NEW shape
    "CfExplainer": "*",   # NEW shape
}

# (minimal_size, full_size) per explainer (by display name)
MARKER_SIZES = {
    "DeFlip":     (20, 20),
    "LIME":       (25, 25),
    "LIME-HPO":   (25, 25),
    "TimeLIME":   (18, 18),
    "SQAPlanner": (40, 40),
    "PyExplainer": (40, 40),
    "CfExplainer": (70, 70),
}

# order for columns
DESIRED_MODEL_ORDER = ["RF", "SVM", "XGB", "CatB", "LGBM"]

# order for legend entries
EXPLAINER_ORDER = [
    "DeFlip", "LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "PyExplainer", "CfExplainer"
]

# =========================================================
# helpers: loading & experiments
# =========================================================

def load_flip_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Explainer" not in df.columns or "Model" not in df.columns:
        raise ValueError(f"flip_rates file missing required columns: {path}")
    if "All" in df["Explainer"].values:
        df = df[df["Explainer"] != "All"].copy()
    return df


def build_experiments(start_df: pd.DataFrame,
                      end_df: pd.DataFrame,
                      cost_and_offsets: dict) -> dict:
    """
    Build:
        experiments[model_name][display_name] = (start_pct, end_pct, cost, offset)
    """
    start_pivot = start_df.pivot(index="Model", columns="Explainer", values="Flip Rate")
    end_pivot = end_df.pivot(index="Model", columns="Explainer", values="Flip Rate")

    experiments = {}
    common_models = sorted(set(start_pivot.index) & set(end_pivot.index))

    for model_name in common_models:
        model_dict = {}
        for expl_token, (disp_name, cost, offset) in cost_and_offsets.items():
            if expl_token not in start_pivot.columns or expl_token not in end_pivot.columns:
                print(f"[WARN] Missing explainer={expl_token} for model={model_name}")
                continue

            start_val = start_pivot.loc[model_name, expl_token]
            end_val = end_pivot.loc[model_name, expl_token]

            # if pd.isna(start_val) or pd.isna(end_val):
            #     print(f"[WARN] NaN flip rate for {model_name}, {expl_token}")
            #     continue

            start_pct = float(start_val) * 100.0
            end_pct = float(end_val) * 100.0
            model_dict[disp_name] = (start_pct, end_pct, cost, offset)

        if model_dict:
            experiments[model_name] = model_dict

    return experiments

# =========================================================
# axis background
# =========================================================

def _setup_axis_background(ax):
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.3, zorder=0)
    ax.axvline(0, color='black', linewidth=1, alpha=0.2, zorder=0)
    for v in [0.25, 0.5, 0.75]:
        ax.axvline(v, color='black', linestyle=':', linewidth=0.8, alpha=0.3, zorder=0)

    LEFT_PAD = 0.3
    RIGHT_PAD = 0.07
    ax.set_xlim(-LEFT_PAD, 1.0 + RIGHT_PAD)
    ax.set_ylim(0, 115)

    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0.0\n", "0.5", "1.0\n"], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)

# =========================================================
# per-row plotters
# =========================================================

def _plot_single_axis_jit(ax, model_name, data):
    """JIT-SDP row (bottom)."""
    _setup_axis_background(ax)

    EDGE_COLOR = 'black'
    START_FILL = 'white'
    FULL_FILL = "0.5"

    for tool_name, val in data.items():   # tool_name is display name
        start_rate, end_rate, cost, offset = val
        marker = MARKERS_BY_TOOL.get(tool_name, "o")
        size_start, size_end = MARKER_SIZES.get(tool_name, (40, 24))

        # DeFlip: only full gray marker with border
        if tool_name == "DeFlip":
            ax.scatter(
                cost, end_rate,
                s=size_end,
                marker=marker,
                facecolors=FULL_FILL,
                edgecolors=EDGE_COLOR,
                linewidth=1.0,
                zorder=7,
            )
            ax.text(
                cost,
                end_rate + 4.0,
                f"{end_rate:.0f}",
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
            )
            continue

        # minimal impl at x=0
        ax.scatter(
            0, start_rate,
            s=size_start,
            marker=marker,
            facecolors=START_FILL,
            edgecolors=EDGE_COLOR,
            linewidth=1.0,
            zorder=6,
        )

        start_label = f"{start_rate:.0f}"

        # horiz pos for minimal labels
        if tool_name in ("LIME", "CfExplainer"):
            min_x = -0.06
            min_ha = "right"
        else:
            min_x = 0.07
            min_ha = "left"

        # vertical offsets to avoid overlaps (your custom logic)
        if tool_name == "LIME-HPO" and model_name in ("LGBM", "RF", "XGB"):
            min_y = start_rate - 1.5
        elif tool_name == "PyExplainer":
            min_y = start_rate + 1.5
        else:
            min_y = start_rate

        ax.text(
            min_x,
            min_y,
            start_label,
            fontsize=7.0,
            ha=min_ha,
            va="center",
            color="black",
        )

        # full impl at cost
        ax.scatter(
            cost, end_rate,
            s=size_end,
            marker=marker,
            facecolors=FULL_FILL,
            edgecolors="none",
            linewidth=0.0,
            zorder=7,
        )

        end_label = f"{end_rate:.0f}"
        if tool_name == "LIME-HPO":
            ax.text(
                cost, end_rate + 4.0,
                end_label,
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
            )
        else:
            ax.text(
                cost, end_rate - 5.0,
                end_label,
                fontsize=7.0,
                ha="center",
                va="top",
                color="black",
            )

    ax.set_title(f"{model_name}", fontsize=9.5, pad=6)


def _plot_single_axis_act(ax, model_name, data):
    """SDP row (top)."""
    _setup_axis_background(ax)

    EDGE_COLOR = 'black'
    START_FILL = 'white'
    FULL_FILL = "0.5"

    for tool_name, val in data.items():
        start_rate, end_rate, cost, offset = val
        marker = MARKERS_BY_TOOL.get(tool_name, "o")
        size_start, size_end = MARKER_SIZES.get(tool_name, (40, 24))

        # DeFlip: only full gray marker with border
        if tool_name == "DeFlip":
            ax.scatter(
                cost, end_rate,
                s=size_end,
                marker=marker,
                facecolors=FULL_FILL,
                edgecolors=EDGE_COLOR,
                linewidth=1.0,
                zorder=7,
            )
            ax.text(
                cost,
                end_rate + 4.0,
                f"{end_rate:.0f}",
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
            )
            continue

        # minimal impl at x=0
        ax.scatter(
            0, start_rate,
            s=size_start,
            marker=marker,
            facecolors=START_FILL,
            edgecolors=EDGE_COLOR,
            linewidth=1.0,
            zorder=6,
        )

        start_label = f"{start_rate:.0f}"

        # horiz pos for minimal labels (your original rules)
        if tool_name in ("LIME", "SQAPlanner"):
            min_x = -0.06
            min_ha = "right"
        else:
            min_x = 0.07
            min_ha = "left"

        ax.text(
            min_x,
            start_rate,
            start_label,
            fontsize=7.0,
            ha=min_ha,
            va="center",
            color="black",
        )

        # full impl at cost
        ax.scatter(
            cost, end_rate,
            s=size_end,
            marker=marker,
            facecolors=FULL_FILL,
            edgecolors="none",
            linewidth=0.0,
            zorder=7,
        )

        end_label = f"{end_rate:.0f}"
        if tool_name == "LIME-HPO":
            ax.text(
                cost, end_rate + 4.0,
                end_label,
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
            )
        else:
            ax.text(
                cost, end_rate - 5.0,
                end_label,
                fontsize=7.0,
                ha="center",
                va="top",
                color="black",
            )

    ax.set_title(f"{model_name}", fontsize=9.5, pad=6)

# =========================================================
# legends
# =========================================================

def _build_legend_handles(names, full: bool):
    handles = []
    for expl_name in names:
        if expl_name not in MARKERS_BY_TOOL:
            continue
        marker = MARKERS_BY_TOOL[expl_name]
        size_start, size_end = MARKER_SIZES.get(expl_name, (40, 24))
        ms = (size_end if full else size_start) ** 0.5

        if full:
            mface = "0.5"
            medge = "none"
            medgewidth = 0.0
        else:
            mface = "white"
            medge = "black"
            medgewidth = 1.0

        handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                color="w",
                markerfacecolor=mface,
                markeredgecolor=medge,
                markeredgewidth=medgewidth,
                markersize=ms,
                label=expl_name,
            )
        )
    return handles

# =========================================================
# combined plot
# =========================================================

def plot_combined(experiments_act: dict,
                  experiments_jit: dict,
                  save_path: Path) -> None:
    # union of all models
    all_models = [
        m for m in DESIRED_MODEL_ORDER
        if (m in experiments_act) or (m in experiments_jit)
    ]
    n_models = len(all_models)
    if n_models == 0:
        print("[WARN] No models to plot.")
        return

    # --------- FIGURE SIZE (overall width/height) ----------
    fig, axes = plt.subplots(
        2, n_models,
        figsize=(10, 5.3),   # <--- change 4.0 to shrink/grow vertically
        sharey=True,
    )
    if n_models == 1:
        axes = axes.reshape(2, 1)

    # ---------------- top row: SDP ----------------
    for col, model_name in enumerate(all_models):
        ax_top = axes[0, col]
        if model_name in experiments_act:
            _plot_single_axis_act(ax_top, model_name, experiments_act[model_name])
        else:
            ax_top.axis("off")

    # ---------------- bottom row: JIT-SDP ----------------
    for col, model_name in enumerate(all_models):
        ax_bot = axes[1, col]
        if model_name in experiments_jit:
            _plot_single_axis_jit(ax_bot, model_name, experiments_jit[model_name])
        else:
            ax_bot.axis("off")
        ax_bot.set_title("")

    # y-label on left
    axes[0, 0].set_ylabel(
        "Predicted Defect Flip Rate (%)",
        fontsize=9.5,
        fontweight="bold",
        labelpad=10,
    )
    axes[0, 0].yaxis.set_label_coords(-0.25, -0.1)  # move left/right, up/down

    # --------- GLOBAL X-LABEL (near very bottom) ----------
    fig.text(
        0.5, 0.27,  # <--- change y to move this label up/down
        "Normalized Exploration Depth (0=Start, 1=Limit)",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )

    # --------- ROW TITLES ON THE RIGHT, VERTICAL ----------
    # x is near the right margin; rotation=270 makes text vertical
    fig.text(
        0.91, 0.805,                    # <--- change y to move top title up/down
        "SDP",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        rotation=270,                  # <--- vertical title
    )

    fig.text(
        0.91, 0.455,                    # <--- change y for bottom title
        "JIT-SDP",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        rotation=270,                  # <--- vertical title
    )

    # ---------- TWO LEGENDS AT THE BOTTOM ----------
    minimal_handles = _build_legend_handles(EXPLAINER_ORDER, full=False)
    full_handles = _build_legend_handles(EXPLAINER_ORDER, full=True)

    # legend for minimal (hollow markers)
    fig.legend(
        handles=minimal_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.16),
        #              ^^^^ ^^^^
        #              x    y  -> tweak y to move legend up/down
        frameon=True,
        fontsize=8,
        borderpad=0.25,
        # edgecolor="black",
        ncol=len(minimal_handles),
        title="Immediate Suggestion",
        title_fontsize=8.5,
        labelspacing=0.3,
    )

    # legend for full (filled markers), just below
    fig.legend(
        handles=full_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),   # <--- y for second legend
        frameon=True,
        fontsize=8,
        borderpad=0.25,
        # edgecolor="black",
        ncol=len(full_handles),
        title="Validated Solution (at Flip Depth)",
        title_fontsize=8.5,
        labelspacing=0.3,
    )

    # --------- SUBPLOT LAYOUT (space allocated for plots) ----------
    fig.subplots_adjust(
        top=0.93,   # <--- decrease to give more space above plots
        bottom=0.33,  # <--- increase to give more space to legends/x-label
        left=0.08,
        right=0.90,  # <--- decrease to leave room on the right for vertical titles
        hspace=0.25,  # <--- vertical gap between rows
        wspace=0.18,  # <--- horizontal gap between model panels
    )

    fig.savefig(save_path, dpi=1200, bbox_inches="tight")
    print(f"Graph saved: {save_path}")
    plt.close(fig)

# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    root_act = Path("./SDP")
    root_jit = Path("./JIT-SDP")

    act_closest_path = root_act / "evaluations_closest" / "flip_rates.csv"
    act_full_path = root_act / "evaluations" / "flip_rates.csv"

    jit_closest_path = root_jit / "evaluations_closest" / "flip_rates.csv"
    jit_full_path = root_jit / "evaluations" / "flip_rates.csv"

    for p in [act_closest_path, act_full_path, jit_closest_path, jit_full_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    act_start_df = load_flip_rates(act_closest_path)
    act_end_df = load_flip_rates(act_full_path)
    experiments_act = build_experiments(act_start_df, act_end_df, COST_AND_OFFSETS_ACT)

    jit_start_df = load_flip_rates(jit_closest_path)
    jit_end_df = load_flip_rates(jit_full_path)
    experiments_jit = build_experiments(jit_start_df, jit_end_df, COST_AND_OFFSETS_JIT)

    out_dir = Path("./figures_rq1_plans")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "figure_rq1_all_models_stacked.png"

    plot_combined(experiments_act, experiments_jit, save_path)