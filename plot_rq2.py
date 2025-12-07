#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stacked L1-norm boxplots:

Top row  : SDP
Bottom   : JIT-SDP

Reads per-model L1 CSVs from:
  - ./SDP/evaluations/norms/{MODEL}_L1.csv
  - ./JIT-SDP/evaluations/norms/{MODEL}_L1.csv

and produces:
  - ./figures_rq2_L1/AllModels_L1_stacked.png
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
# -------------------------------------------------------
# Config
# -------------------------------------------------------
# Bigger y-axis tick labels (left columns)
mpl.rcParams['ytick.labelsize'] = 18

# Where the L1 CSVs live
EVAL_ROOT = Path("./SDP")
JIT_ROOT = Path("./JIT-SDP")

# Per-repo L1 CSV location
L1_REL_PATH = Path("evaluations/norms")

# Models (by abbreviation used in the CSVs)
MODEL_ORDER = ["RF", "SVM", "XGB", "CatB", "LGBM"]

# Explainer orders per dataset (display names as stored in CSVs)
EXPLAINER_ORDER_EVAL = ["DeFlip", "SQAPlanner", "TimeLIME", "LIME-HPO", "LIME"]
EXPLAINER_ORDER_JIT  = ["DeFlip", "CfExplainer", "PyExplainer", "LIME-HPO", "LIME"]


# -------------------------------------------------------
# Helpers: load L1 CSVs and reconstruct l1_scores / succ_counts
# -------------------------------------------------------
def load_flip_rate_table(root: Path) -> Dict[str, Dict[str, float]]:
    """
    Load ./evaluations/flip_rates.csv under the given root and return:
        table[model_abbr][explainer_name] = flip_rate  (fraction in [0,1])

    Assumes the CSV has columns: Model, Explainer, Flip Rate
    and that Explainer names match those used in the L1 CSVs.
    """
    csv_path = root / "evaluations" / "flip_rates.csv"
    if not csv_path.exists():
        print(f"[WARN] flip_rates file not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    required_cols = {"Model", "Explainer", "Flip Rate"}
    if not required_cols.issubset(df.columns):
        print(f"[WARN] flip_rates file missing columns in {csv_path}")
        return {}

    # Build nested dict: model -> explainer -> flip_rate
    table: Dict[str, Dict[str, float]] = {}
    for (model, expl), sub in df.groupby(["Model", "Explainer"]):
        val = float(sub["Flip Rate"].iloc[0])

        # Map internal token "CF" to display name "DeFlip"
        if expl == "CF":
            disp_name = "DeFlip"
        else:
            disp_name = expl

        table.setdefault(model, {})[disp_name] = val

    return table


def load_l1_for_dataset(root: Path) -> Dict[str, Tuple[Dict[str, List[float]], Dict[str, int]]]:
    """
    For a given dataset root (eval or jit), load per-model L1 CSVs and return:

      result[model_abbr] = (l1_scores, succ_counts)

    where:
      l1_scores[explainer] -> list of L1 scores
      succ_counts[explainer] -> count of scores
    """
    out: Dict[str, Tuple[Dict[str, List[float]], Dict[str, int]]] = {}

    base_dir = root / L1_REL_PATH
    if not base_dir.exists():
        print(f"[WARN] L1 directory does not exist: {base_dir}")
        return out

    for model_abbr in MODEL_ORDER:
        csv_path = base_dir / f"{model_abbr}_L1.csv"
        if not csv_path.exists():
            # it's okay if some models are missing
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Expect columns: Model, Explainer, Project, TestIdx, Score
        if "Explainer" not in df.columns or "Score" not in df.columns:
            print(f"[WARN] Missing columns in {csv_path}; skipping.")
            continue

        l1_scores: Dict[str, List[float]] = {}
        succ_counts: Dict[str, int] = {}

        for expl, subdf in df.groupby("Explainer"):
            vals = subdf["Score"].dropna().astype(float).tolist()
            if not vals:
                continue
            l1_scores[expl] = vals
            succ_counts[expl] = len(vals)

        if l1_scores:
            out[model_abbr] = (l1_scores, succ_counts)

    return out


# -------------------------------------------------------
# Single-axis plotting (reuse your style)
# -------------------------------------------------------
def plot_norms_on_axis(
    ax: plt.Axes,
    model_abbr: str,
    l1_scores: Dict[str, List[float]],
    succ_counts: Dict[str, int],
    explainer_order: List[str],
    y_cap: float | None = None,
    bottom_margin_frac: float = 0.10,
    flip_rates: Dict[str, float] | None = None,
    # NEW: optional superscripts for this axis only
    median_sup: str | None = None,
    fr_sup: str | None = None,
):
    """
    Draw the L1 boxplot for a single model on the given axis.

    Style:
      - DeFlip box dark gray; others light gray
      - median line white for DeFlip, black for others
      - median value annotated on the line
      - x-ticks: explainer names
      - flip rate (FR) in italics in the strip below y=0
    """
    labels = [lab for lab in explainer_order if lab in l1_scores and l1_scores[lab]]
    if not labels:
        ax.set_visible(False)
        return

    data_l1 = [l1_scores[lab] for lab in labels]
    n_counts = [succ_counts[lab] for lab in labels]  # kept if you later want n=

    bplot = ax.boxplot(
        data_l1,
        patch_artist=True,
        notch=True,
        vert=True,
        widths=0.9,
        showfliers=False,
    )

    # fill colors
    fill_colors = []
    for lab in labels:
        if lab == "DeFlip":
            fill_colors.append("#555555")   # dark for DeFlip
        else:
            fill_colors.append("#E0E0E0")   # light gray for others

    for patch, color in zip(bplot["boxes"], fill_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
        patch.set_alpha(1.0)

    # median color: white for DeFlip, black otherwise
    for i, median in enumerate(bplot["medians"]):
        color = "white" if labels[i] == "DeFlip" else "black"
        median.set(color=color, linewidth=1.5)

    # ---------- annotate medians inside the boxes ----------
    # ---------- annotate medians inside the boxes ----------
    for i, vals in enumerate(data_l1):
        x = i + 1
        med_y = float(np.median(vals))
        label_txt = f"{med_y:.2f}"

        # superscript only if this axis is marked AND this explainer is DeFlip
        if median_sup and labels[i] == "LIME":
            # bold superscript a
            label_txt = rf"{med_y:.2f}$^{{\mathbf{{{median_sup}}}}}$"

        ax.text(
            x,
            med_y,
            label_txt,
            ha="center",
            va="center",
            fontsize=15,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0),
            zorder=5,
        )

    # ---------- y-limits + small bottom margin ----------
    # start from data-driven limits
    y_min, y_max = ax.get_ylim()

    # apply optional cap
    if y_cap is not None:
        y_max = y_cap
    # we assume L1 >= 0, so bottom is effectively 0
    base_bottom = 0.0

    # add a small margin below 0: 10% of positive span
    span = max(y_max - base_bottom, 1e-6)
    extra = bottom_margin_frac * span
    new_y_min = base_bottom - extra

    ax.set_ylim(new_y_min, y_max)

    # y for FR label: halfway between bottom and 0
    label_y = new_y_min / 2.0

    # ---------- annotate flip rate (FR) in italics in that bottom strip ----------
    # ---------- annotate flip rate (FR) in italics in that bottom strip ----------
    for i, lab in enumerate(labels):
        x = i + 1
        fr_text = "FR=NA"
        if flip_rates is not None and lab in flip_rates:
            fr = flip_rates[lab]          # e.g., 0.37
            fr_text = f"{fr*100:.0f}%"    # e.g., 37%

        # superscript only if this axis is marked AND this explainer is DeFlip
        if fr_sup and lab == "LIME":
            fr_text = rf"{fr_text}$^{{\mathbf{{{fr_sup}}}}}$"

        ax.text(
            x,
            label_y,
            fr_text,
            ha="center",
            va="center",
            fontsize=13,
            fontstyle="italic",
            color="black",
        )

    # ---------- x tick labels: explainer names only ----------
    new_labels = [lab for lab in labels]
    ax.set_xticklabels(
        new_labels,
        fontsize=15,
        fontweight="normal",
        color="black",
        rotation=30,
    )
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    # title
    ax.set_title(model_abbr, fontsize=23, pad=10)

    ax.yaxis.grid(True, linestyle="-", which="major", color="#cccccc", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------------------------------------
# Combined stacked figure
# -------------------------------------------------------
def main():
    # load eval and jit data
    eval_results = load_l1_for_dataset(EVAL_ROOT)
    jit_results  = load_l1_for_dataset(JIT_ROOT)

    # load flip rate tables (per model, per explainer)
    eval_flip_table = load_flip_rate_table(EVAL_ROOT)
    jit_flip_table  = load_flip_rate_table(JIT_ROOT)

    # keep models that exist in at least one dataset
    models_present = [
        m for m in MODEL_ORDER if (m in eval_results) or (m in jit_results)
    ]
    if not models_present:
        print("[WARN] No models with L1 data in either dataset.")
        return

    n_models = len(models_present)

    # -------------------------------------------------
    # Figure layout: 2 rows × (n_models + 1) columns
    # last column is skinny and used only for vertical titles
    # -------------------------------------------------
    fig, axes = plt.subplots(
        2,
        n_models + 1,
        figsize=(4.5 * n_models + 2.0, 10),   # total width & height
        sharey=False,
        gridspec_kw={"width_ratios": [1.0] * n_models + [0.18]},
    )

    if n_models == 1:
        axes = axes.reshape(2, 2)

    # left block are the real plots; rightmost column for titles
    plot_axes = axes[:, :n_models]
    title_axes = axes[:, -1]

    # ------------------- top row: Eval -------------------
    # ------------------- top row: Eval -------------------
    for col, model_abbr in enumerate(models_present):
        ax = plot_axes[0, col]
        if model_abbr in eval_results:
            l1_scores, succ_counts = eval_results[model_abbr]
            flip_rates_for_model = eval_flip_table.get(model_abbr, {})

            median_sup = 'a' if col == 0 else None
            fr_sup     = 'b' if col == 0 else None

            plot_norms_on_axis(
                ax,
                model_abbr,
                l1_scores,
                succ_counts,
                EXPLAINER_ORDER_EVAL,
                y_cap=25.0,
                flip_rates=flip_rates_for_model,
                median_sup=median_sup,
                fr_sup=fr_sup,
            )

            # NEW: explanation text in the top-left plot
            if col == 0:
                ax.text(
                    0.02, 0.78,
                    rf"$^{{\mathbf{{a}}}}$: median" +"\n" +rf"$^{{\mathbf{{b}}}}$: flip rate (within whole range)",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=13,
                )
        else:
            ax.set_visible(False)

    # ------------------- bottom row: JIT -------------------
    for col, model_abbr in enumerate(models_present):
        ax = plot_axes[1, col]
        if model_abbr in jit_results:
            l1_scores, succ_counts = jit_results[model_abbr]
            flip_rates_for_model = jit_flip_table.get(model_abbr, {})
            plot_norms_on_axis(
                ax,
                model_abbr,
                l1_scores,
                succ_counts,
                EXPLAINER_ORDER_JIT,
                y_cap=12.5,                  # CHANGED: JIT cap 20 -> 15
                flip_rates=flip_rates_for_model,
            )
            ax.set_yticks([0, 2.5, 5, 7.5, 10, 12.5])    # match new cap
        else:
            ax.set_visible(False)

    # remove titles from second row (JIT)
    for ax in plot_axes[1, :]:
        ax.set_title("")

    # shared y-label (figure-level)
    fig.text(
        0.05, 0.5,
        "Magnitude of Change",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        rotation=90,
    )

    # ------------------- vertical row titles (right side) -------------------
    row_titles = ["SDP", "JIT-SDP"]
    for ax_title, label in zip(title_axes, row_titles):
        ax_title.set_axis_off()
        ax_title.text(
            -0.25,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=23,
            fontweight="bold",
            rotation=270,
        )

    # ------------------- small legend-style note for superscripts ----------
    # Place a compact note somewhere unobtrusive on the figure.
    # fig.text(
    #     0.82, 0.06,
    #     "a: median    b: flip rate (within whole range)",
    #     ha="left",
    #     va="center",
    #     fontsize=13,
    #     bbox=dict(facecolor="white", edgecolor="black", alpha=0.9, pad=0.3),
    # )

    # hide y tick labels for all but the first column in each row
    for row in range(2):
        for col in range(1, n_models):
            ax = plot_axes[row, col]
            ax.tick_params(labelleft=False)

    # ------------------- layout / spacing -------------------
    fig.subplots_adjust(
        top=0.92,
        bottom=0.15,
        left=0.08,
        right=0.99,   # plots + title column nearly fill width
        hspace=0.3,   # vertical gap between the two rows
        wspace=0.12,  # horizontal gap between model plots
    )

    out_dir = Path("./figures_rq2_L1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rq2.png"

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"[plot] Saved stacked L1 figure -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()