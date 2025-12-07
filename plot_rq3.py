#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined RQ3 Mahalanobis KDE plots (using cached flip distances).

Top row   : SDP (SDP; delta-based history)
Bottom row: JIT-SDP (JIT-SDP; raw-train history)

Assumptions:
  * You already ran EACH repo's RQ3 script, which produced:
        SDP/evaluations/rq3_mahalanobis/{MODEL_ABBR}_{Expl}.csv
        JIT-SDP/evaluations/rq3_mahalanobis/{MODEL_ABBR}_{Expl}.csv
    where each CSV has a 'distance' column.

This script:
  * recomputes ONLY the "Actual History" distances (not saved before),
  * loads explainer distances from the cached CSVs,
  * computes Mann–Whitney + Cliff's delta:
        - each explainer vs DeFlip
        - each explainer vs Actual History
  * plots a single 2×N figure (N = number of models):
        top row    = SDP (eval)
        bottom row = JIT-SDP (jit)
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde, mannwhitneyu
from cliffs_delta import cliffs_delta
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde, mannwhitneyu
from cliffs_delta import cliffs_delta
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # NEW
# ---------------------------------------------------------------------
# Global font
# ---------------------------------------------------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]

# ---------------------------------------------------------------------
# Paths to the two repos
# ---------------------------------------------------------------------

ROOT_EVAL = Path("./SDP")
ROOT_JIT = Path("./JIT-SDP")

RQ3_DIR_EVAL = ROOT_EVAL / "evaluations" / "rq3_mahalanobis"
RQ3_DIR_JIT = ROOT_JIT / "evaluations" / "rq3_mahalanobis"

# ---------------------------------------------------------------------
# Basic mappings
# ---------------------------------------------------------------------

MODEL_ABBR = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

# Explainer → display name (EVAL)
EXPLAINER_NAME_MAP_EVAL = {
    "LIME": "LIME",
    "LIME-HPO": "LIME-HPO",
    "TimeLIME": "TimeLIME",
    "SQAPlanner_confidence": "SQAPlanner",
    "CF": "DeFlip",
}

# Explainer → display name (JIT)
EXPLAINER_NAME_MAP_JIT = {
    "LIME": "LIME",
    "LIME-HPO": "LIME-HPO",
    "PyExplainer": "PyExplainer",
    "CfExplainer": "CfExplainer",
    "CF": "DeFlip",
}

# Lists of explainer *tokens* whose cached CSVs we will load
EVAL_EXPLAINERS = list(EXPLAINER_NAME_MAP_EVAL.keys())
JIT_EXPLAINERS = list(EXPLAINER_NAME_MAP_JIT.keys())

PLOT_ORDER_EVAL = [
    "Actual History",
    "SQAPlanner",
    "DeFlip",
    "TimeLIME",
    "LIME",
    "LIME-HPO",
]

PLOT_ORDER_JIT = [
    "Actual History",
    "CfExplainer",
    "DeFlip",
    "PyExplainer",
    "LIME",
    "LIME-HPO",
]

# Order for legends & inset bars (no Actual History here)
LEGEND_INSET_ORDER_EVAL = ["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner", "DeFlip"]
LEGEND_INSET_ORDER_JIT  = ["LIME", "LIME-HPO", "PyExplainer", "CfExplainer", "DeFlip"]

# ---------------------------------------------------------------------
# Imports from each repo
# ---------------------------------------------------------------------

def import_eval_data_utils():
    sys.path.insert(0, str(ROOT_EVAL))
    import data_utils as eval_data_utils  # type: ignore
    sys.path.pop(0)
    return eval_data_utils


def import_jit_data_utils():
    sys.path.insert(0, str(ROOT_JIT))
    import data_utils as jit_data_utils  # type: ignore
    sys.path.pop(0)
    return jit_data_utils


# ---------------------------------------------------------------------
# History distributions (recomputed only for Actual History)
# ---------------------------------------------------------------------

def build_history_distribution_eval(ds, project_list):
    """
    Original eval RQ3 semantics:
      - history = TRAIN→TEST deltas on overlapping indices.
    """
    feat_union = set()
    for project in project_list:
        if project not in ds:
            continue
        train, test = ds[project]
        feat_union |= {c for c in train.columns if c != "target"}
        feat_union |= {c for c in test.columns if c != "target"}

    if not feat_union:
        raise RuntimeError("[Eval/RQ3] No features across projects.")

    feat_union = sorted(feat_union)
    total_deltas_list = []

    for project in project_list:
        if project not in ds:
            continue
        train, test = ds[project]
        common_idx = train.index.intersection(test.index)
        if len(common_idx) == 0:
            continue

        t_cols = [c for c in test.columns if c != "target"]
        tr_cols = [c for c in train.columns if c != "target"]
        common_feats = sorted(set(t_cols) & set(tr_cols))
        if not common_feats:
            continue

        deltas = test.loc[common_idx, common_feats] - train.loc[common_idx, common_feats]
        deltas = deltas.reindex(columns=feat_union, fill_value=0.0)
        total_deltas_list.append(deltas)

    if not total_deltas_list:
        raise RuntimeError("[Eval/RQ3] No deltas built.")

    total_deltas = pd.concat(total_deltas_list, axis=0)
    total_deltas = total_deltas.loc[:, total_deltas.nunique() > 1]
    feat_cols = list(total_deltas.columns)

    X = total_deltas.values.astype(float)
    mean_vec = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    print(f"[Eval/RQ3] History delta matrix: rows={X.shape[0]}, cols={X.shape[1]}")
    return total_deltas, feat_cols, mean_vec, inv_cov


def build_history_distribution_jit(ds, project_list):
    """
    JIT semantics:
      - history = raw TRAIN snapshots (commit-level metrics).
    """
    history_list = []

    for project in project_list:
        if project not in ds:
            continue

        train, _ = ds[project]
        feat_cols = [c for c in train.columns if c != "target"]

        if len(train) == 0 or not feat_cols:
            print(f"[JIT/RQ3] {project}: skipped (empty train/features).")
            continue

        feats = train[feat_cols].copy()
        history_list.append(feats)

    if not history_list:
        raise RuntimeError("[JIT/RQ3] No TRAIN snapshots for history.")

    total_hist = pd.concat(history_list, axis=0, ignore_index=True)
    total_hist = total_hist.replace([np.inf, -np.inf], np.nan)
    total_hist = total_hist.dropna(axis=1, how="all")
    if total_hist.empty:
        raise RuntimeError("[JIT/RQ3] All history columns NaN/inf.")

    keep_cols = []
    for col in total_hist.columns:
        vals = total_hist[col].to_numpy(dtype=float)
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() < 5:
            continue
        if np.var(vals[finite_mask]) <= 1e-12:
            continue
        keep_cols.append(col)

    total_hist = total_hist[keep_cols]
    if total_hist.empty:
        raise RuntimeError("[JIT/RQ3] All features dropped as low-support/constant.")

    for col in total_hist.columns:
        vals = total_hist[col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        fill_val = np.median(finite_vals) if finite_vals.size > 0 else 0.0
        total_hist[col] = np.where(np.isfinite(vals), vals, fill_val)

    X = total_hist.to_numpy(dtype=float)
    cov = np.cov(X, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    d = cov.shape[0]
    cov = cov + 1e-6 * np.eye(d, dtype=cov.dtype)

    try:
        inv_cov = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        diag = np.diag(cov).copy()
        diag[~np.isfinite(diag)] = 1.0
        diag[diag <= 0] = 1.0
        inv_cov = np.diag(1.0 / diag)

    mean_vec = X.mean(axis=0)
    feat_cols = list(total_hist.columns)

    print(f"[JIT/RQ3] History matrix: rows={X.shape[0]}, cols={X.shape[1]}")
    return total_hist, feat_cols, mean_vec, inv_cov


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def parse_models(arg: str) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return ["RandomForest", "SVM", "XGBoost", "CatBoost", "LightGBM"]
    return [m.strip() for m in arg.replace(",", " ").split() if m.strip()]


def parse_projects(arg: str, ds_keys: List[str]) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return list(sorted(ds_keys))
    return [p.strip() for p in arg.replace(",", " ").split() if p.strip()]


def get_overlap_score(kde1, kde2, x_grid: np.ndarray) -> float:
    y1 = kde1(x_grid)
    y2 = kde2(x_grid)
    return float(np.trapz(np.minimum(y1, y2), x_grid))


def _compute_auto_limits(
    dist_dicts: Dict[str, Dict[str, np.ndarray]],
    fallback_min: float,
    fallback_max: float,
) -> Tuple[float, float]:
    """
    Auto min/max across all models & explainers for a dataset row.
    Only considers finite, non-negative distances.
    """
    mins, maxs = [], []
    for _abbr, expl_dict in dist_dicts.items():
        for _name, vals in expl_dict.items():
            arr = np.asarray(vals, float)
            arr = arr[np.isfinite(arr) & (arr >= 0)]
            if arr.size == 0:
                continue
            mins.append(float(arr.min()))
            maxs.append(float(arr.max()))
    if not mins or not maxs:
        return fallback_min, fallback_max

    x_min = max(0.0, min(mins))
    x_max = max(maxs)
    if x_max <= x_min:
        x_max = x_min + 1.0

    pad = 0.05 * (x_max - x_min)
    return max(0.0, x_min - pad), x_max + pad


def _find_csv_path(dirpath: Path, model_abbr: str, disp_name: str) -> Optional[Path]:
    """
    Be permissive about file naming: try '{abbr}_{name}.csv' first,
    then '{abbr}{name}.csv' as a fallback.
    """
    p1 = dirpath / f"{model_abbr}_{disp_name}.csv"
    if p1.exists() and p1.stat().st_size > 0:
        return p1
    p2 = dirpath / f"{model_abbr}{disp_name}.csv"
    if p2.exists() and p2.stat().st_size > 0:
        return p2
    return None


# ---------------------------------------------------------------------
# Stats for RQ3 (Mahalanobis) – vs DeFlip & vs Actual History
# ---------------------------------------------------------------------

def compute_rq3_stats_for_model(
    model_abbr: str,
    dist_dict: Dict[str, np.ndarray],
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Compute unpaired Mann–Whitney U and Cliff's delta for Mahalanobis distances.

    Baselines:
      - 'DeFlip'         (compare each explainer vs DeFlip)
      - 'Actual History' (compare each explainer vs Actual History)

    dist_dict must contain:
      - key 'Actual History'  -> np.ndarray of distances
      - optionally keys 'DeFlip', 'LIME', 'LIME-HPO', 'PyExplainer', 'CfExplainer', ...
    """
    rows = []

    # Clean everything
    cleaned: Dict[str, np.ndarray] = {}
    for name, arr in dist_dict.items():
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            cleaned[name] = vals

    def _one_baseline(baseline: str):
        if baseline not in cleaned or cleaned[baseline].size < min_n:
            print(f"[RQ3/stats] {model_abbr}: baseline '{baseline}' has too few points.")
            return

        base_vals = cleaned[baseline]
        for expl, other_vals in cleaned.items():
            if expl == baseline:
                continue
            # For DeFlip baseline, skip comparison to Actual History
            if baseline == "DeFlip" and expl == "Actual History":
                continue

            n_b = base_vals.size
            n_o = other_vals.size
            if n_b < min_n or n_o < min_n:
                continue

            _, p_val = mannwhitneyu(other_vals, base_vals, alternative="two-sided")
            cd, cd_mag = cliffs_delta(other_vals.tolist(), base_vals.tolist())

            rows.append(
                [
                    model_abbr,
                    "Mahalanobis",
                    baseline,
                    expl,
                    int(n_b),
                    int(n_o),
                    float(p_val),
                    float(cd),
                    cd_mag,
                ]
            )

    _one_baseline("DeFlip")
    _one_baseline("Actual History")

    df = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Metric",
            "Baseline",
            "Explainer",
            "N_baseline",
            "N_other",
            "p_value",
            "cliffs_delta",
            "cd_magnitude",
        ],
    )

    if not df.empty:
        df["p_value"] = df["p_value"].round(4)
        df["cliffs_delta"] = df["cliffs_delta"].round(3)

    return df


# ---------------------------------------------------------------------
# Load cached distances (no recomputation of flips)
# ---------------------------------------------------------------------

def load_cached_distances_eval(
    model_types: List[str],
    hist_vals: np.ndarray,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Build:
        model_abbr -> { "Actual History": hist_vals, expl1: arr, ... }
    and stats DataFrame across all models.
    """
    dist_dicts: Dict[str, Dict[str, np.ndarray]] = {}
    stats_rows: List[pd.DataFrame] = []

    for model_type in model_types:
        model_abbr = MODEL_ABBR.get(model_type, model_type)
        print(f"[Eval/RQ3] Loading cached distances for model {model_abbr}")

        dist_dict: Dict[str, np.ndarray] = {
            "Actual History": hist_vals
        }

        for expl_token in EVAL_EXPLAINERS:
            disp_name = EXPLAINER_NAME_MAP_EVAL.get(expl_token, expl_token)
            csv_path = _find_csv_path(RQ3_DIR_EVAL, model_abbr, disp_name)
            if csv_path is None:
                print(f"[Eval/RQ3]   - Missing CSV for {disp_name} ({model_abbr}); skipping.")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[Eval/RQ3]   - Failed to read {csv_path}: {e}")
                continue

            if "distance" not in df.columns:
                print(f"[Eval/RQ3]   - No 'distance' column in {csv_path}; skipping.")
                continue

            vals = df["distance"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                print(f"[Eval/RQ3]   - Too few distances for {disp_name} ({model_abbr}); n={vals.size}")
                continue

            dist_dict[disp_name] = vals
            print(f"[Eval/RQ3]   - Loaded {disp_name}: n={vals.size}")

        if len(dist_dict) > 1:
            dist_dicts[model_abbr] = dist_dict
            stats_df_model = compute_rq3_stats_for_model(model_abbr, dist_dict, min_n=5)
            if not stats_df_model.empty:
                stats_rows.append(stats_df_model)
        else:
            print(f"[Eval/RQ3] Not enough data to plot for model {model_abbr} (eval).")

    if stats_rows:
        stats_all = pd.concat(stats_rows, axis=0, ignore_index=True)
    else:
        stats_all = pd.DataFrame(
            columns=[
                "Model",
                "Metric",
                "Baseline",
                "Explainer",
                "N_baseline",
                "N_other",
                "p_value",
                "cliffs_delta",
                "cd_magnitude",
            ]
        )

    return dist_dicts, stats_all


def load_cached_distances_jit(
    model_types: List[str],
    hist_vals: np.ndarray,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Build:
        model_abbr -> { "Actual History": hist_vals, expl1: arr, ... }
    and stats DataFrame across all models.
    """
    dist_dicts: Dict[str, Dict[str, np.ndarray]] = {}
    stats_rows: List[pd.DataFrame] = []

    for model_type in model_types:
        model_abbr = MODEL_ABBR.get(model_type, model_type)
        print(f"[JIT/RQ3] Loading cached distances for model {model_abbr}")

        dist_dict: Dict[str, np.ndarray] = {
            "Actual History": hist_vals
        }

        for expl_token in JIT_EXPLAINERS:
            disp_name = EXPLAINER_NAME_MAP_JIT.get(expl_token, expl_token)
            csv_path = _find_csv_path(RQ3_DIR_JIT, model_abbr, disp_name)
            if csv_path is None:
                print(f"[JIT/RQ3]   - Missing CSV for {disp_name} ({model_abbr}); skipping.")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[JIT/RQ3]   - Failed to read {csv_path}: {e}")
                continue

            if "distance" not in df.columns:
                print(f"[JIT/RQ3]   - No 'distance' column in {csv_path}; skipping.")
                continue

            vals = df["distance"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                print(f"[JIT/RQ3]   - Too few distances for {disp_name} ({model_abbr}); n={vals.size}")
                continue

            dist_dict[disp_name] = vals
            print(f"[JIT/RQ3]   - Loaded {disp_name}: n={vals.size}")

        if len(dist_dict) > 1:
            dist_dicts[model_abbr] = dist_dict
            stats_df_model = compute_rq3_stats_for_model(model_abbr, dist_dict, min_n=5)
            if not stats_df_model.empty:
                stats_rows.append(stats_df_model)
        else:
            print(f"[JIT/RQ3] Not enough data to plot for model {model_abbr} (jit).")

    if stats_rows:
        stats_all = pd.concat(stats_rows, axis=0, ignore_index=True)
    else:
        stats_all = pd.DataFrame(
            columns=[
                "Model",
                "Metric",
                "Baseline",
                "Explainer",
                "N_baseline",
                "N_other",
                "p_value",
                "cliffs_delta",
                "cd_magnitude",
            ]
        )

    return dist_dicts, stats_all


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_kde_on_axis(
    ax: plt.Axes,
    dist_data_dict: Dict[str, np.ndarray],
    model_abbr: str,
    dataset: str,
    x_min: float,
    x_max: float,
    overlap_records: List[Dict],
    show_legend: bool,
    show_model_label: bool,
    show_history_label: bool,
    show_overlap_box: bool,
):
    """
    dataset: "eval" or "jit"

    - Per-axis x-range [x_min, x_max]
    - Local y-max (per subplot)
    - Returns (handles, labels) only for the axis where show_legend=True,
      so the caller can build a single legend per row at figure level.
    """
    if dataset == "eval":
        plot_order = PLOT_ORDER_EVAL
        styles = {
            "Actual History": {"c": "#cccccc", "ls": "-",  "lw": 1.0, "z": 1},
            "SQAPlanner":     {"c": "#999999", "ls": "--", "lw": 1.5, "z": 3},
            "DeFlip":         {"c": "black",   "ls": "-",  "lw": 2.5, "z": 5},
            "TimeLIME":       {"c": "#666666", "ls": "-.", "lw": 1.5, "z": 2},
            "LIME":           {"c": "#555555", "ls": ":",  "lw": 1.5, "z": 2},
            "LIME-HPO":       {"c": "#999999", "ls": "-.", "lw": 1.5, "z": 2},
        }
        dataset_label = "SDP"
        legend_order = LEGEND_INSET_ORDER_EVAL
    else:
        plot_order = PLOT_ORDER_JIT
        styles = {
            "Actual History": {"c": "#cccccc", "ls": "-",  "lw": 1.0, "z": 1},
            "CfExplainer":    {"c": "#999999", "ls": "--", "lw": 1.5, "z": 3},
            "DeFlip":         {"c": "black",   "ls": "-",  "lw": 2.5, "z": 5},
            "PyExplainer":    {"c": "#666666", "ls": "-.", "lw": 1.5, "z": 2},
            "LIME":           {"c": "#555555", "ls": ":",  "lw": 1.5, "z": 2},
            "LIME-HPO":       {"c": "#999999", "ls": "-.", "lw": 1.5, "z": 2},
        }
        dataset_label = "JIT-SDP"
        legend_order = LEGEND_INSET_ORDER_JIT

    x_grid = np.linspace(x_min, x_max, 1000)

    # ---------------- History ----------------
    hist_vals = np.asarray(dist_data_dict["Actual History"], float)
    hist_vals = hist_vals[np.isfinite(hist_vals) & (hist_vals >= 0)]
    if hist_vals.size < 2:
        print(f"[{dataset.upper()}/RQ3] Not enough Actual History points for {model_abbr}.")
        return [], []

    kde_history = gaussian_kde(hist_vals)
    y_hist = kde_history(x_grid)

    # local y-max initialised with history peak
    y_max_local = float(y_hist.max())

    # Plot history (filled)
    st_hist = styles["Actual History"]
    ax.fill_between(x_grid, y_hist, color=st_hist["c"], alpha=0.4, zorder=st_hist["z"])
    ax.plot(x_grid, y_hist, color="#aaaaaa", lw=1.0, zorder=st_hist["z"])

    # Peak label – ONLY if requested (first row, first column)
    if show_history_label:
        peak_idx = int(np.argmax(y_hist))
        ax.text(
            x_grid[peak_idx],
            y_hist[peak_idx] * 1.02,
            "Actual History\n(ref)",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
        )

    # ---------- storage for legend + inset ----------
    handle_by_name: Dict[str, plt.Line2D] = {}
    overlaps_local: Dict[str, float] = {}

    # ---------------- Explainers ----------------
    for name in plot_order:
        if name == "Actual History":
            continue
        if name not in dist_data_dict:
            continue

        data = np.asarray(dist_data_dict[name], float)
        data = data[np.isfinite(data) & (data >= 0)]
        if data.size < 2:
            continue

        kde = gaussian_kde(data)
        y_vals = kde(x_grid)
        st = styles[name]

        # track y-max
        y_max_local = max(y_max_local, float(y_vals.max()))

        line_handle, = ax.plot(
            x_grid,
            y_vals,
            color=st["c"],
            linestyle=st["ls"],
            linewidth=st["lw"],
            zorder=st["z"],
        )

        handle_by_name[name] = line_handle

        # Overlap: store globally + locally
        ov = get_overlap_score(kde, kde_history, x_grid)
        overlap_records.append(
            {"Dataset": dataset_label, "Model": model_abbr, "Explainer": name, "Overlap": ov}
        )
        overlaps_local[name] = ov

    # ---------------- Axes styling ----------------
    ax.set_xlim(x_min, x_max)

    if dataset == "jit":
        ax.set_ylim(bottom=0.0, top=0.6)
    else:
        ax.set_ylim(bottom=0.0, top=y_max_local * 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Model label – ONLY on top row
    if show_model_label:
        ax.text(
            0.5,
            1.02,
            model_abbr,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # ---------------- Overlap inset bar plot (horizontal, DeFlip filled) ----------------
    # ---------------- Overlap inset bar plot (horizontal, DeFlip filled) ----------------
    # from matplotlib.transforms import Bbox

    # draw barh as before
    if show_overlap_box and overlaps_local:
        # bbox = Bbox.from_extents(0.60, 0.40, 0.95, 0.80)  # (x0, y0, x1, y1)

        inset_ax = inset_axes(
            ax,
            width="100%",          # relative to parent axes
            height="100%",
            # loc="center right",
            bbox_to_anchor=(0.63, 0.45, 0.33, 0.38),
            bbox_transform=ax.transAxes,
            loc="lower left",
            borderpad=0.0,
        )

        # REVERSED order for inset vs legend
        expl_names = [n for n in reversed(legend_order) if n in overlaps_local]
        overlaps = [overlaps_local[n] for n in expl_names]

        y_pos = np.arange(len(expl_names))
        bars = inset_ax.barh(
            y_pos,
            overlaps,
            edgecolor="black",
            facecolor="none",
            linewidth=0.8,
        )

        # Highlight DeFlip bar only
        for bar, name in zip(bars, expl_names):
            if name == "DeFlip":
                bar.set_facecolor("0.2")
                bar.set_edgecolor("black")
            else:
                bar.set_facecolor("none")
                bar.set_edgecolor("black")

        inset_ax.set_yticks(y_pos)
        inset_ax.set_yticklabels(expl_names, fontsize=6)
        inset_ax.set_xlim(0.0, 1.0)

        # ✅ Only show x-label for the *upper-left* inset (where show_history_label is True)
        # (and only for SDP row if you want)
        if show_history_label and dataset == "eval":
            inset_ax.set_xlabel("Overlap [0-1]", fontsize=6, labelpad=1)

        # Remove x ticks & labels everywhere
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])

        inset_ax.tick_params(axis="y", pad=1)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.8)

    # ---------------- Return handles/labels for a single row legend ----------------
    if show_legend:
        handles, labels = [], []
        for n in legend_order:
            if n in handle_by_name:
                handles.append(handle_by_name[n])
                labels.append(n)
        return handles, labels

    return [], []

def plot_combined_horizontal(
    eval_dist: Dict[str, Dict[str, np.ndarray]],
    jit_dist: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
    x_limits_eval: Tuple[float, float],
    x_limits_jit: Tuple[float, float],
) -> pd.DataFrame:
    """
    Plot 2×N grid and return a DataFrame with overlap values.

    Legends:
      - One row-level legend centered at top for SDP (top row)
      - One row-level legend centered mid-figure for JIT-SDP (bottom row)
    """
    model_abbrs = [
        abbr for abbr in ["RF", "SVM", "XGB", "CatB", "LGBM"]
        if abbr in eval_dist or abbr in jit_dist
    ]
    n_models = len(model_abbrs)
    if n_models == 0:
        print("[Plot] No models to plot.")
        return pd.DataFrame(columns=["Dataset", "Model", "Explainer", "Overlap"])

    fig, axes = plt.subplots(2, n_models, figsize=(3.0 * n_models, 5.5), sharey=False)
    if n_models == 1:
        axes = axes.reshape(2, 1)

    overlap_records: List[Dict] = []

    # these will hold the handles/labels used for row-level legends
    eval_handles, eval_labels = [], []
    jit_handles, jit_labels = [], []

    # ---------------- Top row = SDP (eval) ----------------
    for col, abbr in enumerate(model_abbrs):
        ax = axes[0, col]
        if abbr in eval_dist:
            show_legend        = (col == 0)   # only first subplot returns handles/labels
            show_overlap_box   = True    # others: inset bar
            show_model_label   = True         # model names only on top row
            show_history_label = (col == 0)   # Actual History label only first col

            handles, labels = plot_kde_on_axis(
                ax,
                eval_dist[abbr],
                abbr,
                dataset="eval",
                x_min=x_limits_eval[0],
                x_max=x_limits_eval[1],
                overlap_records=overlap_records,
                show_legend=show_legend,
                show_model_label=show_model_label,
                show_history_label=show_history_label,
                show_overlap_box=show_overlap_box,
            )
            if show_legend and handles:
                eval_handles, eval_labels = handles, labels
        else:
            ax.set_visible(False)

    # ---------------- Bottom row = JIT-SDP ----------------
    for col, abbr in enumerate(model_abbrs):
        ax = axes[1, col]
        if abbr in jit_dist:
            show_legend        = (col == 0)
            show_overlap_box   = True
            show_model_label   = False
            show_history_label = False

            handles, labels = plot_kde_on_axis(
                ax,
                jit_dist[abbr],
                abbr,
                dataset="jit",
                x_min=x_limits_jit[0],
                x_max=x_limits_jit[1],
                overlap_records=overlap_records,
                show_legend=show_legend,
                show_model_label=show_model_label,
                show_history_label=show_history_label,
                show_overlap_box=show_overlap_box,
            )
            if show_legend and handles:
                jit_handles, jit_labels = handles, labels
        else:
            ax.set_visible(False)

    # Only show y tick labels for the first plot of each row
    for row in range(2):
        for col in range(1, n_models):
            axes[row, col].tick_params(labelleft=False)

    # Remove per-axis xlabels; use a single global label instead
    for col in range(n_models):
        axes[0, col].set_xlabel("")
        axes[1, col].set_xlabel("")

    # Shared Y
    fig.text(
        0.04,
        0.5,
        "Probability Density",
        va="center",
        ha="left",
        rotation="vertical",
        fontsize=15,
        fontweight="bold",
    )

    # Single global X
    fig.text(
        0.5,
        0.12,
        "Mahalanobis Distance",
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )

    # Row labels on right
    fig.text(0.99, 0.75, "SDP",     ha="right", va="center", fontsize=15, fontweight="bold", rotation=270)
    fig.text(0.99, 0.35, "JIT-SDP", ha="right", va="center", fontsize=15, fontweight="bold", rotation=270)

    # -------- Row-level legends (center top of each row) --------
    # top row legend
    if eval_handles:
        fig.legend(
            eval_handles,
            eval_labels,
            loc="upper center",
            bbox_to_anchor=(0.505, 0.99),   # near top of figure
            ncol=len(eval_labels),
            frameon=True,
            framealpha=1.0,
            # edgecolor="black",
            fontsize=8,
            handlelength=1.6,
            handletextpad=1.5,
            labelspacing=0.4,
        )

    # bottom row legend (roughly mid-figure)
    if jit_handles:
        fig.legend(
            jit_handles,
            jit_labels,
            loc="upper center",
            bbox_to_anchor=(0.505, 0.535),   # adjust this up/down as you like
            ncol=len(jit_labels),
            frameon=True,
            framealpha=1.0,
            # edgecolor="black",
            fontsize=8,
            handlelength=1.6,
            handletextpad=1.5,
            labelspacing=0.4,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        top=0.90,    # slightly lower to leave room for top legend
        bottom=0.20,
        hspace=0.62,
        wspace=0.25,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"[Plot] Saved combined RQ3 figure -> {save_path}")
    plt.close(fig)

    return pd.DataFrame(overlap_records)


def save_overlap_table(overlap_df: pd.DataFrame, out_root: Path):
    if overlap_df.empty:
        print("[Overlap] No overlap data to save.")
        return

    overlap_df = overlap_df.sort_values(["Dataset", "Model", "Explainer"]).reset_index(drop=True)

    csv_path = out_root / "rq3_mahalanobis_overlap_table.csv"
    overlap_df.to_csv(csv_path, index=False)
    print(f"[Overlap] Saved overlap table CSV -> {csv_path}")

    n_rows = len(overlap_df)
    fig_height = max(1.6, 0.28 * n_rows)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=overlap_df.round({"Overlap": 2}).values,
        colLabels=overlap_df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    png_path = out_root / "rq3_mahalanobis_overlap_table.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Overlap] Saved overlap table figure -> {png_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default="RandomForest, SVM, XGBoost, CatBoost, LightGBM",
        help='Models spaced/comma (e.g., "RandomForest,SVM") or "all"',
    )
    ap.add_argument("--projects_eval", type=str, default="all", help='Eval projects spaced/comma or "all"')
    ap.add_argument("--projects_jit",  type=str, default="all", help='JIT projects spaced/comma or "all"')
    ap.add_argument("--xmin_eval", type=float, default=None, help="Manual x-axis MIN for SDP row (default: auto).")
    ap.add_argument("--xmax_eval", type=float, default=None, help="Manual x-axis MAX for SDP row (default: auto).")
    ap.add_argument("--xmin_jit",  type=float, default=None, help="Manual x-axis MIN for JIT row (default: auto).")
    ap.add_argument("--xmax_jit",  type=float, default=None, help="Manual x-axis MAX for JIT row (default: auto).")

    args = ap.parse_args()

    eval_data_utils = import_eval_data_utils()
    jit_data_utils = import_jit_data_utils()

    ds_eval = eval_data_utils.read_dataset()
    ds_jit = jit_data_utils.read_dataset()

    eval_projects = parse_projects(args.projects_eval, sorted(ds_eval.keys()))
    jit_projects  = parse_projects(args.projects_jit,  sorted(ds_jit.keys()))
    model_types   = parse_models(args.models)

    print(f"[Eval] Projects ({len(eval_projects)}): {eval_projects}")
    print(f"[JIT ] Projects ({len(jit_projects)}): {jit_projects}")
    print(f"Models: {model_types}\n")

    # Eval history
    total_deltas_eval, feat_cols_eval, mean_eval, inv_cov_eval = \
        build_history_distribution_eval(ds_eval, eval_projects)
    X_hist_eval = total_deltas_eval[feat_cols_eval].values.astype(float)
    hist_eval_vals = np.array(
        [float(mahalanobis(x, mean_eval, inv_cov_eval)) for x in X_hist_eval],
        dtype=float,
    )

    # JIT history
    total_hist_jit, feat_cols_jit, mean_jit, inv_cov_jit = \
        build_history_distribution_jit(ds_jit, jit_projects)
    X_hist_jit = total_hist_jit[feat_cols_jit].values.astype(float)
    hist_jit_vals = np.array(
        [float(mahalanobis(x, mean_jit, inv_cov_jit)) for x in X_hist_jit],
        dtype=float,
    )

    # Load cached explainer distances + stats
    eval_dist_dicts, eval_stats = load_cached_distances_eval(model_types, hist_eval_vals)
    jit_dist_dicts, jit_stats   = load_cached_distances_jit(model_types, hist_jit_vals)

    if not (eval_dist_dicts or jit_dist_dicts):
        print("[RQ3] No models had enough cached data to plot.")
        return

    # X-ranges (auto then cut to 20/10 as you wanted)
    auto_min_eval, auto_max_eval = _compute_auto_limits(eval_dist_dicts, 0.0, 20.0) if eval_dist_dicts else (0.0, 20.0)
    auto_min_jit,  auto_max_jit  = _compute_auto_limits(jit_dist_dicts,  0.0, 10.0) if jit_dist_dicts  else (0.0, 10.0)

    xmin_eval = args.xmin_eval if args.xmin_eval is not None else auto_min_eval
    xmax_eval = args.xmax_eval if args.xmax_eval is not None else auto_max_eval
    xmin_jit  = args.xmin_jit  if args.xmin_jit  is not None else auto_min_jit
    xmax_jit  = args.xmax_jit  if args.xmax_jit  is not None else auto_max_jit

    # hard caps
    xmax_eval = 20.0
    xmax_jit  = 10.0

    if xmax_eval <= xmin_eval:
        xmax_eval = xmin_eval + 1.0
    if xmax_jit <= xmin_jit:
        xmax_jit = xmin_jit + 1.0

    print(f"[X-range] SDP:     {xmin_eval:.3f} → {xmax_eval:.3f}")
    print(f"[X-range] JIT-SDP: {xmin_jit:.3f}  → {xmax_jit:.3f}")

    out_root = Path("./figures_rq3")
    out_root.mkdir(parents=True, exist_ok=True)

    # Plot
    out_path = out_root / "rq3_mahalanobis_sdp_vs_jit_horizontal_cached.png"
    overlap_df = plot_combined_horizontal(
        eval_dist_dicts,
        jit_dist_dicts,
        out_path,
        x_limits_eval=(xmin_eval, xmax_eval),
        x_limits_jit=(xmin_jit,  xmax_jit),
    )
    save_overlap_table(overlap_df, out_root)

    # Save stats
    if not eval_stats.empty:
        eval_stats.to_csv(out_root / "rq3_mahalanobis_stats_eval.csv", index=False)
        print(f"[Stats] Saved eval stats -> {out_root / 'rq3_mahalanobis_stats_eval.csv'}")
    else:
        print("[Stats] No eval stats to save.")

    if not jit_stats.empty:
        jit_stats.to_csv(out_root / "rq3_mahalanobis_stats_jit.csv", index=False)
        print(f"[Stats] Saved jit stats -> {out_root / 'rq3_mahalanobis_stats_jit.csv'}")
    else:
        print("[Stats] No jit stats to save.")


if __name__ == "__main__":
    main()