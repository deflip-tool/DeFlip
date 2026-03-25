# =============================
# nice/__init__.py — NICE wrapper (no reward)
# =============================
from __future__ import annotations
from typing import Optional, Iterable, Any
import numpy as np

from nice.utils.distance import HEOM, StandardDistance, MinMaxDistance, NearestNeighbour
from nice.utils.data import data_NICE
from nice.utils.optimization.heuristic import best_first

CRITERIA_DIS = {"HEOM": HEOM}
CRITERIA_NRM = {"std": StandardDistance, "minmax": MinMaxDistance}


class NICE:
    """
    NICE wrapper using beam search **without** any reward function.

    Public API and constructor signature remain compatible; unused arguments are
    accepted and ignored to avoid breaking downstream callers.
    """

    def __init__(
        self,
        predict_fn,
        X_train: np.ndarray,
        cat_feat: list | tuple,
        num_feat: Any = "auto",
        y_train: Optional[np.ndarray] = None,
        optimization: str = "sparsity",   # accepted but ignored for search logic
        justified_cf: bool = True,
        distance_metric: str = "HEOM",
        num_normalization: str = "minmax",
        auto_encoder=None,                 # accepted but unused
        max_edits: Optional[int] = None,
        # Post-hoc controls
        posthoc_refine: bool = True,
        refine_bisect_steps: int = 12,
        refine_hist_k: int = 200,
    ):
        print(f"[NICE] Initialization (reward-free beam), optimization='{optimization}'")
        self.optimization = optimization
        self.posthoc_refine = bool(posthoc_refine)
        self.refine_bisect_steps = int(refine_bisect_steps)
        self.refine_hist_k = int(refine_hist_k)
        self.max_edits = max_edits

        # Core data plumbing
        self.data = data_NICE(
            X_train=X_train,
            y_train=y_train,
            cat_feat=list(cat_feat),
            num_feat=num_feat,
            predict_fn=predict_fn,
            justified_cf=justified_cf,
            eps=1e-11,
        )
        self._feat_min = self.data.X_train.min(axis=0)
        self._feat_max = self.data.X_train.max(axis=0)

        # Distance + NN
        self.distance_metric = CRITERIA_DIS[distance_metric](self.data, CRITERIA_NRM[num_normalization])
        self.nearest_neighbour = NearestNeighbour(self.data, self.distance_metric)

        # Optimizer: use best_first with no reward. If optimization == 'none', skip search.
        if optimization != "none":
            self.optimizer = best_first(
                self.data,
                max_edits=max_edits,
                beam_width=12,
                per_state_top=4,
                max_iters=500,
            )
        else:
            self.optimizer = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def explain(self, X: np.ndarray, target_class: str | Iterable[int] = "other") -> np.ndarray:
        self.data.fit_to_X(X, target_class)
        NN = self.nearest_neighbour.find_neighbour(self.data.X)

        CF = NN if (self.optimizer is None) else self.optimizer.optimize(NN)

        if self.posthoc_refine and self._is_flip(CF):
            CF = self._refine_coord_toward_original(self.data.X, CF, passes=2, steps=self.refine_bisect_steps)
            # Optional: history tightening (kept off by default)
            # CF = self._refine_toward_history(self.data.X, CF, steps=self.refine_bisect_steps, hist_k=self.refine_hist_k)
        return CF

    # ---------------------------------------------------------------------
    # Flip & edit helpers
    # ---------------------------------------------------------------------
    def _is_flip(self, x: np.ndarray) -> bool:
        y = int(np.argmax(self.data.predict_fn(x), axis=1)[0])
        tc = self.data.target_class
        if isinstance(tc, str):  # 'other' → not original class
            y0 = int(np.argmax(self.data.predict_fn(self.data.X), axis=1)[0])
            return y != y0
        tc_list = np.array(tc).astype(int).ravel().tolist()
        return y in tc_list
    def _heom_contribution(self, x0, x, j, rtol=1e-7, atol=1e-8) -> float:
        """
        Normalized HEOM-style contribution of feature j between x0 and x:
        |x_j - x0_j| / (max_j - min_j), clipped to [0,1].

        - For categorical features: 1.0 if changed, 0.0 if not.
        - For numeric features with zero range: 0.0.
        """
        cat = set(self.data.cat_feat)
        # Categorical: 1 if changed, else 0
        if j in cat:
            return 1.0 if x[0, j] != x0[0, j] else 0.0

        # Numeric: normalized absolute difference
        rng = float(self._feat_max[j] - self._feat_min[j])
        if rng <= 0:
            return 0.0
        diff = abs(float(x[0, j]) - float(x0[0, j]))
        return float(max(0.0, min(1.0, diff / rng)))

    def _edit_count(self, x: np.ndarray, x0: np.ndarray, rtol=1e-7, atol=1e-8) -> int:
        x = np.asarray(x).reshape(1, -1)
        x0 = np.asarray(x0).reshape(1, -1)
        cnt, cat = 0, set(self.data.cat_feat)
        for j in range(x.shape[1]):
            if j in cat:
                if x[0, j] != x0[0, j]:
                    cnt += 1
            else:
                if not np.isclose(float(x[0, j]), float(x0[0, j]), rtol=rtol, atol=atol):
                    cnt += 1
        return cnt

    # ---------------------------------------------------------------------
    # Post-hoc #1: coordinate tightening toward the original
    # ---------------------------------------------------------------------
    def _refine_coord_toward_original(self, x0, x_cf, passes=2, steps=12, rtol=1e-7, atol=1e-8):
        """Iteratively try to revert each changed feature toward original without losing the flip."""
        x0 = np.asarray(x0).reshape(1, -1).copy()
        x = np.asarray(x_cf).reshape(1, -1).copy()
        if not self._is_flip(x):
            return x

        d, cat = x.shape[1], set(self.data.cat_feat)

        def changed_mask():
            m = np.zeros(d, dtype=bool)
            for j in range(d):
                if j in cat:
                    m[j] = (x[0, j] != x0[0, j])
                else:
                    m[j] = not np.isclose(float(x[0, j]), float(x0[0, j]), rtol=rtol, atol=atol)
            return m

        for _ in range(int(passes)):
            m = changed_mask()
            improved = False

            # Categorical: full revert if flip holds
            cat_mask = np.array([j in cat for j in range(d)], dtype=bool)
            for j in np.where(m & cat_mask)[0]:
                prev = x[0, j]
                x[0, j] = x0[0, j]
                if not self._is_flip(x):
                    x[0, j] = prev
                else:
                    improved = True

            # Numeric: HEOM-normalized ordering + bisection toward original
            num_mask = ~cat_mask
            num_indices = np.where(m & num_mask)[0]

            # Order changed numeric features by normalized HEOM contribution (largest first)
            ordered_num_indices = sorted(
                num_indices,
                key=lambda j: self._heom_contribution(x0, x, j, rtol=rtol, atol=atol),
                reverse=True,
            )

            for j in ordered_num_indices:
                a = float(x0[0, j]); b = float(x[0, j])
                if np.isclose(a, b, rtol=rtol, atol=atol):
                    continue

                # Try full revert first
                tmp = x.copy(); tmp[0, j] = a
                if self._is_flip(tmp):
                    x = tmp; improved = True; continue

                # Otherwise, binary search between original and current CF value
                lo, hi = a, b  # lo = no-flip side, hi = flip side (current)
                for _ in range(int(steps)):
                    mid = (lo + hi) / 2.0
                    tmp = x.copy(); tmp[0, j] = mid
                    if self._is_flip(tmp):
                        hi = mid; x = tmp; improved = True
                    else:
                        lo = mid

            if not improved:
                break
        return x

    # ---------------------------------------------------------------------
    # Post-hoc #2: gentle snap/tighten toward nearest historical exemplar
    # ---------------------------------------------------------------------
    def _hist_pool(self, hist_k: int):
        H = self.data.X_train
        if (self.data.y_train is not None) and getattr(self.data, "target_class", None) is not None:
            H = H[np.isin(self.data.y_train, self.data.target_class)]
        if H.shape[0] > hist_k:
            rng = np.random.RandomState(0)
            H = H[rng.choice(H.shape[0], hist_k, replace=False)]
        return H

    def _nearest_hist_point(self, x, H):
        d = self.distance_metric.measure(x, H).ravel()
        idx = int(np.argmin(d))
        return H[idx:idx + 1, :], float(d[idx])

    def _refine_toward_history(self, x0, x_cf, steps=12, hist_k=200, rtol=1e-7, atol=1e-8):
        x0 = np.asarray(x0).reshape(1, -1)
        x = np.asarray(x_cf).reshape(1, -1).copy()
        if not self._is_flip(x):
            return x

        cat = set(self.data.cat_feat)
        H = self._hist_pool(int(hist_k))
        Hpt, base_dist = self._nearest_hist_point(x, H)

        # Categorical snaps
        for j in range(x.shape[1]):
            if j in cat and x[0, j] != x0[0, j]:
                prev = x[0, j]
                cand = Hpt[0, j]
                if cand == prev:
                    continue
                x[0, j] = cand
                if self._is_flip(x):
                    new = float(self.distance_metric.measure(x, H).ravel().min())
                    if new + 1e-12 < base_dist:
                        base_dist = new
                    else:
                        x[0, j] = prev
                else:
                    x[0, j] = prev

        # Numeric tightening toward exemplar
        for j in range(x.shape[1]):
            if j in cat:
                continue
            if np.isclose(float(x[0, j]), float(x0[0, j]), rtol=rtol, atol=atol):
                continue

            target = float(Hpt[0, j])
            right = float(x[0, j])

            for _ in range(int(steps)):
                mid = (right + target) / 2.0
                cur = x.copy(); cur[0, j] = mid
                if self._is_flip(cur):
                    new = float(self.distance_metric.measure(cur, H).ravel().min())
                    if new + 1e-12 <= base_dist:
                        x, right, base_dist = cur, mid, new
                    else:
                        break
                else:
                    break

        assert self._edit_count(x, x0, rtol, atol) <= self._edit_count(x_cf, x0, rtol, atol)
        return x
