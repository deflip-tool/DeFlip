# =============================
# heuristic.py — beam search (no reward)
# =============================
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class optimization(ABC):
    @abstractmethod
    def optimize(self):
        raise NotImplementedError


class best_first(optimization):
    """
    Best-first beam search **without** a reward function.
    Behavior matches the prior non-reward version:
      • Global pool ranked by model target probability minus a small L0 penalty.
      • Candidate values built from target-class rows (if labels available) or all rows.
      • Numeric candidates also include steps toward the nearest neighbour (NN).
      • Edit budget enforced by `max_edits`.
    Note: `per_state_top` is accepted for API compatibility but not used when reward is absent.
    """

    def __init__(
        self,
        data,
        max_edits: int | None = 5,
        beam_width: int = 12,
        per_state_top: int = 4,  # kept for compatibility; unused here
        max_iters: int = 500,
        rtol: float = 1e-7,
        atol: float = 1e-8,
        num_percentiles: Tuple[int, int, int, int, int] = (10, 25, 50, 75, 90),
        cat_top_k: int = 3,
        nn_steps: Tuple[float, float, float, float] = (0.25, 0.5, 0.75, 1.0),
    ):
        self.data = data
        self.max_edits = max_edits
        self.beam_width = int(beam_width)
        self.per_state_top = int(per_state_top)
        self.max_iters = int(max_iters)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.num_percentiles = tuple(num_percentiles)
        self.cat_top_k = int(cat_top_k)
        self.nn_steps = tuple(nn_steps)

        # LAZY: candidate values are built in optimize(), after target_class exists
        self._cand_values = None
        self._cand_cache_key = None  # (tuple(target_class), id(X_train), id(y_train))

    # ---------- helpers ----------
    def _isclose(self, a, b) -> np.ndarray:
        return np.isclose(a, b, rtol=self.rtol, atol=self.atol)

    def _count_edits(self, X, X0) -> int:
        return int((~self._isclose(X, X0)).sum())

    def _score_targets(self, X_batch: np.ndarray) -> np.ndarray:
        scores = self.data.predict_fn(X_batch)  # (n, C)
        tgt = np.array(self.data.target_class, dtype=int)
        return scores[:, tgt].max(axis=1)

    def _prepare_candidate_values(self):
        """Build per-feature candidate pool using current target_class; cached across calls."""
        tgt_tuple = tuple(sorted(self.data.target_class)) if hasattr(self.data, "target_class") else ("ALL",)
        cache_key = (tgt_tuple, id(self.data.X_train), id(self.data.y_train))
        if self._cand_values is not None and cache_key == self._cand_cache_key:
            return  # already prepared for this setting

        Xtr = self.data.X_train
        ytr = self.data.y_train
        d = Xtr.shape[1]

        # If target_class not set yet or labels absent, use all rows
        if getattr(self.data, "target_class", None) is None or ytr is None:
            tgt_mask = np.ones(Xtr.shape[0], dtype=bool)
        else:
            tgt_mask = np.isin(ytr, self.data.target_class)

        Xt = Xtr[tgt_mask]
        cat_idx = set(self.data.cat_feat)

        cands: List[List[float]] = []
        for j in range(d):
            if j in cat_idx:
                vals, counts = np.unique(Xt[:, j], return_counts=True)
                order = np.argsort(-counts)
                cands_j = vals[order][: self.cat_top_k].tolist()
            else:
                col = Xt[:, j].astype(float)
                qs = np.percentile(col, self.num_percentiles).tolist()
                cands_j = qs
            cands.append(cands_j)

        self._cand_values = cands
        self._cand_cache_key = cache_key

    # ---------- main search ----------
    def optimize(self, NN: np.ndarray) -> np.ndarray:
        # Ensure fit_to_X has run so that target_class is set
        self._prepare_candidate_values()

        x0 = self.data.X.copy()  # (1, d)
        d = x0.shape[1]

        # Ensure NN values are included in candidate pool
        cand_values = [list(cv) for cv in self._cand_values]
        for j in range(d):
            nn_val = NN[0, j]
            if not any(self._isclose(nn_val, v) for v in cand_values[j]):
                cand_values[j].append(nn_val)

        beam = [x0.copy()]

        for _ in range(self.max_iters):
            preds = self.data.predict_fn(np.vstack(beam))
            if np.any(np.isin(np.argmax(preds, axis=1), self.data.target_class)):
                scores = self._score_targets(np.vstack(beam))
                return beam[int(np.argmax(scores))]

            pool, seen = [], set()
            for state in beam:
                edits_curr = self._count_edits(state, x0)

                if (self.max_edits is None) or (edits_curr < self.max_edits):
                    for j in range(d):
                        vals = list(cand_values[j])
                        if j not in set(self.data.cat_feat):
                            delta = float(NN[0, j]) - float(state[0, j])
                            for s in self.nn_steps:
                                vals.append(float(state[0, j]) + s * delta)

                        for v in vals:
                            if self._isclose(state[0, j], v):
                                continue
                            was_changed = int(not self._isclose(state[0, j], x0[0, j]))
                            will_changed = int(not self._isclose(v,          x0[0, j]))
                            edits_next = edits_curr - was_changed + will_changed
                            if (self.max_edits is not None) and (edits_next > self.max_edits):
                                continue
                            x_next = state.copy()
                            x_next[0, j] = v
                            key = tuple(np.asarray(x_next).ravel())
                            if key in seen:
                                continue
                            seen.add(key)
                            pool.append(x_next)

            if not pool:
                scores = self._score_targets(np.vstack(beam))
                return beam[int(np.argmax(scores))]

            X_pool = np.vstack(pool)
            sc_pool = self._score_targets(X_pool)
            edits_pool = np.array([self._count_edits(x.reshape(1, d), x0) for x in pool], dtype=float)
            rank_pool = sc_pool - 1e-3 * edits_pool
            top = np.argsort(-rank_pool)[: self.beam_width]
            beam = [pool[i] for i in top]

        scores = self._score_targets(np.vstack(beam))
        return beam[int(np.argmax(scores))]


