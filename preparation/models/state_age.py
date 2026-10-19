"""Load the saved category and age probability models without fitting."""
from __future__ import annotations
import numpy as np
from methods.models import SparseBackoffMarkov, _SparseLevel

def sampled_age(states: np.ndarray, blocks: np.ndarray, step: int = 5) -> np.ndarray:
    """Causal age per phase; -1 until an observed change within a retained block."""
    states = np.asarray(states)
    blocks = np.asarray(blocks)
    if states.ndim != 1 or states.shape != blocks.shape or step < 1:
        raise ValueError("invalid sequence, block vector or step")
    age = np.full(len(states), -1, dtype=np.int64)
    for i in range(step, len(states)):
        if blocks[i] < 0 or not np.all(blocks[i-step:i+1] == blocks[i]):
            continue
        if states[i] != states[i-step]:
            age[i] = 0
        elif age[i-step] >= 0:
            age[i] = age[i-step] + 1
    return age


class AgeModel:
    def __init__(self, edges, alpha, base_alpha=None):
        self.edges = tuple(map(int, edges))
        if not self.edges or self.edges[0] != 0 or any(np.diff(self.edges) <= 0):
            raise ValueError("age bin lower edges must start at zero and increase")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("positive finite smoothing required")
        self.alpha = float(alpha)
        self.base_alpha = self.alpha if base_alpha is None else float(base_alpha)

    def fit(self, ctx, age, targets):
        ctx = np.asarray(ctx, dtype=np.int64)
        age = np.asarray(age, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        self.base = SparseBackoffMarkov(1, alpha=self.base_alpha, reflection_pool=True).fit(ctx, targets)
        self.counts = np.zeros((len(self.edges), 3, 3), dtype=np.float64)
        known = age >= 0
        bins = np.searchsorted(self.edges, age[known], side="right")-1
        current, future = ctx[known, 0], targets[known]
        np.add.at(self.counts, (bins, current, future), 1)
        np.add.at(self.counts, (bins, 2-current, 2-future), 1)
        return self

    def predict_proba(self, ctx, age):
        ctx, age = np.asarray(ctx), np.asarray(age)
        p = self.base.predict_proba(ctx)
        known = age >= 0
        bins = np.searchsorted(self.edges, age[known], side="right")-1
        counts = self.counts[bins, ctx[known, 0]]
        p[known] = (counts + self.alpha*p[known])/(counts.sum(axis=1, keepdims=True)+self.alpha)
        return p

def load_saved(document):
    base=SparseBackoffMarkov(document["order"], alpha=document["alpha"],
                             reflection_pool=document["reflection_pool"])
    base._counts=tuple(_SparseLevel(np.asarray(x["codes"],dtype=np.int64),
                                    np.asarray(x["counts"],dtype=np.float64))
                       for x in document["levels"])
    if "age_counts" not in document:
        return base
    model=AgeModel(document["age_bin_lower_steps"],document["age_alpha"],document["alpha"])
    model.base=base
    model.counts=np.asarray(document["age_counts"],dtype=np.float64)
    return model
