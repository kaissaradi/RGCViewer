"""Suggested classes for a new run, from the lab's labelled cells (PLAN.md Q36).

* A random forest (class-balanced) on ``type_features`` over TRAIN_CLASSES.
  Leave-one-prep-out it scored 0.90 balanced accuracy; its probability is
  usable for ordering a review queue (PLAN.md Q36).
* ON / OFF is decided by the STA sign rule, not learned: classes of the
  other polarity get probability 0.
* Novelty gate: the forest gives confident answers to types it never saw.
  A cell whose distance to its k-th nearest labelled cell is beyond what
  labelled cells show among themselves (NOVEL_PERCENTILE) gets no
  suggestion ("unlike any labelled cell").

Train at run time from ``type_library``; nothing is saved to the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .type_library import Library

N_TREES = 300
K_NOVEL = 10
NOVEL_PERCENTILE = 99.0
REVIEW_BELOW = 0.8            # suggestions below this go to the top of the review queue


@dataclass
class Suggestion:
    ranked: List[Tuple[str, float]]    # classes by probability (polarity-consistent only)
    novel: bool                        # unlike any labelled cell: no suggestion shown
    novelty: float                     # k-NN distance / library threshold (> 1: novel)
    polarity: int

    @property
    def best(self) -> Optional[Tuple[str, float]]:
        return None if self.novel or not self.ranked else self.ranked[0]

    @property
    def confident(self) -> bool:
        b = self.best
        return b is not None and b[1] >= REVIEW_BELOW


def _prep(X: np.ndarray) -> np.ndarray:
    """No-fit RF sizes become the run median (0): the forest needs numbers."""
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)


class Suggester:
    def __init__(self, library: Library, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import NearestNeighbors
        if library.n == 0:
            raise ValueError("the type library is empty")
        X = _prep(library.X)
        self.classes = sorted(set(library.y))
        self.forest = RandomForestClassifier(
            n_estimators=N_TREES, class_weight="balanced", random_state=seed, n_jobs=-1)
        self.forest.fit(X, library.y)
        self._mu, self._sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self._mu) / self._sd
        self._nn = NearestNeighbors(n_neighbors=K_NOVEL + 1).fit(Z)
        d, _ = self._nn.kneighbors(Z)
        self.novel_threshold = float(np.percentile(d[:, -1], NOVEL_PERCENTILE))  # skip self
        self.n_train = library.n

    def suggest(self, X: np.ndarray, polarity: Sequence[int]) -> List[Suggestion]:
        X = _prep(X)
        if len(X) == 0:
            return []
        proba = self.forest.predict_proba(X)
        names = list(self.forest.classes_)
        on = np.array([n.startswith("ON") for n in names])
        d, _ = self._nn.kneighbors((X - self._mu) / self._sd, n_neighbors=K_NOVEL)
        novelty = d[:, -1] / self.novel_threshold
        out = []
        for i, p in enumerate(proba):
            pol = int(polarity[i])
            if pol != 0:
                p = np.where(on == (pol > 0), p, 0.0)
            total = p.sum()
            ranked = [] if total <= 0 else sorted(
                ((names[k], float(p[k] / total)) for k in range(len(names)) if p[k] > 0),
                key=lambda t: -t[1])
            out.append(Suggestion(ranked, bool(novelty[i] > 1.0) or pol == 0,
                                  float(novelty[i]), pol))
        return out
