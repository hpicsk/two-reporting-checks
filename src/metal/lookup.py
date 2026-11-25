"""The predictor every dataset in this repository is scored against.

Copy the nearest training row's target. Nothing is fitted to the target. There
is no hyperparameter but k = 1, and no fitted quantity except the min-max range the
distance needs, which is read off the training rows alone. A paper about what a
partition leaks cannot fit its own scaler on the test rows, so `scale` takes
the training block and every caller passes it a training block.

Ties are averaged rather than broken by index order. On a dataset in which
many rows record identical features the tie set is large, and a tie broken by
index order would make the score depend on the order of the rows; the mean
over the tied set does not.

Distances are L1 in the scaled feature space, which is the space the lookup
itself searches, so a distance reported beside an R^2 describes the geometry
that produced it.

Degrees of freedom:
  K               CHOSEN -- one neighbour, the nearest-neighbour baseline of
                  Meredig et al. (2018); see the BASELINE tag.
  METRIC_P        CHOSEN -- the L1 distance, in features rescaled to span 0 to
                  1 over the training rows (sec:predictors). It sums each
                  feature's absolute difference with equal weight, so a large
                  difference in one feature counts less than under the
                  Euclidean distance, which squares it. No other distance was
                  scored.
  TIE_RADIUS_EPS  CHOSEN -- ties are exact; the epsilon absorbs the kd-tree's
                  floating-point rounding of a distance.
  Nothing is FITTED to the target.

BASELINE: the 1-nearest-neighbour lookup of Meredig et al., Mol. Syst. Des.
Eng. 3, 819 (2018) | obvious alternative: more than one neighbour, or a
Euclidean distance on standardized features | tuning: none | discharge:
sec:predictors -- the lookup is the field's named baseline, the
nearest-neighbour baseline of Meredig et al.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

K = 1
METRIC_P = 1          # L1
TIE_RADIUS_EPS = 1e-12  # ties are exact; the epsilon absorbs kd-tree rounding


def scale(train_X: np.ndarray):
    """Min-max range fit on the training block. Zero-width columns map to 1."""
    train_X = np.asarray(train_X, dtype=float)
    lo, hi = train_X.min(axis=0), train_X.max(axis=0)
    return lo, np.where(hi - lo == 0, 1.0, hi - lo)


def one_nn(train_X, train_y, test_X):
    """Predict each test row by its nearest training row, ties averaged.

    Returns the predictions and each test row's L1 distance to that neighbour.
    """
    train_X = np.asarray(train_X, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    lo, span = scale(train_X)
    ref, query = (train_X - lo) / span, (np.asarray(test_X, float) - lo) / span

    tree = cKDTree(ref)
    distance, _ = tree.query(query, k=K, p=METRIC_P)
    predicted = np.empty(len(query))
    for i, (row, d) in enumerate(zip(query, distance)):
        tied = tree.query_ball_point(row, r=d + TIE_RADIUS_EPS, p=METRIC_P)
        predicted[i] = train_y[tied].mean()
    return predicted, distance
