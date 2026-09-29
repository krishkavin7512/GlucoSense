"""Turning Naive Bayes scores into honest probabilities by counting.

Naive Bayes multiplies 21 likelihoods as if they were independent. They are
not (general health and poor-physical-health days are strongly linked), so the
same evidence gets counted twice and the scores drift towards 0 and 1.

The fix is a conditional-probability estimate on held-out data:

    P(D | score in bin b) = (# people with diabetes in bin b) / (# people in bin b)

Bins hold equal numbers of validation people. The per-bin rates are made
non-decreasing with the pool-adjacent-violators rule (a higher score must never
mean a lower risk), then interpolated between bin centres on the log-odds scale.
"""
import numpy as np


def logit(p, eps=1e-9):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def pool_adjacent_violators(values, weights):
    """Weighted least-squares non-decreasing fit to a sequence."""
    blocks = [[float(v), float(w), 1] for v, w in zip(values, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out.extend([v] * n)
    return np.array(out)


class BinningCalibrator:
    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins
        self.x = None  # bin centres in log-odds of the raw score
        self.y = None  # calibrated probability at each centre

    def fit(self, scores, y) -> "BinningCalibrator":
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(scores)
        chunks = np.array_split(order, self.n_bins)
        centres = np.array([logit(scores[c]).mean() for c in chunks])
        # Laplace's rule of succession: (k + 1) / (n + 2) never reports exactly 0 or 1.
        rates = np.array([(y[c].sum() + 1) / (len(c) + 2) for c in chunks])
        counts = np.array([len(c) for c in chunks], dtype=float)
        self.raw_rates = rates
        self.x = centres
        self.y = pool_adjacent_violators(rates, counts)
        return self

    def transform(self, scores) -> np.ndarray:
        return np.interp(logit(scores), self.x, self.y)

    def to_dict(self):
        return {"x": self.x.tolist(), "y": self.y.tolist(), "raw_rates": self.raw_rates.tolist()}

    @classmethod
    def from_dict(cls, d):
        c = cls(len(d["x"]))
        c.x, c.y, c.raw_rates = np.array(d["x"]), np.array(d["y"]), np.array(d["raw_rates"])
        return c
