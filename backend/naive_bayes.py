"""Mixed Naive Bayes classifier, written from Bayes' rule with NumPy.

    P(D | x) = P(D) * prod_j p(x_j | D) / sum_d P(d) * prod_j p(x_j | d)

The "naive" part is the conditional-independence assumption: given the class,
every answer is treated as independent of every other answer, so the joint
likelihood factorises into one term per feature. Each feature gets the density
that suits its type:

* binary / ordinal / binned-days answers -> categorical likelihood with Laplace
  smoothing  p(x = k | d) = (N_dk + alpha) / (N_d + K * alpha)
* BMI (continuous, right-skewed) -> log-normal density  ln BMI | d ~ N(mu_d, var_d)
"""
import math

import numpy as np
import pandas as pd

from . import config
from .data import bin_days


def _levels(feature: dict) -> list:
    kind = feature["kind"]
    if kind == "binary":
        return [0, 1]
    if kind == "ordinal":
        return feature["levels"]
    if kind == "days":
        return list(range(len(config.DAY_BIN_LABELS)))
    if kind == "binned":
        return list(range(len(feature["edges"]) + 1))
    raise ValueError(kind)


def encode_column(feature: dict, values) -> np.ndarray:
    """Turn raw survey answers into category indices 0..K-1 (or floats for continuous)."""
    v = np.asarray(values, dtype=float)
    kind = feature["kind"]
    if kind == "continuous":
        return v
    if kind == "binary":
        return v.astype(int)
    if kind == "ordinal":
        return (v - feature["levels"][0]).astype(int)
    if kind == "days":
        return bin_days(v)
    if kind == "binned":
        return np.digitize(v, feature["edges"])
    raise ValueError(kind)


def _logsumexp(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class MixedNaiveBayes:
    def __init__(self, features=None, alpha: float = config.SMOOTHING_ALPHA, continuous: str = "lognormal"):
        self.features = features if features is not None else config.FEATURES
        self.alpha = alpha
        self.continuous = continuous  # "lognormal" or "gaussian"
        self.params: dict = {}
        self.log_prior = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "MixedNaiveBayes":
        y = np.asarray(y).astype(int)
        n_d = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float)
        self.class_counts = n_d.tolist()
        self.log_prior = np.log(n_d / n_d.sum())
        for f in self.features:
            col = encode_column(f, X[f["key"]])
            if f["kind"] == "continuous":
                z = np.log(col) if self.continuous == "lognormal" else col
                mu = np.array([z[y == d].mean() for d in (0, 1)])
                var = np.array([z[y == d].var() for d in (0, 1)])  # MLE variance
                self.params[f["key"]] = {"type": self.continuous, "mu": mu.tolist(), "var": var.tolist()}
            else:
                K = len(_levels(f))
                counts = np.zeros((2, K))
                for d in (0, 1):
                    counts[d] = np.bincount(col[y == d], minlength=K)[:K]
                probs = (counts + self.alpha) / (counts.sum(axis=1, keepdims=True) + K * self.alpha)
                self.params[f["key"]] = {"type": "categorical", "counts": counts.tolist(),
                                         "log_prob": np.log(probs).tolist()}
        return self

    # ------------------------------------------------------------ inference
    def feature_log_likelihoods(self, X: pd.DataFrame) -> np.ndarray:
        """log p(x_j | d) for every row, feature and class -> shape (n, n_features, 2)."""
        n = len(X)
        out = np.empty((n, len(self.features), 2))
        for j, f in enumerate(self.features):
            p = self.params[f["key"]]
            col = encode_column(f, X[f["key"]])
            if p["type"] == "categorical":
                lp = np.array(p["log_prob"])
                idx = np.clip(col, 0, lp.shape[1] - 1)
                out[:, j, :] = lp[:, idx].T
            else:
                mu, var = np.array(p["mu"]), np.array(p["var"])
                z = np.log(col) if p["type"] == "lognormal" else col
                ll = -0.5 * np.log(2 * math.pi * var) - (z[:, None] - mu) ** 2 / (2 * var)
                if p["type"] == "lognormal":
                    ll = ll - z[:, None]  # Jacobian of the log transform (cancels in the posterior)
                out[:, j, :] = ll
        return out

    def joint_log_likelihood(self, X: pd.DataFrame) -> np.ndarray:
        """log P(d) + sum_j log p(x_j | d)  -> shape (n, 2)."""
        return self.log_prior + self.feature_log_likelihoods(X).sum(axis=1)

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return jll - _logsumexp(jll)[:, None]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(D = 1 | x) for each row."""
        return np.exp(self.predict_log_proba(X)[:, 1])

    def explain(self, X: pd.DataFrame) -> dict:
        """Break one respondent's posterior log-odds into prior + one term per answer.

        log-odds(D | x) = log[P(D)/P(not D)] + sum_j log[p(x_j | D) / p(x_j | not D)]
        """
        fll = self.feature_log_likelihoods(X)[0]
        terms = fll[:, 1] - fll[:, 0]
        prior_logodds = float(self.log_prior[1] - self.log_prior[0])
        return {
            "prior_log_odds": prior_logodds,
            "terms": {f["key"]: float(t) for f, t in zip(self.features, terms)},
            "posterior_log_odds": prior_logodds + float(terms.sum()),
        }

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "continuous": self.continuous,
                "log_prior": self.log_prior.tolist(), "class_counts": self.class_counts,
                "params": self.params, "features": [f["key"] for f in self.features]}

    @classmethod
    def from_dict(cls, d: dict, features=None) -> "MixedNaiveBayes":
        feats = features if features is not None else [config.FEATURE_BY_KEY[k] for k in d["features"]]
        m = cls(feats, alpha=d["alpha"], continuous=d["continuous"])
        m.log_prior = np.array(d["log_prior"])
        m.class_counts = d["class_counts"]
        m.params = d["params"]
        return m


BMI_EDGES = [18.5, 22, 25, 27.5, 30, 32.5, 35, 40, 45]


def all_categorical_features() -> list[dict]:
    """Same features, but BMI binned -> a model sklearn's CategoricalNB can reproduce exactly."""
    feats = []
    for f in config.FEATURES:
        if f["kind"] == "continuous":
            f = {**f, "kind": "binned", "edges": BMI_EDGES}
        feats.append(f)
    return feats


def encode_matrix(features: list[dict], X: pd.DataFrame) -> np.ndarray:
    """Integer-coded design matrix (used to feed sklearn for validation)."""
    return np.column_stack([encode_column(f, X[f["key"]]) for f in features])
