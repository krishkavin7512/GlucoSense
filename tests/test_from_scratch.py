"""Every from-scratch implementation is checked against a reference library."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.naive_bayes import CategoricalNB

from backend import metrics
from backend import probability as P
from backend.calibration import BinningCalibrator, pool_adjacent_violators
from backend.naive_bayes import MixedNaiveBayes, all_categorical_features, encode_matrix, _levels
from backend.polyfit import PolynomialCurveFit, scale
from backend import config

rng = np.random.default_rng(0)


def synthetic(n=4000):
    """A small fake survey with the same columns and value ranges as the real data."""
    d = {}
    y = rng.random(n) < 0.2
    for f in config.FEATURES:
        k = f["key"]
        if f["kind"] == "binary":
            d[k] = (rng.random(n) < np.where(y, 0.6, 0.3)).astype(int)
        elif f["kind"] == "ordinal":
            d[k] = rng.choice(f["levels"], n)
        elif f["kind"] == "days":
            d[k] = rng.choice([0, 0, 0, 2, 5, 10, 20, 30], n)
        else:
            d[k] = np.round(np.exp(rng.normal(np.where(y, 3.45, 3.3), 0.2)), 0).clip(12, 98)
    d[config.TARGET] = y.astype(int)
    return pd.DataFrame(d)


def test_naive_bayes_matches_sklearn_categorical():
    df = synthetic()
    feats = all_categorical_features()
    ours = MixedNaiveBayes(feats, alpha=1.0).fit(df, df[config.TARGET]).predict_proba(df)
    E = encode_matrix(feats, df)
    sk = CategoricalNB(alpha=1.0, min_categories=[len(_levels(f)) for f in feats]).fit(E, df[config.TARGET])
    assert np.abs(ours - sk.predict_proba(E)[:, 1]).max() < 1e-10


def test_explanation_adds_up_to_posterior():
    df = synthetic()
    m = MixedNaiveBayes().fit(df, df[config.TARGET])
    row = df.iloc[[3]]
    ex = m.explain(row)
    p = 1 / (1 + np.exp(-ex["posterior_log_odds"]))
    assert p == pytest.approx(m.predict_proba(row)[0], abs=1e-12)


def test_quantile_and_moments_match_numpy():
    x = rng.normal(size=1001) ** 2
    for p in (0, 0.05, 0.25, 0.5, 0.9, 1):
        assert P.quantile(x, p) == pytest.approx(np.quantile(x, p))
    d = P.describe(x)
    assert d["var"] == pytest.approx(np.var(x, ddof=1))
    assert d["mean"] == pytest.approx(np.mean(x))


def test_covariance_matches_numpy():
    X = rng.normal(size=(500, 4))
    assert np.allclose(P.covariance_matrix(X), np.cov(X, rowvar=False))
    assert np.allclose(P.correlation_matrix(X), np.corrcoef(X, rowvar=False))


def test_normal_ppf():
    p = np.linspace(0.001, 0.999, 99)
    assert np.abs(P.normal_ppf(p) - norm.ppf(p)).max() < 1e-8


def test_bayes_rule_equals_direct_count():
    f = rng.random(10000) < 0.3
    d = rng.random(10000) < np.where(f, 0.4, 0.1)
    b = P.bayes_rule(f.astype(int), d.astype(int))
    assert b["posterior"] == pytest.approx(b["direct_posterior"])


def test_mutual_information_zero_when_independent():
    a = rng.integers(0, 3, 200000)
    b = rng.integers(0, 4, 200000)
    assert P.mutual_information(a, b) < 1e-3
    assert P.mutual_information(a, a) == pytest.approx(P.entropy(np.bincount(a) / len(a)), rel=1e-9)


def test_conditional_independence_detected():
    c = rng.integers(0, 2, 100000)
    a = np.where(rng.random(100000) < 0.8, c, 1 - c)   # a and b both copy c noisily...
    b = np.where(rng.random(100000) < 0.8, c, 1 - c)   # ...so they are dependent, but independent given c
    assert P.mutual_information(a, b) > 0.05
    assert P.conditional_mutual_information(a, b, c) < 1e-3


def test_polyfit_matches_numpy_and_ridge():
    x = rng.uniform(15, 60, 40)
    t = 0.1 + 0.005 * x + rng.normal(0, 0.02, 40)
    m = PolynomialCurveFit(3).fit(x, t)
    ref = np.polyfit(scale(x), t, 3)[::-1]
    assert np.allclose(m.w, ref, atol=1e-8)
    m2 = PolynomialCurveFit(5, lam=0.1).fit(x, t)
    Phi = m2.design(x)
    sk = Ridge(alpha=0.1, fit_intercept=False).fit(Phi, t)
    assert np.allclose(m2.w, sk.coef_, atol=1e-8)


def test_metrics_match_sklearn():
    y = rng.random(3000) < 0.15
    p = np.clip(rng.normal(np.where(y, 0.6, 0.4), 0.2), 0, 1)
    assert metrics.roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert metrics.average_precision(y, p) == pytest.approx(average_precision_score(y, p))


def test_threshold_for_recall_reaches_target():
    y = rng.random(5000) < 0.2
    p = rng.random(5000) * 0.5 + y * 0.3
    t = metrics.threshold_for_recall(y, p, 0.8)
    assert metrics.scores_at(y, p, t)["recall"] >= 0.8


def test_calibration_is_monotone():
    v = pool_adjacent_violators([0.1, 0.3, 0.2, 0.5, 0.4], [1, 1, 1, 1, 1])
    assert np.all(np.diff(v) >= -1e-12)
    s = rng.random(5000)
    y = rng.random(5000) < s ** 2
    c = BinningCalibrator(20).fit(s, y)
    out = c.transform(np.sort(s))
    assert np.all(np.diff(out) >= -1e-12)
