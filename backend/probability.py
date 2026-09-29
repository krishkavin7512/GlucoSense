"""Probability and statistics toolkit, written from the definitions with NumPy.

Covers: discrete random variables and the sum/product rules, Bayes' rule,
independence and conditional independence (mutual information, chi-square),
continuous random variables (densities, quantiles, mean, variance), maximum
likelihood density fits, expectation, and covariance.
"""
import math

import numpy as np
from scipy.stats import chi2 as _chi2  # only used for the chi-square tail probability

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# Continuous random variables: moments and quantiles
# --------------------------------------------------------------------------- #
def quantile(x, p):
    """Sample quantile by linear interpolation between order statistics.

    For sorted x_(0..n-1) the position of probability p is h = (n-1)p, and
    Q(p) = x_(floor h) + (h - floor h) * (x_(floor h + 1) - x_(floor h)).
    """
    xs = np.sort(np.asarray(x, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=float))
    h = (len(xs) - 1) * p
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, len(xs) - 1)
    q = xs[lo] + (h - lo) * (xs[hi] - xs[lo])
    return q if q.size > 1 else float(q[0])


def describe(x) -> dict:
    """Mean, variance, standard deviation, quantiles and shape of a sample."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mean = x.sum() / n
    dev = x - mean
    var = (dev ** 2).sum() / (n - 1)          # unbiased sample variance
    std = math.sqrt(var)
    m2 = (dev ** 2).mean()
    skew = (dev ** 3).mean() / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = (dev ** 4).mean() / m2 ** 2 - 3.0 if m2 > 0 else 0.0
    q = quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "n": int(n), "mean": float(mean), "var": float(var), "std": float(std),
        "min": float(q[0]), "q05": float(q[1]), "q25": float(q[2]), "median": float(q[3]),
        "q75": float(q[4]), "q95": float(q[5]), "max": float(q[6]),
        "iqr": float(q[4] - q[2]), "skewness": float(skew), "kurtosis": float(kurt),
    }


def ecdf(x):
    """Empirical CDF F(t) = (1/n) * #{x_i <= t}, returned at each distinct value."""
    xs = np.sort(np.asarray(x, dtype=float))
    values, counts = np.unique(xs, return_counts=True)
    return values, np.cumsum(counts) / len(xs)


def normal_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out


# --------------------------------------------------------------------------- #
# Probability densities and maximum-likelihood fits
# --------------------------------------------------------------------------- #
def gaussian_pdf(x, mu, var):
    x = np.asarray(x, dtype=float)
    return np.exp(-(x - mu) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)


def lognormal_pdf(x, mu, var):
    """Density of X when ln X ~ N(mu, var): includes the 1/x Jacobian."""
    x = np.asarray(x, dtype=float)
    return gaussian_pdf(np.log(x), mu, var) / x


def fit_gaussian(x) -> dict:
    """MLE: mu = mean(x), sigma^2 = mean((x - mu)^2) (the biased estimator)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    ll = float(np.log(gaussian_pdf(x, mu, var)).sum())
    return {"family": "Gaussian", "mu": float(mu), "var": float(var), "loglik": ll,
            "aic": 2 * 2 - 2 * ll, "var_unbiased": float(var * len(x) / (len(x) - 1))}


def fit_lognormal(x) -> dict:
    """MLE of a log-normal: fit a Gaussian to ln x."""
    x = np.asarray(x, dtype=float)
    lx = np.log(x)
    mu = lx.mean()
    var = ((lx - mu) ** 2).mean()
    ll = float(np.log(lognormal_pdf(x, mu, var)).sum())
    return {"family": "Log-normal", "mu": float(mu), "var": float(var), "loglik": ll,
            "aic": 2 * 2 - 2 * ll, "mean": float(math.exp(mu + var / 2)),
            "median": float(math.exp(mu))}


# --------------------------------------------------------------------------- #
# Expectation and covariance
# --------------------------------------------------------------------------- #
def covariance_matrix(X):
    """Sample covariance: Cov = (X - mean)^T (X - mean) / (n - 1)."""
    X = np.asarray(X, dtype=float)
    D = X - X.mean(axis=0)
    return D.T @ D / (len(X) - 1)


def correlation_matrix(X):
    C = covariance_matrix(X)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def total_expectation(x, y) -> dict:
    """Check the law of total expectation and the law of total variance.

    E[X] = sum_y P(y) E[X | y]
    Var[X] = E[Var(X | Y)] + Var(E[X | Y])
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    p = np.array([(y == c).mean() for c in classes])
    cond_mean = np.array([x[y == c].mean() for c in classes])
    cond_var = np.array([x[y == c].var() for c in classes])
    e_total = float((p * cond_mean).sum())
    within = float((p * cond_var).sum())
    between = float((p * (cond_mean - e_total) ** 2).sum())
    return {
        "classes": classes.tolist(), "p": p.tolist(), "cond_mean": cond_mean.tolist(),
        "cond_var": cond_var.tolist(), "e_direct": float(x.mean()), "e_total": e_total,
        "var_direct": float(x.var()), "var_within": within, "var_between": between,
        "var_total": within + between,
    }


# --------------------------------------------------------------------------- #
# Discrete random variables: joint tables, sum rule, product rule, Bayes' rule
# --------------------------------------------------------------------------- #
def joint_table(a, b) -> dict:
    """Joint distribution p(A, B) of two discrete variables plus its marginals.

    Sum rule:     p(A) = sum_B p(A, B)
    Product rule: p(A, B) = p(B | A) p(A)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    av, ai = np.unique(a, return_inverse=True)
    bv, bi = np.unique(b, return_inverse=True)
    counts = np.zeros((len(av), len(bv)))
    np.add.at(counts, (ai, bi), 1)
    joint = counts / counts.sum()
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    b_given_a = joint / pa[:, None]
    product_check = float(np.abs(b_given_a * pa[:, None] - joint).max())
    return {
        "a_values": av.tolist(), "b_values": bv.tolist(), "counts": counts.astype(int).tolist(),
        "joint": joint.tolist(), "p_a": pa.tolist(), "p_b": pb.tolist(),
        "b_given_a": b_given_a.tolist(), "product_rule_max_error": product_check,
    }


def bayes_rule(feature, target, value=1) -> dict:
    """P(D | F=value) from the prior, the likelihoods and the evidence.

    P(D | F) = P(F | D) P(D) / P(F),  with  P(F) = P(F | D) P(D) + P(F | not D) P(not D).
    """
    f = np.asarray(feature) == value
    d = np.asarray(target) == 1
    prior = d.mean()
    like_d = f[d].mean()
    like_nd = f[~d].mean()
    evidence = like_d * prior + like_nd * (1 - prior)
    posterior = like_d * prior / evidence
    posterior_not = (1 - like_d) * prior / (1 - evidence)
    return {
        "prior": float(prior), "likelihood_d": float(like_d), "likelihood_not_d": float(like_nd),
        "evidence": float(evidence), "posterior": float(posterior),
        "posterior_if_absent": float(posterior_not),
        "direct_posterior": float(d[f].mean()),  # counted straight from the data, must match
        "likelihood_ratio": float(like_d / like_nd), "lift": float(posterior / prior),
        "n_with_feature": int(f.sum()),
    }


# --------------------------------------------------------------------------- #
# Independence and conditional independence
# --------------------------------------------------------------------------- #
def entropy(p):
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(a, b) -> float:
    """I(A;B) = sum p(a,b) log2[p(a,b) / (p(a)p(b))]  (0 exactly when A and B are independent)."""
    t = joint_table(a, b)
    joint = np.array(t["joint"])
    outer = np.outer(t["p_a"], t["p_b"])
    m = joint > 0
    return float((joint[m] * np.log2(joint[m] / outer[m])).sum())


def conditional_mutual_information(a, b, c) -> float:
    """I(A;B | C) = sum_c p(c) I(A;B | C=c)  (0 exactly when A and B are independent given C)."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    total = 0.0
    for v in np.unique(c):
        m = c == v
        total += m.mean() * mutual_information(a[m], b[m])
    return float(total)


def chi_square_independence(a, b) -> dict:
    """Pearson chi-square test of H0: A and B are independent, plus Cramer's V effect size."""
    t = joint_table(a, b)
    obs = np.array(t["counts"], dtype=float)
    n = obs.sum()
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    stat = float(((obs - expected) ** 2 / expected).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    k = min(obs.shape) - 1
    return {
        "chi2": stat, "dof": int(dof), "p_value": float(_chi2.sf(stat, dof)),
        "cramers_v": float(math.sqrt(stat / (n * k))) if k > 0 else 0.0,
    }
