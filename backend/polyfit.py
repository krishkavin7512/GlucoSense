"""Polynomial curve fitting (Bishop, PRML section 1.1), written with NumPy.

Model:        y(x, w) = w0 + w1 x + w2 x^2 + ... + wM x^M
Error:        E(w) = 1/2 sum_n (y(x_n, w) - t_n)^2 + lambda/2 ||w||^2
Solution:     (Phi^T Phi + lambda I) w = Phi^T t      (Phi_nj = x_n^j)
RMS error:    E_RMS = sqrt(2 E(w*) / N)

Here x is body-mass index and t is the fraction of people at that BMI who have
diabetes or prediabetes, so the fitted curve is a population risk curve.
"""
import math

import numpy as np

BMI_RANGE = (15, 60)
X_CENTER = (BMI_RANGE[0] + BMI_RANGE[1]) / 2
X_SCALE = (BMI_RANGE[1] - BMI_RANGE[0]) / 2  # maps BMI 15..60 onto x in [-1, 1]


def scale(bmi):
    return (np.asarray(bmi, dtype=float) - X_CENTER) / X_SCALE


class PolynomialCurveFit:
    def __init__(self, degree: int, lam: float = 0.0):
        self.degree = int(degree)
        self.lam = float(lam)
        self.w = None

    def design(self, bmi) -> np.ndarray:
        x = scale(bmi)
        return np.vander(x, self.degree + 1, increasing=True)

    def fit(self, bmi, t) -> "PolynomialCurveFit":
        Phi = self.design(bmi)
        t = np.asarray(t, dtype=float)
        if self.lam > 0:
            A = Phi.T @ Phi + self.lam * np.eye(self.degree + 1)
            self.w = np.linalg.solve(A, Phi.T @ t)
        else:
            # Plain least squares; lstsq stays stable even when M+1 >= N (interpolation).
            self.w, *_ = np.linalg.lstsq(Phi, t, rcond=None)
        return self

    def predict(self, bmi) -> np.ndarray:
        return self.design(bmi) @ self.w

    def erms(self, bmi, t) -> float:
        r = self.predict(bmi) - np.asarray(t, dtype=float)
        return float(math.sqrt(np.mean(r ** 2)))


def prevalence_points(bmi, y, min_count: int = 3):
    """Group people by whole-number BMI and return (BMI, diabetes rate, group size)."""
    bmi = np.asarray(bmi, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (bmi >= BMI_RANGE[0]) & (bmi <= BMI_RANGE[1])
    b = np.round(bmi[keep]).astype(int)
    yy = y[keep]
    values = np.arange(BMI_RANGE[0], BMI_RANGE[1] + 1)
    counts = np.bincount(b - BMI_RANGE[0], minlength=len(values))
    pos = np.bincount(b - BMI_RANGE[0], weights=yy, minlength=len(values))
    m = counts >= min_count
    return values[m].astype(float), (pos[m] / counts[m]), counts[m]


def sweep_degree(train, test, degrees=range(0, 13), lam=0.0) -> dict:
    rows = []
    for M in degrees:
        model = PolynomialCurveFit(M, lam).fit(train[0], train[1])
        rows.append({"degree": M, "train_rms": model.erms(train[0], train[1]),
                     "test_rms": model.erms(test[0], test[1]),
                     "weights": model.w.tolist()})
    return {"lambda": lam, "rows": rows}


def sweep_lambda(train, test, degree=9, log_lambdas=np.linspace(-20, 2, 45)) -> dict:
    rows = []
    for ll in log_lambdas:
        model = PolynomialCurveFit(degree, math.exp(ll)).fit(train[0], train[1])
        rows.append({"log_lambda": float(ll), "train_rms": model.erms(train[0], train[1]),
                     "test_rms": model.erms(test[0], test[1])})
    return {"degree": degree, "rows": rows}
