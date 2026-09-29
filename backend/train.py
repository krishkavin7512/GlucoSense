"""Training pipeline: statistics -> Naive Bayes -> curve fitting -> evaluation -> artifacts.

Run with:  python -m backend.train
Writes artifacts/model.json, artifacts/report.json, artifacts/poly_data.json, artifacts/graphs.json
"""
import json
import math
import time

import numpy as np
import pandas as pd

from . import config, metrics
from . import probability as P
from .data import bin_days, load_raw, stratified_split, summary_table
from .calibration import BinningCalibrator
from .naive_bayes import MixedNaiveBayes, all_categorical_features, encode_matrix, BMI_EDGES, _levels
from .polyfit import PolynomialCurveFit, prevalence_points, sweep_degree, sweep_lambda

Y = config.TARGET


def log(msg):
    print(f"[train] {msg}", flush=True)


def _r(x, nd=6):
    """Round floats recursively so the JSON stays small."""
    if isinstance(x, float):
        # significant digits, so tiny values such as a 1e-15 agreement check survive
        return float(f"{x:.{nd + 1}g}") if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, (np.floating,)):
        return _r(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def discrete_view(df: pd.DataFrame, key: str) -> np.ndarray:
    """Every column as a small discrete variable (BMI and day-counts binned) for MI / chi-square."""
    f = config.FEATURE_BY_KEY[key]
    if f["kind"] == "continuous":
        return np.digitize(df[key].to_numpy(), BMI_EDGES)
    if f["kind"] == "days":
        return bin_days(df[key].to_numpy())
    return df[key].to_numpy()


# --------------------------------------------------------------------------- #
# Probability and statistics
# --------------------------------------------------------------------------- #
EVENTS = [  # (id, label, function(df) -> boolean mask)
    ("HighBP", "High blood pressure", lambda d: d.HighBP == 1),
    ("HighChol", "High cholesterol", lambda d: d.HighChol == 1),
    ("Obese", "BMI 30 or more", lambda d: d.BMI >= 30),
    ("PoorHealth", "Fair or poor general health", lambda d: d.GenHlth >= 4),
    ("Age65", "Aged 65 or older", lambda d: d.Age >= 10),
    ("HeartDiseaseorAttack", "Heart disease / attack", lambda d: d.HeartDiseaseorAttack == 1),
    ("DiffWalk", "Difficulty walking", lambda d: d.DiffWalk == 1),
    ("Stroke", "Past stroke", lambda d: d.Stroke == 1),
    ("Inactive", "No physical activity", lambda d: d.PhysActivity == 0),
    ("Smoker", "Smoker (100+ cigarettes)", lambda d: d.Smoker == 1),
    ("HvyAlcoholConsump", "Heavy drinker", lambda d: d.HvyAlcoholConsump == 1),
    ("NoVeggies", "No daily vegetables", lambda d: d.Veggies == 0),
]


def compute_statistics(df: pd.DataFrame) -> dict:
    y = df[Y].to_numpy()
    out = {"n_rows": int(len(df)), "n_features": len(config.FEATURE_KEYS),
           "duplicates": int(df.duplicated().sum()), "missing": int(df.isna().sum().sum()),
           "positives": int(y.sum()), "base_rate": float(y.mean())}
    out["summary"] = summary_table(df)

    # Bayes' rule for a set of risk events
    bayes = []
    for eid, label, fn in EVENTS:
        b = P.bayes_rule(fn(df).to_numpy().astype(int), y)
        bayes.append({"id": eid, "label": label, **b})
    out["bayes"] = sorted(bayes, key=lambda r: -r["posterior"])

    # Sum / product rule on a 2x2 joint table
    out["joint_highbp"] = P.joint_table(df.HighBP.to_numpy(), y)

    # Independence: every feature vs the target, and every feature pair
    keys = config.FEATURE_KEYS
    disc = {k: discrete_view(df, k) for k in keys}
    target_tests = []
    for k in keys:
        chi = P.chi_square_independence(disc[k], y)
        target_tests.append({"key": k, "label": config.LABEL[k], "mi": P.mutual_information(disc[k], y), **chi})
    out["target_independence"] = sorted(target_tests, key=lambda r: -r["mi"])

    n = len(keys)
    mi = np.zeros((n, n))
    cmi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi[i, j] = mi[j, i] = P.mutual_information(disc[keys[i]], disc[keys[j]])
            cmi[i, j] = cmi[j, i] = P.conditional_mutual_information(disc[keys[i]], disc[keys[j]], y)
    out["mi_matrix"] = {"keys": keys, "labels": [config.LABEL[k] for k in keys], "mi": mi, "cmi": cmi}
    pairs = [{"a": keys[i], "b": keys[j], "mi": mi[i, j], "cmi": cmi[i, j]}
             for i in range(n) for j in range(i + 1, n)]
    out["top_dependent_pairs"] = sorted(pairs, key=lambda r: -r["cmi"])[:12]

    # Continuous variable: BMI
    bmi = df.BMI.to_numpy(dtype=float)
    cont = {"overall": P.describe(bmi), "by_class": {}}
    probs = np.linspace(0, 1, 101)
    cont["quantiles"] = {"p": probs, "all": P.quantile(bmi, probs)}
    for c in (0, 1):
        cont["by_class"][str(c)] = P.describe(bmi[y == c])
        cont["quantiles"][str(c)] = P.quantile(bmi[y == c], probs)
    cont["fits"] = {"gaussian": P.fit_gaussian(bmi), "lognormal": P.fit_lognormal(bmi),
                    "by_class": {str(c): {"gaussian": P.fit_gaussian(bmi[y == c]),
                                          "lognormal": P.fit_lognormal(bmi[y == c])} for c in (0, 1)}}
    grid = np.linspace(12, 70, 233)
    edges = np.arange(12, 71, 1.0)
    hist = {}
    for c in (0, 1):
        h, _ = np.histogram(bmi[y == c], bins=edges, density=True)
        f = cont["fits"]["by_class"][str(c)]
        hist[str(c)] = {"density": h,
                        "lognormal_pdf": P.lognormal_pdf(grid, f["lognormal"]["mu"], f["lognormal"]["var"]),
                        "gaussian_pdf": P.gaussian_pdf(grid, f["gaussian"]["mu"], f["gaussian"]["var"])}
    cont["histogram"] = {"edges": edges, "grid": grid, **hist}
    ev, ef = P.ecdf(bmi)
    cont["ecdf"] = {"x": ev, "F": ef}
    qp = (np.arange(1, 200) - 0.5) / 199
    rng = np.random.default_rng(config.SEED)
    sample = np.log(rng.choice(bmi, 20000, replace=False))
    cont["qq"] = {"theoretical": P.normal_ppf(qp),
                  "sample": (P.quantile(sample, qp) - sample.mean()) / sample.std(),
                  "raw_sample": (P.quantile(bmi, qp) - bmi.mean()) / bmi.std()}
    cont["total_expectation"] = P.total_expectation(bmi, y)
    out["bmi"] = cont
    out["days"] = {k: P.describe(df[k].to_numpy()) for k in ("MentHlth", "PhysHlth")}

    # Covariance / correlation of every column
    cols = keys + [Y]
    Xall = df[cols].to_numpy(dtype=float)
    out["covariance"] = {"keys": cols, "labels": [config.LABEL[k] for k in cols],
                         "cov": P.covariance_matrix(Xall), "corr": P.correlation_matrix(Xall)}

    # Conditional prevalence by ordered groups
    def rate_by(col, labels):
        g = df.groupby(col)[Y].agg(["mean", "count"])
        return {"labels": labels, "rate": g["mean"].tolist(), "count": g["count"].tolist()}
    out["rate_by"] = {"Age": rate_by("Age", config.AGE_LABELS),
                      "GenHlth": rate_by("GenHlth", config.GENHLTH_LABELS),
                      "Income": rate_by("Income", config.INCOME_LABELS),
                      "Education": rate_by("Education", config.EDU_LABELS)}
    return out


# --------------------------------------------------------------------------- #
# Naive Bayes training and evaluation
# --------------------------------------------------------------------------- #
def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(len(idx)) % k
    return folds


def roc_points(y, p, max_points=400):
    fpr, tpr, thr = metrics.roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return {"fpr": fpr[keep], "tpr": tpr[keep]}


def pr_points(y, p, max_points=400):
    prec, rec, thr = metrics.pr_curve(y, p)
    keep = np.unique(np.linspace(0, len(prec) - 1, min(max_points, len(prec))).astype(int))
    return {"precision": prec[keep], "recall": rec[keep]}


def train_models(split) -> dict:
    Xtr, ytr = split.train, split.train[Y].to_numpy()
    Xva, yva = split.val, split.val[Y].to_numpy()
    Xte, yte = split.test, split.test[Y].to_numpy()
    rep = {"sizes": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)}}

    # 1) Which density for BMI? Compare validation log-likelihood of Gaussian vs log-normal.
    dens = {}
    for fam in ("gaussian", "lognormal"):
        m = MixedNaiveBayes(continuous=fam).fit(Xtr, ytr)
        ll = m.feature_log_likelihoods(Xva)[:, config.FEATURE_KEYS.index("BMI"), :]
        ll_true = np.where(yva == 1, ll[:, 1], ll[:, 0]).mean()
        p = m.predict_proba(Xva)
        dens[fam] = {"val_bmi_loglik": float(ll_true), "val_auc": metrics.roc_auc(yva, p),
                     "val_log_loss": metrics.log_loss(yva, p)}
    rep["bmi_density_choice"] = dens
    family = "lognormal" if dens["lognormal"]["val_bmi_loglik"] > dens["gaussian"]["val_bmi_loglik"] else "gaussian"
    log(f"BMI density: {family}  {dens}")

    # 2) Laplace smoothing alpha: validation curve at two training-set sizes.
    alphas = [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
    rng = np.random.default_rng(config.SEED)
    small = Xtr.iloc[rng.choice(len(Xtr), 1000, replace=False)]
    sweep = {"alphas": alphas, "full": [], "small": []}
    for a in alphas:
        for name, data in (("full", Xtr), ("small", small)):
            m = MixedNaiveBayes(alpha=a, continuous=family).fit(data, data[Y])
            p = m.predict_proba(Xva)
            sweep[name].append({"alpha": a, "log_loss": metrics.log_loss(yva, p), "auc": metrics.roc_auc(yva, p)})
    rep["alpha_sweep"] = sweep
    # Rank quality (AUC) is what a screener needs. Every alpha within 1e-4 of the best
    # validation AUC is equivalent, so take the one closest to Laplace's classic alpha = 1.
    best_auc = max(r["auc"] for r in sweep["full"])
    tied = [r["alpha"] for r in sweep["full"] if r["auc"] >= best_auc - 1e-4]
    best_alpha = min(tied, key=lambda a: abs(math.log(a)))
    rep["alpha"] = best_alpha
    log(f"alpha chosen: {best_alpha}")

    # 3) Final model on the training split.
    model = MixedNaiveBayes(alpha=best_alpha, continuous=family).fit(Xtr, ytr)
    pva = model.predict_proba(Xva)
    pte = model.predict_proba(Xte)

    # Calibrate raw scores into observed frequencies (validation set only).
    calib = BinningCalibrator(30).fit(pva, yva)
    cte = calib.transform(pte)
    rep["calibrator"] = calib.to_dict()

    # Screening operating point: highest raw threshold with >= 80% recall on validation.
    thr = metrics.threshold_for_recall(yva, pva, config.TARGET_RECALL)
    rep["threshold"] = thr
    rep["threshold_calibrated"] = float(calib.transform([thr])[0])
    rep["val"] = metrics.summary(yva, pva, thr)
    rep["test"] = metrics.summary(yte, pte, thr)
    rep["test"]["log_loss_calibrated"] = metrics.log_loss(yte, cte)
    rep["test"]["brier_calibrated"] = metrics.brier(yte, cte)
    rep["test_at_half"] = metrics.scores_at(yte, pte, 0.5)
    log(f"threshold {thr:.4f} (calibrated {rep['threshold_calibrated']:.3f})  test AUC {rep['test']['roc_auc']:.4f}  "
        f"recall {rep['test']['recall']:.3f}  precision {rep['test']['precision']:.3f}  "
        f"log-loss raw {rep['test']['log_loss']:.3f} -> calibrated {rep['test']['log_loss_calibrated']:.3f}")

    # The sensitivity slider in the app: one operating point per target recall.
    table = []
    for target in np.round(np.arange(0.50, 0.96, 0.01), 2):
        t = metrics.threshold_for_recall(yva, pva, float(target))
        s = metrics.scores_at(yte, pte, t)
        table.append({"target_recall": float(target), "threshold": t,
                      "threshold_calibrated": float(calib.transform([t])[0]),
                      **{k: s[k] for k in ("recall", "precision", "specificity", "flagged_rate", "tp", "fp", "fn", "tn")}})
    rep["operating_points"] = table

    # Curves on the test set
    rep["roc"] = {"ours": roc_points(yte, pte)}
    rep["pr"] = {"ours": pr_points(yte, pte)}
    rep["calibration"] = {"raw": metrics.calibration_bins(yte, pte, 12),
                          "calibrated": metrics.calibration_bins(yte, cte, 12)}
    grid = np.round(np.linspace(0.005, 0.5, 100), 4)
    rep["threshold_curve"] = {"threshold": grid,
                              **{k: [metrics.scores_at(yte, cte, t)[k] for t in grid]
                                 for k in ("precision", "recall", "f1", "specificity", "flagged_rate")}}
    edges = np.linspace(0, 0.6, 41)
    rep["score_hist"] = {"edges": edges,
                         "neg": np.histogram(cte[yte == 0], bins=edges)[0],
                         "pos": np.histogram(cte[yte == 1], bins=edges)[0]}

    # Naive Bayes "weights": log-likelihood ratio of every answer level
    llr = []
    for f in config.FEATURES:
        p = model.params[f["key"]]
        if p["type"] == "categorical":
            lp = np.array(p["log_prob"])
            names = f.get("options") or (config.DAY_BIN_LABELS if f["kind"] == "days" else None)
            for k, lev in enumerate(_levels(f)):
                llr.append({"key": f["key"], "feature": f["label"], "level": names[k] if names else str(lev),
                            "llr": float(lp[1, k] - lp[0, k])})
    rep["llr"] = llr

    # 4) Validation against scikit-learn
    from sklearn.naive_bayes import CategoricalNB, GaussianNB
    cat_feats = all_categorical_features()
    ours_cat = MixedNaiveBayes(cat_feats, alpha=best_alpha).fit(Xtr, ytr)
    Etr, Ete = encode_matrix(cat_feats, Xtr), encode_matrix(cat_feats, Xte)
    sk = CategoricalNB(alpha=best_alpha, min_categories=[len(_levels(f)) for f in cat_feats]).fit(Etr, ytr)
    p_ours_cat = ours_cat.predict_proba(Xte)
    p_sk_cat = sk.predict_proba(Ete)[:, 1]
    gnb = GaussianNB().fit(Xtr[config.FEATURE_KEYS], ytr)
    p_gnb = gnb.predict_proba(Xte[config.FEATURE_KEYS])[:, 1]
    rng = np.random.default_rng(1)
    pick = rng.choice(len(Xte), 1500, replace=False)
    rep["sklearn"] = {
        "max_abs_diff_categorical": float(np.abs(p_ours_cat - p_sk_cat).max()),
        "auc_ours_mixed": rep["test"]["roc_auc"],
        "auc_ours_categorical": metrics.roc_auc(yte, p_ours_cat),
        "auc_sklearn_categorical": metrics.roc_auc(yte, p_sk_cat),
        "auc_sklearn_gaussian": metrics.roc_auc(yte, p_gnb),
        "log_loss_ours_mixed": rep["test"]["log_loss"],
        "log_loss_sklearn_gaussian": metrics.log_loss(yte, p_gnb),
        "agreement_sample": {"ours": p_ours_cat[pick], "sklearn": p_sk_cat[pick]},
    }
    rep["roc"]["sklearn_categorical"] = roc_points(yte, p_sk_cat)
    rep["roc"]["sklearn_gaussian"] = roc_points(yte, p_gnb)
    log(f"sklearn check: max |p_ours - p_sklearn| = {rep['sklearn']['max_abs_diff_categorical']:.2e}; "
        f"GaussianNB AUC {rep['sklearn']['auc_sklearn_gaussian']:.4f}")

    # 5) Learning curve and 5-fold cross-validation
    sizes = [200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, len(Xtr)]
    lc = []
    for s in sizes:
        tr_auc, va_auc, tr_ll, va_ll = [], [], [], []
        for rep_i in range(3 if s < len(Xtr) else 1):
            idx = np.random.default_rng(100 + rep_i).choice(len(Xtr), s, replace=False)
            sub = Xtr.iloc[idx]
            m = MixedNaiveBayes(alpha=best_alpha, continuous=family).fit(sub, sub[Y])
            ptr = m.predict_proba(sub)
            pv = m.predict_proba(Xva)
            tr_auc.append(metrics.roc_auc(sub[Y], ptr))
            va_auc.append(metrics.roc_auc(yva, pv))
            tr_ll.append(metrics.log_loss(sub[Y], ptr))
            va_ll.append(metrics.log_loss(yva, pv))
        lc.append({"size": s, "train_auc": float(np.mean(tr_auc)), "val_auc": float(np.mean(va_auc)),
                   "train_log_loss": float(np.mean(tr_ll)), "val_log_loss": float(np.mean(va_ll))})
    rep["learning_curve"] = lc

    pool = pd.concat([Xtr, Xva], ignore_index=True)
    ypool = pool[Y].to_numpy()
    folds = stratified_folds(ypool, 5, config.SEED)
    cv = []
    for k in range(5):
        m = MixedNaiveBayes(alpha=best_alpha, continuous=family).fit(pool[folds != k], ypool[folds != k])
        p = m.predict_proba(pool[folds == k])
        cv.append({"fold": k + 1, "auc": metrics.roc_auc(ypool[folds == k], p),
                   "recall_at_thr": metrics.scores_at(ypool[folds == k], p, thr)["recall"]})
    rep["cv"] = cv
    log(f"5-fold CV AUC: {np.mean([c['auc'] for c in cv]):.4f} +- {np.std([c['auc'] for c in cv]):.4f}")
    return model, rep


# --------------------------------------------------------------------------- #
# Polynomial curve fitting
# --------------------------------------------------------------------------- #
POLY_SIZES = [300, 1000, 3000, 10000, 30000]


def train_polyfit(split) -> tuple[dict, dict]:
    tr, va, te = split.train, split.val, split.test
    test_pts = prevalence_points(te.BMI, te[Y], min_count=25)
    val_pts = prevalence_points(va.BMI, va[Y], min_count=25)
    samples = {}
    for s in POLY_SIZES:
        idx = np.random.default_rng(7).choice(len(tr), s, replace=False)
        samples[str(s)] = prevalence_points(tr.BMI.iloc[idx], tr[Y].iloc[idx], min_count=3)
    samples["all"] = prevalence_points(tr.BMI, tr[Y], min_count=25)
    poly_data = {"test": test_pts, "val": val_pts, "samples": samples, "default_size": "1000"}

    base = samples["1000"]
    rep = {
        "degree_sweep": sweep_degree(base, test_pts),
        "degree_sweep_reg": sweep_degree(base, test_pts, lam=math.exp(-6)),
        "lambda_sweep": sweep_lambda(base, test_pts, degree=9),
        "size_effect": {},
        "example_fits": {},
        "sample_points": {"train": base, "test": test_pts},
        "all_points": samples["all"],
    }
    grid = np.linspace(15, 60, 181)
    for M in (0, 1, 3, 9):
        m = PolynomialCurveFit(M).fit(*base[:2])
        rep["example_fits"][str(M)] = {"curve": m.predict(grid), "weights": m.w,
                                       "train_rms": m.erms(*base[:2]), "test_rms": m.erms(*test_pts[:2])}
    for s in ("300", "3000", "all"):
        m = PolynomialCurveFit(9).fit(*samples[s][:2])
        rep["size_effect"][s] = {"curve": m.predict(grid), "test_rms": m.erms(*test_pts[:2]),
                                 "n_points": int(len(samples[s][0]))}
    rep["grid"] = grid

    # Model selection on the validation curve, using all training people.
    best = None
    for M in range(0, 11):
        for ll in [-30, -12, -9, -6, -3]:
            m = PolynomialCurveFit(M, math.exp(ll)).fit(*samples["all"][:2])
            v = m.erms(*val_pts[:2])
            if best is None or v < best[0] - 1e-6:
                best = (v, M, ll, m)
    v, M, ll, m = best
    rep["selected"] = {"degree": M, "log_lambda": ll, "val_rms": v, "test_rms": m.erms(*test_pts[:2]),
                       "weights": m.w, "curve": m.predict(grid)}
    log(f"polynomial risk curve: M={M}, ln(lambda)={ll}, val RMS={v:.4f}, test RMS={rep['selected']['test_rms']:.4f}")
    return rep, poly_data


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    config.ARTIFACTS.mkdir(exist_ok=True)
    log("loading data")
    df = load_raw()
    split = stratified_split(df)
    log(f"rows {len(df):,}  train/val/test {len(split.train):,}/{len(split.val):,}/{len(split.test):,}")

    log("computing probability statistics")
    stats = compute_statistics(df)
    log("training Naive Bayes")
    model, nb_rep = train_models(split)
    log("fitting polynomial curves")
    poly_rep, poly_data = train_polyfit(split)

    report = _r({"stats": stats, "nb": nb_rep, "poly": poly_rep,
                 "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    (config.ARTIFACTS / "model.json").write_text(json.dumps(_r({
        "naive_bayes": model.to_dict(), "threshold": nb_rep["threshold"],
        "calibrator": nb_rep["calibrator"], "operating_points": nb_rep["operating_points"],
        "default_recall": config.TARGET_RECALL,
        "polynomial": {"degree": poly_rep["selected"]["degree"], "log_lambda": poly_rep["selected"]["log_lambda"],
                       "weights": poly_rep["selected"]["weights"]},
    }, 12)))
    (config.ARTIFACTS / "report.json").write_text(json.dumps(report))
    (config.ARTIFACTS / "poly_data.json").write_text(json.dumps(_r(poly_data)))

    from .charts import build_all
    graphs = build_all(report)
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(graphs))
    log(f"{len(graphs)} graphs written; done in {time.time() - t0:.1f}s")


def rebuild_charts():
    """Regenerate graphs.json from an existing report.json (no retraining)."""
    from .charts import build_all
    report = json.loads((config.ARTIFACTS / "report.json").read_text())
    graphs = build_all(report)
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(graphs))
    log(f"{len(graphs)} graphs rebuilt")


if __name__ == "__main__":
    import sys
    rebuild_charts() if "--charts-only" in sys.argv else main()
