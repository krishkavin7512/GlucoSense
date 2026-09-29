"""Classification metrics written from their definitions (no sklearn.metrics)."""
import numpy as np


def confusion(y, pred) -> dict:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def scores_at(y, p, threshold) -> dict:
    c = confusion(y, np.asarray(p) >= threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**c, "threshold": float(threshold), "recall": recall, "precision": precision,
            "specificity": specificity, "f1": f1,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "balanced_accuracy": (recall + specificity) / 2,
            "flagged_rate": (tp + fp) / (tp + tn + fp + fn)}


def roc_curve(y, p):
    """False-positive rate and true-positive rate at every distinct score, highest first."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    distinct = np.where(np.diff(p_sorted))[0]
    idx = np.r_[distinct, len(p_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (idx + 1) - tps
    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]
    thr = np.r_[np.inf, p_sorted[idx]]
    return fpr, tpr, thr


def auc(x, y) -> float:
    """Area under a curve by the trapezoid rule."""
    x, y = np.asarray(x), np.asarray(y)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def roc_auc(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return auc(fpr, tpr)


def pr_curve(y, p):
    """Precision and recall at every distinct score threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    distinct = np.where(np.diff(p_sorted))[0]
    idx = np.r_[distinct, len(p_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (idx + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return precision, recall, p_sorted[idx]


def average_precision(y, p) -> float:
    """AP = sum_k (R_k - R_{k-1}) P_k."""
    precision, recall, _ = pr_curve(y, p)
    r = np.r_[0, recall]
    return float(np.sum((r[1:] - r[:-1]) * precision))


def log_loss(y, p, eps=1e-15) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def calibration_bins(y, p, n_bins=10) -> dict:
    """Mean predicted probability vs observed frequency in equal-count bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    return {
        "mean_predicted": [float(p[c].mean()) for c in chunks],
        "observed": [float(y[c].mean()) for c in chunks],
        "count": [int(len(c)) for c in chunks],
    }


def threshold_for_recall(y, p, target) -> float:
    """Highest threshold whose recall is still >= target (fewest false alarms at that recall)."""
    precision, recall, thr = pr_curve(y, p)
    ok = np.where(recall >= target)[0]
    return float(thr[ok[0]])


def summary(y, p, threshold) -> dict:
    return {**scores_at(y, p, threshold), "roc_auc": roc_auc(y, p),
            "average_precision": average_precision(y, p), "log_loss": log_loss(y, p),
            "brier": brier(y, p), "base_rate": float(np.mean(y))}
