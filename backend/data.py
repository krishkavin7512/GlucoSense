"""Import, load, view, and summarise the dataset.

Also holds the stratified train/validation/test split, written with NumPy only
so every row's destination is reproducible from the seed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .probability import describe


def load_raw() -> pd.DataFrame:
    """Read the UCI CSV exactly as published (ID column dropped)."""
    df = pd.read_csv(config.DATA_FILE)
    return df.drop(columns=["ID"], errors="ignore")


def bin_days(days) -> np.ndarray:
    """Map 0-30 'days in the last month' answers onto the 6 categorical bins."""
    return np.digitize(np.asarray(days), config.DAY_BINS[1:-1], right=False)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_split(df: pd.DataFrame, seed: int = config.SEED) -> Split:
    """Shuffle each class separately, then cut 70/15/15 so every part keeps the 13.9% base rate."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, idx in df.groupby(config.TARGET).indices.items():
        idx = rng.permutation(idx)
        n_train = int(round(len(idx) * config.SPLIT[0]))
        n_val = int(round(len(idx) * config.SPLIT[1]))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])
    take = {k: df.iloc[rng.permutation(np.concatenate(v))].reset_index(drop=True) for k, v in parts.items()}
    return Split(**take)


def summary_table(df: pd.DataFrame) -> list[dict]:
    """Per-column summary statistics."""
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        d = describe(x)
        rows.append({
            "column": col,
            "label": config.LABEL.get(col, col),
            "kind": config.FEATURE_BY_KEY.get(col, {}).get("kind", "target"),
            "count": int(len(x)),
            "missing": int(np.isnan(x).sum()),
            "unique": int(len(np.unique(x))),
            **{k: d[k] for k in ("mean", "std", "min", "q25", "median", "q75", "max", "skewness", "kurtosis")},
        })
    return rows


def preview(df: pd.DataFrame, offset: int = 0, limit: int = 25) -> dict:
    """A page of raw rows for the data viewer."""
    page = df.iloc[offset:offset + limit]
    return {
        "columns": list(df.columns),
        "rows": page.to_numpy().tolist(),
        "offset": offset,
        "limit": limit,
        "total": int(len(df)),
    }
