"""
Fraud detection: rule features + LightGBM, with temporal validation.

Explainability & ops:
  - SHAP per-alert reason codes for analyst trust / compliance
  - Weekly PR-AUC drift monitoring with scheduled retrain
  - Six raw rule guardrails running in parallel with the model
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 1. Load & parse
# ---------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(
        path,
        dtype={
            "transaction_id": "string",
            "card_id": "string",
            "merchant_name": "string",
            "merchant_category": "string",
            "channel": "string",
            "cardholder_country": "string",
            "merchant_country": "string",
            "device_id": "string",
            "ip_address": "string",
        },
        parse_dates=["timestamp"],
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def shrink(df):
    """Reduce memory footprint for large files."""
    df["amount"] = pd.to_numeric(df["amount"], downcast="float")
    for c in [
        "merchant_category",
        "channel",
        "cardholder_country",
        "merchant_country",
    ]:
        df[c] = df[c].astype("category")
    return df

# ---------------------------------------------------------------------------
# 2. Feature engineering
#    Every feature must be computable using ONLY past+current rows of a card,
#    so it generalizes to streaming / future data with no leakage.
# ---------------------------------------------------------------------------
def add_time_features(df):
    g = df.copy()
    ts = g["timestamp"]
    g["hour_of_day"] = ts.dt.hour
    g["day_of_week"] = ts.dt.dayofweek
    g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)
    g["is_night"] = ((g["hour_of_day"] >= 23) | (g["hour_of_day"] <= 5)).astype(int)
    g["date"] = ts.dt.date
    return g


def add_amount_features(df):
    g = df.copy()
    a = g["amount"]
    g["amt_log"] = np.log1p(a)
    g["amt_is_round"] = (a == a.round(0)).astype(int)
    g["amt_card_test"] = a.isin([1.0, 5.0, 10.0]).astype(int)
    g["amt_just_below_100"] = ((a >= 99.50) & (a < 100.00)).astype(int)
    g["amt_just_below_500"] = ((a >= 499.50) & (a < 500.00)).astype(int)
    return g


def add_card_velocity_features(df):
    """Per-card trailing velocity & spend. Uses time-based rolling windows."""
    g = df.sort_values("timestamp").copy()
    idx = g.set_index("timestamp").groupby("card_id")["amount"]

    g["tx_1min"] = idx.transform(lambda s: s.rolling("1min").count()).values
    g["tx_5min"] = idx.transform(lambda s: s.rolling("5min").count()).values
    g["tx_1h"] = idx.transform(lambda s: s.rolling("1h").count()).values
    g["tx_24h"] = idx.transform(lambda s: s.rolling("24h").count()).values
    g["spend_24h"] = idx.transform(lambda s: s.rolling("24h").sum()).values

    grp = g.groupby("card_id")
    g["time_since_last"] = grp["timestamp"].diff().dt.total_seconds().fillna(1e7)
    g["tx_of_day"] = g.groupby(["card_id", "date"]).cumcount() + 1
    return g


def add_change_features(df):
    """Did merchant/device/country change vs the card's previous tx?"""
    g = df.sort_values("timestamp").copy()
    grp = g.groupby("card_id")
    g["merchant_change"] = (g["merchant_name"] != grp["merchant_name"].shift()).fillna(False).astype(int)
    g["device_change"] = (
        g["device_id"].fillna("NA") != grp["device_id"].shift().fillna("NA")
    ).fillna(False).astype(int)
    prev_country = grp["merchant_country"].shift()
    g["country_hop"] = (
        (g["merchant_country"] != prev_country) & prev_country.notna()
    ).fillna(False).astype(int)
    minutes = g["time_since_last"] / 60.0
    g["fast_country_hop"] = ((g["country_hop"] == 1) & (minutes < 60)).fillna(False).astype(int)
    g["cross_border"] = (
        g["cardholder_country"].astype(str) != g["merchant_country"].astype(str)
    ).fillna(False).astype(int)
    return g


def _rolling_distinct_count(times_ns, codes, window_ns):
    """Exact distinct count in trailing window (t-W, t], O(n) per group.
    times_ns must be sorted ascending."""
    n = len(times_ns)
    out = np.empty(n, dtype=np.float64)
    counts = {}
    left = 0
    for right in range(n):
        c = codes[right]
        counts[c] = counts.get(c, 0) + 1
        while times_ns[right] - times_ns[left] > window_ns:
            lc = codes[left]
            counts[lc] -= 1
            if counts[lc] == 0:
                del counts[lc]
            left += 1
        out[right] = len(counts)
    return out


def add_device_ip_velocity(df, window="24h"):
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    window_ns = pd.Timedelta(window).value
    card_codes = pd.Categorical(g["card_id"]).codes.astype(np.int64)
    ts_ns = g["timestamp"].values.astype("datetime64[ns]").astype(np.int64)

    def fanout(col):
        results = np.full(len(g), np.nan)
        mask = g[col].notna().values
        if not mask.any():
            return results
        idx = np.flatnonzero(mask)
        ent = pd.Categorical(g.loc[mask, col]).codes
        sub_t, sub_c = ts_ns[idx], card_codes[idx]
        order = np.argsort(ent, kind="stable")
        ent_s, t_s, c_s, orig_s = ent[order], sub_t[order], sub_c[order], idx[order]
        bounds = np.flatnonzero(np.diff(ent_s)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(ent_s)]))
        for s, e in zip(starts, ends):
            results[orig_s[s:e]] = _rolling_distinct_count(
                t_s[s:e], c_s[s:e], window_ns
            )
        return results

    g["device_card_fanout_24h"] = pd.Series(fanout("device_id")).fillna(1)
    g["ip_card_fanout_24h"] = pd.Series(fanout("ip_address")).fillna(1)
    return g


def _expanding_amount_zscore(g: pd.DataFrame, group_col: str) -> pd.Series:
    """Z-score vs expanding mean/std within group_col; current row excluded."""
    grp = g.groupby(group_col)
    a = g["amount"]
    cnt = grp.cumcount()
    csum = grp["amount"].cumsum() - a
    csumsq = grp["amount"].transform(lambda s: (s ** 2).cumsum()) - a ** 2

    denom = cnt.replace(0, np.nan)
    mean = csum / denom
    var = (csumsq - csum ** 2 / denom) / (cnt - 1).replace(0, np.nan)
    std = np.sqrt(var)
    return ((a - mean.fillna(a)) / std.fillna(0).replace(0, np.nan)).fillna(0)


def add_card_history_features(df):
    """Per-card running stats computed from PAST rows only (expanding,
    shifted by 1 so the current row never sees its own value)."""
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    grp = g.groupby("card_id")
    a = g["amount"]

    cnt = grp.cumcount()
    csum = grp["amount"].cumsum() - a
    csumsq = grp["amount"].transform(lambda s: (s ** 2).cumsum()) - a ** 2

    denom = cnt.replace(0, np.nan)
    mean = csum / denom
    var = (csumsq - csum ** 2 / denom) / (cnt - 1).replace(0, np.nan)
    std = np.sqrt(var)

    g["card_amt_mean_so_far"] = mean.fillna(a)
    g["card_amt_std_so_far"] = std.fillna(0)
    g["amt_z_vs_card"] = _expanding_amount_zscore(g, "card_id")
    return g


def add_category_history_features(df):
    """Amount z-score vs merchant_category norm (prior txs in category only)."""
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    g["amt_z_vs_category"] = _expanding_amount_zscore(g, "merchant_category")
    return g


def _card_rolling_distinct(g: pd.DataFrame, value_col: str, window: str) -> np.ndarray:
    """Per-card trailing distinct count of value_col in a time window."""
    window_ns = pd.Timedelta(window).value
    results = np.ones(len(g), dtype=np.float64)
    for _, card_df in g.groupby("card_id", sort=False):
        card_df = card_df.sort_values("timestamp")
        times_ns = card_df["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        codes = pd.Categorical(card_df[value_col]).codes.astype(np.int64)
        distinct = _rolling_distinct_count(times_ns, codes, window_ns)
        results[card_df.index.to_numpy()] = distinct
    return results


def add_card_category_diversity(df):
    """Distinct merchant_category count per card in trailing 24h."""
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    g["distinct_categories_24h"] = _card_rolling_distinct(g, "merchant_category", "24h")
    return g


def add_card_hour_pattern_features(df, min_history: int = 3):
    """Hour rarity vs the card's own prior transaction pattern."""
    g = df.sort_values("timestamp").reset_index(drop=True).copy()
    rarity = np.zeros(len(g), dtype=np.float64)
    never_seen = np.zeros(len(g), dtype=np.int64)

    for _, card_df in g.groupby("card_id", sort=False):
        hour_counts = np.zeros(24, dtype=np.int64)
        total = 0
        for idx, hour in zip(
            card_df.sort_values("timestamp").index,
            card_df.sort_values("timestamp")["hour_of_day"].astype(int),
        ):
            if total >= min_history:
                freq = hour_counts[hour] / total
                rarity[idx] = 1.0 - freq
                never_seen[idx] = int(hour_counts[hour] == 0)
            hour_counts[hour] += 1
            total += 1

    g["hour_rarity_for_card"] = rarity
    g["hour_never_seen_for_card"] = never_seen
    return g


def build_features(df):
    g = add_time_features(df)
    g = add_amount_features(g)
    g = add_card_velocity_features(g)
    g = add_change_features(g)
    g = add_device_ip_velocity(g)
    g = add_card_history_features(g)
    g = add_category_history_features(g)
    g = add_card_category_diversity(g)
    g = add_card_hour_pattern_features(g)

    # missing-value flags carry signal (e.g. online tx have no device on POS)
    g["device_missing"] = g["device_id"].isna().astype(int)
    g["ip_missing"] = g["ip_address"].isna().astype(int)
    return g


# ---------------------------------------------------------------------------
# 3. Categorical encoding
#    Use LightGBM's native categorical support (no leakage, no target enc).
# ---------------------------------------------------------------------------
CATEGORICAL = [
    "merchant_category",
    "channel",
    "cardholder_country",
    "merchant_country",
]

NUMERIC = [
    "amount", "amt_log", "amt_is_round", "amt_card_test",
    "amt_just_below_100", "amt_just_below_500",
    "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "user_age", "city_pop",
    "tx_1min", "tx_5min", "tx_1h", "tx_24h", "spend_24h",
    "time_since_last", "tx_of_day",
    "merchant_change", "device_change", "country_hop",
    "fast_country_hop", "cross_border",
    "device_card_fanout_24h", "ip_card_fanout_24h",
    "card_amt_mean_so_far", "card_amt_std_so_far", "amt_z_vs_card",
    "amt_z_vs_category", "distinct_categories_24h",
    "hour_rarity_for_card", "hour_never_seen_for_card",
    "device_missing", "ip_missing",
]

FEATURES = NUMERIC + CATEGORICAL

INFERENCE_OPTIONAL_COLUMNS = ("user_age", "distance_to_merchant", "city_pop")


def ensure_inference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill training-only columns missing from challenge / production CSVs."""
    out = df.copy()
    for col in INFERENCE_OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def prepare_features(g: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for inference (no labels)."""
    X = g[FEATURES].copy()
    for name in NUMERIC:
        X[name] = pd.to_numeric(X[name], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def prepare_matrix(g):
    X = prepare_features(g)
    y = g["is_fraud"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 4. Temporal split  (train / validation / test — past → future)
#     Threshold is picked on validation; test is held out for final metrics.
# ---------------------------------------------------------------------------
def temporal_split(
    g: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("require 0 < train_frac, 0 < val_frac, train_frac + val_frac < 1")
    cut_val = g["timestamp"].quantile(train_frac)
    cut_test = g["timestamp"].quantile(train_frac + val_frac)
    train = g[g["timestamp"] <= cut_val]
    val = g[(g["timestamp"] > cut_val) & (g["timestamp"] <= cut_test)]
    test = g[g["timestamp"] > cut_test]
    test_frac = 1.0 - train_frac - val_frac
    print(
        f"Split at {cut_val} / {cut_test} | "
        f"train={len(train):,} val={len(val):,} test={len(test):,} "
        f"({train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%})"
    )
    print(
        f"Fraud rate  train={train.is_fraud.mean():.4%}  "
        f"val={val.is_fraud.mean():.4%}  test={test.is_fraud.mean():.4%}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# 4b. Label-leakage / dataset realism — fraud vs legit describe()
#     If raw features separate classes cleanly, metrics will look great but
#     won't generalize; temper expectations before trusting the model.
# ---------------------------------------------------------------------------
LEAK_CHECK_FEATURES = [
    "distance_to_merchant",
    "amount",
    "user_age",
    "city_pop",
]

# IQR overlap below this + large Cohen's d → "cleanly separated" (easy dataset).
_SEPARATION_IQR_OVERLAP_MAX = 0.15
_SEPARATION_COHEN_D_MIN = 1.0


def _numeric_by_label(df: pd.DataFrame, feature: str) -> Tuple[pd.Series, pd.Series]:
    if "is_fraud" not in df.columns:
        raise ValueError("is_fraud column required for label validation")
    values = pd.to_numeric(df[feature], errors="coerce")
    labeled = df.assign(_value=values)
    legit = labeled.loc[labeled["is_fraud"] == 0, "_value"].dropna()
    fraud = labeled.loc[labeled["is_fraud"] == 1, "_value"].dropna()
    return legit, fraud


def describe_label_split(df: pd.DataFrame, feature: str) -> None:
    """Print describe() for legit vs fraud rows of one raw feature."""
    legit, fraud = _numeric_by_label(df, feature)
    print(f"\n{feature} - legit (n={len(legit):,}):")
    print(legit.describe().to_string())
    print(f"{feature} - fraud (n={len(fraud):,}):")
    print(fraud.describe().to_string())


def _iqr_overlap_ratio(a: pd.Series, b: pd.Series) -> float:
    q1_a, q3_a = a.quantile(0.25), a.quantile(0.75)
    q1_b, q3_b = b.quantile(0.25), b.quantile(0.75)
    overlap = max(0.0, min(q3_a, q3_b) - max(q1_a, q1_b))
    span = max(q3_a, q3_b) - min(q1_a, q1_b)
    return float(overlap / span) if span > 0 else 0.0


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    std_a, std_b = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def assess_feature_separation(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """Return separation stats; cleanly_separated flags trivially easy classes."""
    legit, fraud = _numeric_by_label(df, feature)
    if legit.empty or fraud.empty:
        return {
            "feature": feature,
            "n_legit": len(legit),
            "n_fraud": len(fraud),
            "cohen_d": 0.0,
            "iqr_overlap": 1.0,
            "cleanly_separated": False,
            "note": "missing rows for one class",
        }

    iqr_overlap = _iqr_overlap_ratio(legit, fraud)
    cohen_d = _cohen_d(legit, fraud)
    p10_legit, p90_legit = legit.quantile(0.10), legit.quantile(0.90)
    p10_fraud, p90_fraud = fraud.quantile(0.10), fraud.quantile(0.90)
    disjoint_p10_p90 = p90_legit < p10_fraud or p90_fraud < p10_legit
    cleanly_separated = (
        disjoint_p10_p90
        or (iqr_overlap <= _SEPARATION_IQR_OVERLAP_MAX and abs(cohen_d) >= _SEPARATION_COHEN_D_MIN)
    )
    return {
        "feature": feature,
        "n_legit": len(legit),
        "n_fraud": len(fraud),
        "cohen_d": round(cohen_d, 3),
        "iqr_overlap": round(iqr_overlap, 3),
        "cleanly_separated": cleanly_separated,
    }


def validate_dataset_labels(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run fraud vs legit describe() on raw features; warn if the dataset is trivial."""
    if "is_fraud" not in df.columns:
        if verbose:
            print("\nLabel validation skipped: no is_fraud column.")
        return {"features": {}, "cleanly_separated": [], "dataset_likely_easy": False}

    features = list(features or LEAK_CHECK_FEATURES)
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if verbose:
        print("\n--- Label validation (fraud vs legit describe) ---")
        if missing:
            print(f"  (skipped missing columns: {', '.join(missing)})")

    results: Dict[str, Dict[str, Any]] = {}
    separated: List[str] = []
    for feat in present:
        if verbose:
            describe_label_split(df, feat)
        stats = assess_feature_separation(df, feat)
        results[feat] = stats
        if stats["cleanly_separated"]:
            separated.append(feat)

    if verbose:
        print("\nSeparation summary (Cohen's d, IQR overlap; * = cleanly separated):")
        for feat, stats in results.items():
            flag = " *" if stats["cleanly_separated"] else ""
            print(
                f"  {feat:<22} d={stats['cohen_d']:+.3f}  "
                f"iqr_overlap={stats['iqr_overlap']:.3f}{flag}"
            )
        if separated:
            print(
                f"\n  WARNING: {', '.join(separated)} separate fraud/legit cleanly - "
                "the dataset is easy; strong offline metrics may not generalize."
            )
        else:
            print(
                "\n  Raw features overlap across labels - model signal likely comes "
                "from engineered features, not trivial splits."
            )

    return {
        "features": results,
        "cleanly_separated": separated,
        "dataset_likely_easy": bool(separated),
    }


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
def train_model(
    X_tr,
    y_tr,
    *,
    params: Optional[Dict[str, Any]] = None,
    scale_pos_weight: Optional[float] = None,
):
    """Train LightGBM. Optional ``params`` overrides defaults (e.g. from ``lgbm_params.load_lgbm_params``)."""
    from algo.lgbm_params import merge_lgbm_params, natural_scale_pos_weight

    lgbm_kw = merge_lgbm_params(params)
    if scale_pos_weight is None:
        scale_pos_weight = natural_scale_pos_weight(y_tr)
    model = LGBMClassifier(
        **lgbm_kw,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=CATEGORICAL)
    return model


# ---------------------------------------------------------------------------
# 6. Evaluate  (the metrics that actually matter for imbalanced fraud)
# ---------------------------------------------------------------------------
# Tune to real economics: if false alarms are costly (analyst time, friction),
# raise cost_fp and/or lower cost_fn — then pick threshold from the curve below.
DEFAULT_COST_FN = 50.0   # cost of missing a fraud
DEFAULT_COST_FP = 10.0   # cost of a false alarm


def precision_at_k(y_true, scores, k):
    """Precision among the top-k highest-scored transactions —
    mirrors an analyst reviewing the k riskiest alerts per day."""
    order = np.argsort(scores)[::-1][:k]
    return y_true.iloc[order].mean()


def build_threshold_curve(
    y_true,
    scores,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> pd.DataFrame:
    """Precision / recall / FP at every cutoff on the PR curve, plus expected cost."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    _, _, thr = precision_recall_curve(y, scores)
    rows: List[Dict[str, Any]] = []
    for t in thr:
        pred = scores >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = n_pos - tp
        flagged = tp + fp
        precision = tp / flagged if flagged else 0.0
        recall = tp / n_pos if n_pos else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "flagged": flagged,
                "cost": fn * cost_fn + fp * cost_fp,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)


def print_threshold_curve_summary(
    curve: pd.DataFrame,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    selected_threshold: Optional[float] = None,
) -> None:
    """Print cost-optimal and selected cutoffs only (no per-score table)."""
    if curve.empty:
        print("\nThreshold curve: no scored rows.")
        return

    best_idx = int(curve["cost"].idxmin())
    if selected_threshold is not None:
        marker_idx = int((curve["threshold"] - selected_threshold).abs().idxmin())
    else:
        marker_idx = best_idx

    print(
        f"\nThreshold curve (cost_fn={cost_fn:g}, cost_fp={cost_fp:g}):"
    )
    best = curve.loc[best_idx]
    print(
        f"  * lowest expected cost at threshold={best['threshold']:.4f} "
        f"(precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
        f"FP={int(best['fp'])}, cost={best['cost']:,.0f})"
    )
    if selected_threshold is not None and marker_idx != best_idx:
        row = curve.loc[marker_idx]
        print(
            f"  → using threshold={selected_threshold:.4f} "
            f"(precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"FP={int(row['fp'])}, cost={row['cost']:,.0f})"
        )


def print_threshold_curve_table(
    curve: pd.DataFrame,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    selected_threshold: Optional[float] = None,
) -> None:
    """Print every cutoff on the curve (verbose; use summary for normal runs)."""
    if curve.empty:
        print("\nThreshold curve: no scored rows.")
        return

    display = curve.copy()
    display["threshold"] = display["threshold"].map(lambda t: f"{t:.4f}")
    display["precision"] = display["precision"].map(lambda v: f"{v:.4f}")
    display["recall"] = display["recall"].map(lambda v: f"{v:.4f}")
    display["cost"] = display["cost"].map(lambda v: f"{v:,.0f}")
    print(
        f"\nThreshold curve (cost_fn={cost_fn:g}, cost_fp={cost_fp:g}) — "
        "read the tradeoffs, then pick a cutoff:"
    )
    print(
        display[
            ["threshold", "precision", "recall", "tp", "fp", "fn", "flagged", "cost"]
        ].to_string(index=False)
    )
    print_threshold_curve_summary(
        curve,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        selected_threshold=selected_threshold,
    )


def pick_threshold_by_cost(
    y_true,
    scores,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> Tuple[float, float, pd.DataFrame]:
    """Return the cutoff on the curve that minimizes expected cost."""
    curve = build_threshold_curve(y_true, scores, cost_fn=cost_fn, cost_fp=cost_fp)
    if curve.empty:
        return 0.5, np.inf, curve
    best = curve.loc[curve["cost"].idxmin()]
    return float(best["threshold"]), float(best["cost"]), curve


def tune_threshold(
    y_true,
    scores,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    threshold: Optional[float] = None,
    print_table: bool = False,
    print_summary: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """Pick a threshold from the PR curve; optional full table or summary lines."""
    curve = build_threshold_curve(y_true, scores, cost_fn=cost_fn, cost_fp=cost_fp)
    if threshold is None:
        threshold, cost, _ = pick_threshold_by_cost(y_true, scores, cost_fn, cost_fp)
        if print_table:
            print_threshold_curve_table(
                curve, cost_fn=cost_fn, cost_fp=cost_fp, selected_threshold=threshold
            )
        elif print_summary:
            print_threshold_curve_summary(
                curve, cost_fn=cost_fn, cost_fp=cost_fp, selected_threshold=threshold
            )
        print(
            f"\nCost-optimal threshold: {threshold:.4f} (expected cost={cost:,.0f})"
        )
        if print_table:
            print(
                "  Override pipeline.threshold after reading the table if ops "
                "needs a different precision/recall point."
            )
    else:
        if print_table:
            print_threshold_curve_table(
                curve, cost_fn=cost_fn, cost_fp=cost_fp, selected_threshold=threshold
            )
        elif print_summary:
            print_threshold_curve_summary(
                curve, cost_fn=cost_fn, cost_fp=cost_fp, selected_threshold=threshold
            )
    return threshold, curve


def evaluate(
    model,
    X_te,
    y_te,
    *,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    threshold: Optional[float] = None,
):
    scores = model.predict_proba(X_te)[:, 1]
    print(f"\nPR-AUC : {average_precision_score(y_te, scores):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_te, scores):.4f}")
    for k in (100, 500, 1000):
        k = min(k, len(y_te))
        print(f"Precision@{k:<5}: {precision_at_k(y_te, scores, k):.4f}")

    t, _ = tune_threshold(
        y_te, scores, cost_fn=cost_fn, cost_fp=cost_fp, threshold=threshold
    )
    pred = (scores >= t).astype(int)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_te, pred))
    print("\n", classification_report(y_te, pred, digits=4))
    return scores


def show_importances(model, top=20):
    imp = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
        .head(top)
    )
    print("\nTop feature importances:")
    print(imp.to_string())


# ---------------------------------------------------------------------------
# 7. Six rule guardrails (parallel to model — catch novel fraud)
# ---------------------------------------------------------------------------
RULE_COLUMNS = [
    "rule_amount",
    "rule_velocity",
    "rule_geo",
    "rule_offhours",
    "rule_device_ip",
    "rule_merchant_burst",
]

RULE_LABELS = {
    "rule_amount": "amount",
    "rule_velocity": "velocity",
    "rule_geo": "geo",
    "rule_offhours": "off-hours",
    "rule_device_ip": "device/ip fanout",
    "rule_merchant_burst": "merchant burst",
}


def apply_rule_guardrails(g: pd.DataFrame) -> pd.DataFrame:
    """Six deterministic rules that always run alongside the ML model."""
    out = g.copy()
    out["rule_amount"] = (out["amt_z_vs_card"] >= 3.0) | (out["amt_z_vs_category"] >= 3.5)
    out["rule_velocity"] = (out["tx_5min"] >= 4) | (out["tx_1h"] >= 8)
    out["rule_geo"] = (out["cross_border"] == 1) & (
        (out["country_hop"] == 1) | (out["fast_country_hop"] == 1)
    )
    out["rule_offhours"] = (
        (out["hour_never_seen_for_card"] == 1) | (out["hour_rarity_for_card"] >= 0.85)
    ) & ((out["amt_z_vs_card"] >= 2.0) | (out["amt_z_vs_category"] >= 2.5))
    out["rule_device_ip"] = (out["device_card_fanout_24h"] >= 3) | (
        out["ip_card_fanout_24h"] >= 3
    )
    out["rule_merchant_burst"] = (out["merchant_change"] == 1) & (out["tx_1h"] >= 3)
    out["rule_guardrail"] = out[RULE_COLUMNS].any(axis=1)
    out["rule_reason_codes"] = [
        _rule_reason_codes(row) for _, row in out.iterrows()
    ]
    return out


def _rule_reason_codes(row: pd.Series) -> List[str]:
    parts: List[str] = []
    if row["rule_amount"]:
        card_sigma = float(row["amt_z_vs_card"])
        cat_sigma = float(row["amt_z_vs_category"])
        if card_sigma >= 3.0:
            parts.append(f"amount {max(card_sigma, 3.0):.0f}σ above card norm")
        if cat_sigma >= 3.5:
            cat = row.get("merchant_category", "category")
            parts.append(f"amount {cat_sigma:.0f}σ above {cat} norm")
    if row["rule_velocity"]:
        parts.append(
            f"velocity spike ({int(row['tx_5min'])} tx/5m, {int(row['tx_1h'])} tx/1h)"
        )
    if row["rule_geo"]:
        parts.append(
            f"geo hop ({row['cardholder_country']}→{row['merchant_country']})"
        )
    if row["rule_offhours"]:
        if row["hour_never_seen_for_card"] == 1:
            parts.append(f"atypical hour for card ({int(row['hour_of_day']):02d}:00)")
        else:
            parts.append(f"rare hour for card ({int(row['hour_of_day']):02d}:00)")
    if row["rule_device_ip"]:
        if row["device_card_fanout_24h"] >= 3:
            parts.append(f"{int(row['device_card_fanout_24h'])} cards on this device")
        if row["ip_card_fanout_24h"] >= 3:
            parts.append(f"{int(row['ip_card_fanout_24h'])} cards on this IP")
    if row["rule_merchant_burst"]:
        parts.append(
            f"merchant burst ({int(row['tx_1h'])} tx/1h after merchant change)"
        )
    return parts


def format_alert_reason(
    shap_codes: Sequence[str],
    rule_codes: Sequence[str],
    *,
    model_score: Optional[float] = None,
    rule_guardrail: bool = False,
) -> str:
    """Analyst-facing reason string, e.g. 'flagged: amount 6σ above card norm + 9 cards on this IP'."""
    seen: set[str] = set()
    ordered: List[str] = []
    for code in list(rule_codes) + list(shap_codes):
        key = code.strip().lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(code.strip())
    if not ordered:
        if model_score is not None and model_score >= 0.5:
            ordered.append(f"model score {model_score:.2f}")
        elif rule_guardrail:
            ordered.append("rule guardrail triggered")
        else:
            return ""
    prefix = "flagged (guardrail)" if rule_guardrail and not shap_codes else "flagged"
    return f"{prefix}: " + " + ".join(ordered)


# ---------------------------------------------------------------------------
# 8. SHAP explainability — per-alert reason codes
# ---------------------------------------------------------------------------
def _feature_reason_code(name: str, shap_val: float, row: pd.Series) -> Optional[str]:
    """Map a positively-contributing feature to a short analyst reason."""
    if shap_val <= 0:
        return None
    val = row.get(name)
    if name == "amt_z_vs_card" and val is not None:
        sigma = abs(float(val))
        if sigma >= 2:
            return f"amount {sigma:.0f}σ above card norm"
        return f"amount elevated vs card norm ({sigma:.1f}σ)"
    if name == "amt_z_vs_category" and val is not None:
        sigma = abs(float(val))
        cat = row.get("merchant_category", "category")
        if sigma >= 2:
            return f"amount {sigma:.0f}σ above {cat} norm"
        return f"amount elevated vs {cat} norm ({sigma:.1f}σ)"
    if name == "distinct_categories_24h" and val is not None and float(val) >= 3:
        return f"{int(val)} merchant categories in 24h"
    if name == "hour_never_seen_for_card" and val == 1:
        return f"atypical hour for card ({int(row.get('hour_of_day', 0)):02d}:00)"
    if name == "hour_rarity_for_card" and val is not None and float(val) >= 0.85:
        return f"rare hour for card ({int(row.get('hour_of_day', 0)):02d}:00)"
    if name == "ip_card_fanout_24h" and val is not None and float(val) >= 2:
        return f"{int(val)} cards on this IP"
    if name == "device_card_fanout_24h" and val is not None and float(val) >= 2:
        return f"{int(val)} cards on this device"
    if name == "tx_5min" and val is not None and float(val) >= 2:
        return f"{int(val)} tx in 5 minutes"
    if name == "tx_1h" and val is not None and float(val) >= 4:
        return f"{int(val)} tx in 1 hour"
    if name == "cross_border" and val == 1:
        return f"cross-border ({row.get('cardholder_country')}→{row.get('merchant_country')})"
    if name == "fast_country_hop" and val == 1:
        return "fast country hop (<60 min)"
    if name == "is_night" and val == 1:
        return f"night transaction ({int(row.get('hour_of_day', 0)):02d}:00)"
    if name == "merchant_change" and val == 1:
        return "new merchant for card"
    if name == "device_change" and val == 1:
        return "new device for card"
    if name == "amt_just_below_100" and val == 1:
        return "amount just below $100 threshold"
    if name == "amt_just_below_500" and val == 1:
        return "amount just below $500 threshold"
    if shap_val >= 0.02:
        readable = name.replace("_", " ")
        return f"elevated {readable}"
    return None


def build_shap_explainer(model: LGBMClassifier, X_background: Optional[pd.DataFrame] = None):
    # LightGBM categorical splits require tree_path_dependent when using SHAP.
    _ = X_background
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def _prepare_shap_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[FEATURES].copy()
    for name in NUMERIC:
        X[name] = pd.to_numeric(X[name], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def _extract_shap_contributions(explainer, X: pd.DataFrame) -> np.ndarray:
    """Batch SHAP contributions, shape (n_samples, n_features)."""
    shap_values = explainer.shap_values(X, from_call=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def _contributions_to_reason_codes(
    row: pd.Series,
    contributions: np.ndarray,
    *,
    top_k: int = 3,
    min_shap: float = 0.01,
) -> List[str]:
    ranked = sorted(
        zip(FEATURES, contributions),
        key=lambda item: item[1],
        reverse=True,
    )
    reasons: List[str] = []
    for name, contrib in ranked:
        if len(reasons) >= top_k:
            break
        if contrib < min_shap:
            continue
        code = _feature_reason_code(name, float(contrib), row)
        if code:
            reasons.append(code)
    return reasons


def shap_reason_codes(
    explainer,
    row: pd.Series,
    *,
    top_k: int = 3,
    min_shap: float = 0.01,
) -> List[str]:
    """Top SHAP contributors as human-readable reason codes for one alert."""
    X_row = _prepare_shap_matrix(pd.DataFrame([row]))
    contributions = _extract_shap_contributions(explainer, X_row)[0]
    return _contributions_to_reason_codes(row, contributions, top_k=top_k, min_shap=min_shap)


def explain_alerts(
    explainer,
    g: pd.DataFrame,
    model_scores: np.ndarray,
    *,
    flagged_mask: Optional[np.ndarray] = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Attach SHAP + rule reason codes to flagged rows."""
    out = g.copy()
    if flagged_mask is None:
        flagged_mask = out["is_fraud"].values if "is_fraud" in out.columns else model_scores >= 0.5
    flagged_idx = np.flatnonzero(flagged_mask)
    shap_by_index: Dict[int, List[str]] = {}

    if len(flagged_idx) > 0:
        flagged_rows = out.iloc[flagged_idx]
        contributions = _extract_shap_contributions(
            explainer, _prepare_shap_matrix(flagged_rows)
        )
        for pos, row_idx in enumerate(flagged_idx):
            shap_by_index[int(row_idx)] = _contributions_to_reason_codes(
                out.iloc[row_idx],
                contributions[pos],
                top_k=top_k,
            )

    reason_codes: List[str] = []
    shap_codes_col: List[List[str]] = []
    for i, (_, row) in enumerate(out.iterrows()):
        if not flagged_mask[i]:
            reason_codes.append("")
            shap_codes_col.append([])
            continue
        shap_codes = shap_by_index.get(i, [])
        rule_codes = row.get("rule_reason_codes") or []
        if isinstance(rule_codes, str):
            rule_codes = json.loads(rule_codes) if rule_codes.startswith("[") else [rule_codes]
        alert = format_alert_reason(
            shap_codes,
            rule_codes,
            model_score=float(model_scores[i]),
            rule_guardrail=bool(row.get("rule_guardrail", False)),
        )
        reason_codes.append(alert)
        shap_codes_col.append(shap_codes)
    out["shap_reason_codes"] = shap_codes_col
    out["alert_reason"] = reason_codes
    return out


# ---------------------------------------------------------------------------
# 9. Drift monitoring — weekly PR-AUC + scheduled retrain
# ---------------------------------------------------------------------------
OPS_DIR = Path(__file__).resolve().parent / "ops"
DEFAULT_METRICS_PATH = OPS_DIR / "drift_metrics.json"
DEFAULT_MODEL_PATH = OPS_DIR / "fraud_model.pkl"


def _iso_week(ts: Optional[datetime] = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    year, week, _ = ts.isocalendar()
    return f"{year}-W{week:02d}"


class DriftMonitor:
    """Track PR-AUC weekly; alert and retrain when performance drifts."""

    PR_AUC_DROP_THRESHOLD = 0.05
    RETRAIN_INTERVAL_WEEKS = 4

    def __init__(self, metrics_path: Path | str = DEFAULT_METRICS_PATH):
        self.path = Path(metrics_path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"baseline_pr_auc": None, "weekly": [], "last_retrain_week": None}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_weekly(
        self,
        pr_auc: float,
        n_samples: int,
        *,
        week: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        week = week or _iso_week()
        data = self.load()
        if data["baseline_pr_auc"] is None:
            data["baseline_pr_auc"] = round(pr_auc, 6)
        entry = {
            "week": week,
            "pr_auc": round(float(pr_auc), 6),
            "n_samples": int(n_samples),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
        weekly = [row for row in data["weekly"] if row.get("week") != week]
        weekly.append(entry)
        data["weekly"] = sorted(weekly, key=lambda row: row["week"])[-52:]
        self.save(data)
        return entry

    def latest_pr_auc(self) -> Optional[float]:
        weekly = self.load()["weekly"]
        if not weekly:
            return None
        return float(weekly[-1]["pr_auc"])

    def drift_detected(self) -> Tuple[bool, str]:
        data = self.load()
        baseline = data.get("baseline_pr_auc")
        latest = self.latest_pr_auc()
        if baseline is None or latest is None:
            return False, "insufficient history"
        drop = baseline - latest
        if drop >= self.PR_AUC_DROP_THRESHOLD:
            return True, f"PR-AUC dropped {drop:.4f} vs baseline ({baseline:.4f}→{latest:.4f})"
        return False, "within tolerance"

    def weeks_since_retrain(self) -> Optional[int]:
        data = self.load()
        last = data.get("last_retrain_week")
        if not last:
            return None
        last_year, last_week = map(int, last.split("-W"))
        now_year, now_week = map(int, _iso_week().split("-W"))
        return (now_year - last_year) * 52 + (now_week - last_week)

    def should_retrain(self) -> Tuple[bool, str]:
        drift, drift_msg = self.drift_detected()
        if drift:
            return True, drift_msg
        elapsed = self.weeks_since_retrain()
        if elapsed is None:
            return False, "no prior retrain recorded"
        if elapsed >= self.RETRAIN_INTERVAL_WEEKS:
            return True, f"scheduled retrain ({elapsed} weeks since last)"
        return False, f"next retrain in {self.RETRAIN_INTERVAL_WEEKS - elapsed} week(s)"

    def mark_retrained(self, pr_auc: float) -> None:
        data = self.load()
        data["baseline_pr_auc"] = round(float(pr_auc), 6)
        data["last_retrain_week"] = _iso_week()
        data["last_retrain_at"] = datetime.now(timezone.utc).isoformat()
        self.save(data)

    def summary(self) -> str:
        data = self.load()
        lines = [
            f"baseline PR-AUC: {data.get('baseline_pr_auc')}",
            f"last retrain week: {data.get('last_retrain_week')}",
        ]
        for row in data.get("weekly", [])[-8:]:
            lines.append(f"  {row['week']}: PR-AUC={row['pr_auc']:.4f} (n={row['n_samples']})")
        drift, msg = self.drift_detected()
        retrain, retrain_msg = self.should_retrain()
        lines.append(f"drift: {msg} ({'ALERT' if drift else 'ok'})")
        lines.append(f"retrain: {retrain_msg} ({'YES' if retrain else 'no'})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Hybrid pipeline — model + rule guardrails + SHAP + drift ops
# ---------------------------------------------------------------------------
class FraudDetectionPipeline:
    """Train/evaluate/score with ML model, parallel rules, SHAP reasons, drift tracking."""

    def __init__(
        self,
        model_threshold: Optional[float] = None,
        *,
        cost_fn: float = DEFAULT_COST_FN,
        cost_fp: float = DEFAULT_COST_FP,
        metrics_path: Path | str = DEFAULT_METRICS_PATH,
        print_threshold_table: bool = False,
    ):
        self.model: Optional[LGBMClassifier] = None
        self.explainer = None
        self.threshold = 0.5 if model_threshold is None else model_threshold
        self._threshold_from_curve = model_threshold is None
        self._threshold_tuned_on_val = False
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.print_threshold_table = print_threshold_table
        self.monitor = DriftMonitor(metrics_path)

    def fit(
        self,
        path: str,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
    ) -> "FraudDetectionPipeline":
        df = shrink(load(path))
        validate_dataset_labels(df)
        g = apply_rule_guardrails(build_features(df))
        train, val, test = temporal_split(g, train_frac=train_frac, val_frac=val_frac)
        X_tr, y_tr = prepare_matrix(train)
        X_val, y_val = prepare_matrix(val)
        X_te, y_te = prepare_matrix(test)
        self.model = train_model(X_tr, y_tr)
        self.explainer = build_shap_explainer(self.model, X_tr)
        self._last_train = train
        self._last_val = val
        self._last_test = test
        self._last_X_val = X_val
        self._last_y_val = y_val
        self._last_X_te = X_te
        self._last_y_te = y_te
        self._threshold_tuned_on_val = False
        if self._threshold_from_curve:
            print("\n--- Validation set (threshold selection) ---")
            val_scores = self.model.predict_proba(X_val)[:, 1]
            t, _ = tune_threshold(
                y_val,
                val_scores,
                cost_fn=self.cost_fn,
                cost_fp=self.cost_fp,
                print_table=self.print_threshold_table,
            )
            self.threshold = t
            self._threshold_tuned_on_val = True
        return self

    def model_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Pipeline not fitted — call fit() first")
        return self.model.predict_proba(X)[:, 1]

    def hybrid_scores(self, g: pd.DataFrame, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (model probability, combined score capped at 1.0)."""
        model_prob = self.model_scores(X)
        rule_boost = g["rule_guardrail"].astype(float).values * 0.35
        combined = np.clip(model_prob + rule_boost, 0.0, 1.0)
        return model_prob, combined

    def predict(
        self,
        g: pd.DataFrame,
        X: pd.DataFrame,
        *,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Flag if model OR any rule guardrail fires."""
        threshold = self.threshold if threshold is None else threshold
        g = apply_rule_guardrails(g) if "rule_guardrail" not in g.columns else g
        model_prob, combined = self.hybrid_scores(g, X)
        flagged = (model_prob >= threshold) | g["rule_guardrail"].values
        out = g.copy()
        out["model_score"] = np.round(model_prob, 4)
        out["fraud_score"] = np.round(combined, 4)
        out["is_fraud"] = flagged
        out["flagged_by_model"] = model_prob >= threshold
        out["flagged_by_rules"] = g["rule_guardrail"].values
        if self.explainer is not None:
            out = explain_alerts(self.explainer, out, model_prob, flagged_mask=flagged)
        else:
            out["alert_reason"] = [
                format_alert_reason([], row.get("rule_reason_codes") or [], rule_guardrail=row["rule_guardrail"])
                if flagged[i]
                else ""
                for i, (_, row) in enumerate(out.iterrows())
            ]
        return out

    def evaluate(
        self,
        model=None,
        X_te=None,
        y_te=None,
        *,
        cost_fn: Optional[float] = None,
        cost_fp: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> float:
        model = model or self.model
        X_te = X_te if X_te is not None else getattr(self, "_last_X_te", None)
        y_te = y_te if y_te is not None else getattr(self, "_last_y_te", None)
        if model is None or X_te is None or y_te is None:
            raise RuntimeError("Nothing to evaluate — fit the pipeline or pass model/X/y")
        cost_fn = self.cost_fn if cost_fn is None else cost_fn
        cost_fp = self.cost_fp if cost_fp is None else cost_fp
        scores = model.predict_proba(X_te)[:, 1]
        pr_auc = float(average_precision_score(y_te, scores))
        print("\n--- Test set (held out) ---")
        print(f"\nPR-AUC : {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_te, scores):.4f}")
        for k in (100, 500, 1000):
            k = min(k, len(y_te))
            print(f"Precision@{k:<5}: {precision_at_k(y_te, scores, k):.4f}")
        t = self.threshold if threshold is None else threshold
        if self._threshold_tuned_on_val and threshold is None:
            print(f"\nUsing threshold={t:.4f} (selected on validation; not re-tuned on test)")
        t, _ = tune_threshold(
            y_te,
            scores,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            threshold=t,
            print_table=self.print_threshold_table,
        )
        pred = (scores >= t).astype(int)
        print("\nConfusion matrix:")
        print(confusion_matrix(y_te, pred))
        print("\n", classification_report(y_te, pred, digits=4))
        return pr_auc

    def evaluate_hybrid(self) -> float:
        test = getattr(self, "_last_test", None)
        X_te = getattr(self, "_last_X_te", None)
        y_te = getattr(self, "_last_y_te", None)
        if test is None or X_te is None or y_te is None:
            raise RuntimeError("Nothing to evaluate — call fit() first")
        scored = self.predict(test, X_te)
        hybrid_prob = scored["fraud_score"].values
        pr_auc = float(average_precision_score(y_te, hybrid_prob))
        rule_hits = int(scored["flagged_by_rules"].sum())
        model_hits = int(scored["flagged_by_model"].sum())
        both = int((scored["flagged_by_model"] & scored["flagged_by_rules"]).sum())
        print(f"\nHybrid PR-AUC: {pr_auc:.4f}")
        print(f"Flags — model: {model_hits}, rules: {rule_hits}, overlap: {both}")
        flagged = scored[scored["is_fraud"]].head(5)
        if not flagged.empty and "alert_reason" in flagged.columns:
            print("\nSample alert reasons:")
            for _, row in flagged.iterrows():
                print(f"  {row.get('transaction_id', '?')}: {row['alert_reason']}")
        return pr_auc

    def record_drift_metrics(self, pr_auc: Optional[float] = None) -> Dict[str, Any]:
        y_te = getattr(self, "_last_y_te", None)
        n_samples = len(y_te) if y_te is not None else 0
        if pr_auc is None:
            pr_auc = self.evaluate()
        entry = self.monitor.record_weekly(pr_auc, n_samples)
        print("\nDrift monitor:")
        print(self.monitor.summary())
        return entry

    def maybe_retrain(self, path: str, *, force: bool = False) -> Optional["FraudDetectionPipeline"]:
        should, reason = self.monitor.should_retrain()
        if not (force or should):
            print(f"Skipping retrain: {reason}")
            return None
        print(f"Retraining: {reason}")
        self.fit(path)
        pr_auc = self.evaluate()
        self.monitor.mark_retrained(pr_auc)
        self.monitor.record_weekly(pr_auc, len(self._last_y_te))
        return self

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
        if self.model is None:
            raise RuntimeError("Pipeline not fitted — nothing to save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "threshold": float(self.threshold),
            "features": list(FEATURES),
            "version": 1,
        }
        with target.open("wb") as handle:
            pickle.dump(artifact, handle)
        return target

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH) -> "FraudDetectionPipeline":
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"No model artifact at {target}")
        with target.open("rb") as handle:
            artifact = pickle.load(handle)
        pipeline = cls(model_threshold=float(artifact["threshold"]))
        pipeline.model = artifact["model"]
        pipeline.threshold = float(artifact["threshold"])
        pipeline._threshold_from_curve = False
        pipeline._threshold_tuned_on_val = True
        pipeline.explainer = None
        return pipeline


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(path):
    pipeline = FraudDetectionPipeline()
    pipeline.fit(path)
    pipeline.evaluate()
    show_importances(pipeline.model)
    pipeline.evaluate_hybrid()
    pipeline.record_drift_metrics()
    retrain, reason = pipeline.monitor.should_retrain()
    if retrain:
        print(f"\nRetrain recommended: {reason}")
    return pipeline


if __name__ == "__main__":
    import sys

    # Running as a script puts this directory on sys.path[0], so ``import algo``
    # resolves to algo.py instead of the package. Prefer the backend root.
    _backend = Path(__file__).resolve().parent.parent
    _script_dir = Path(__file__).resolve().parent
    if sys.path and sys.path[0] == str(_script_dir):
        sys.path.pop(0)
    if str(_backend) not in sys.path:
        sys.path.insert(0, str(_backend))

    main(sys.argv[1] if len(sys.argv) > 1 else "fraudTrain_part1.csv")