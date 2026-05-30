"""
Fraud detection: rule features + LightGBM, with temporal validation.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
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

    # set timestamp index per group for time-based rolling
    def roll_count(s, win):
        return s.rolling(win).count()

    def roll_sum(s, win):
        return s.rolling(win).sum()

    idx = g.set_index("timestamp").groupby("card_id")["amount"]

    g["tx_1min"] = idx.transform(lambda s: roll_count(s, "1min")).values
    g["tx_5min"] = idx.transform(lambda s: roll_count(s, "5min")).values
    g["tx_1h"] = idx.transform(lambda s: roll_count(s, "1h")).values
    g["tx_24h"] = idx.transform(lambda s: roll_count(s, "24h")).values
    g["spend_24h"] = idx.transform(lambda s: roll_sum(s, "24h")).values

    grp = g.groupby("card_id")
    g["time_since_last"] = grp["timestamp"].diff().dt.total_seconds()
    g["time_since_last"] = g["time_since_last"].fillna(1e7)  # first tx = "long ago"
    g["tx_of_day"] = g.groupby(["card_id", "date"]).cumcount() + 1
    return g


def add_change_features(df):
    """Did merchant/device/country change vs the card's previous tx?"""
    g = df.sort_values("timestamp").copy()
    grp = g.groupby("card_id")
    g["merchant_change"] = (
        g["merchant_name"] != grp["merchant_name"].shift()
    ).astype(int)
    g["device_change"] = (
        g["device_id"].fillna("NA") != grp["device_id"].shift().fillna("NA")
    ).astype(int)
    g["prev_merch_country"] = grp["merchant_country"].shift()
    g["country_hop"] = (
        (g["merchant_country"] != g["prev_merch_country"])
        & g["prev_merch_country"].notna()
    ).astype(int)
    g["minutes_since_last"] = g["time_since_last"] / 60.0
    # fast country hop = classic cloning tell
    g["fast_country_hop"] = (
        (g["country_hop"] == 1) & (g["minutes_since_last"] < 60)
    ).astype(int)
    g["cross_border"] = (
        g["cardholder_country"] != g["merchant_country"]
    ).astype(int)
    return g


def add_device_ip_velocity(df):
    """How many distinct cards touched this device/IP recently?
    A device or IP fanning out across many cards is a strong fraud-ring tell."""
    g = df.sort_values("timestamp").copy()

    def fanout(col, win="24h"):
        # rolling count of distinct cards per entity in a time window
        sub = g[["timestamp", col, "card_id"]].dropna(subset=[col]).copy()
        sub = sub.set_index("timestamp")
        # nunique isn't supported in rolling; approximate via factorize trick
        out = (
            sub.groupby(col)["card_id"]
            .transform(
                lambda s: s.rolling(win).apply(
                    lambda x: len(pd.unique(x)), raw=False
                )
            )
        )
        return out.reindex(g.index).values

    # NOTE: rolling.apply with unique is O(n*w); fine for moderate data.
    # For large data, switch to a hashed approximate-distinct or pre-agg.
    g["device_card_fanout_24h"] = pd.Series(
        fanout("device_id"), index=g.index
    ).fillna(1)
    g["ip_card_fanout_24h"] = pd.Series(
        fanout("ip_address"), index=g.index
    ).fillna(1)
    return g


def add_card_history_features(df):
    """Per-card running stats computed from PAST rows only (expanding,
    shifted by 1 so the current row never sees its own value)."""
    g = df.sort_values("timestamp").copy()
    grp = g.groupby("card_id")["amount"]
    g["card_amt_mean_so_far"] = grp.transform(
        lambda s: s.shift().expanding().mean()
    )
    g["card_amt_std_so_far"] = grp.transform(
        lambda s: s.shift().expanding().std()
    )
    # z-score of current amount vs card's own history
    g["amt_z_vs_card"] = (
        (g["amount"] - g["card_amt_mean_so_far"])
        / g["card_amt_std_so_far"].replace(0, np.nan)
    )
    g["amt_z_vs_card"] = g["amt_z_vs_card"].fillna(0)
    g["card_amt_mean_so_far"] = g["card_amt_mean_so_far"].fillna(g["amount"])
    g["card_amt_std_so_far"] = g["card_amt_std_so_far"].fillna(0)
    return g


def build_features(df):
    g = add_time_features(df)
    g = add_amount_features(g)
    g = add_card_velocity_features(g)
    g = add_change_features(g)
    g = add_device_ip_velocity(g)
    g = add_card_history_features(g)

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
    "amount",
    "amt_log",
    "amt_is_round",
    "amt_card_test",
    "amt_just_below_100",
    "amt_just_below_500",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "user_age",
    "distance_to_merchant",
    "city_pop",
    "tx_1min",
    "tx_5min",
    "tx_1h",
    "tx_24h",
    "spend_24h",
    "time_since_last",
    "tx_of_day",
    "merchant_change",
    "device_change",
    "country_hop",
    "fast_country_hop",
    "cross_border",
    "device_card_fanout_24h",
    "ip_card_fanout_24h",
    "card_amt_mean_so_far",
    "card_amt_std_so_far",
    "amt_z_vs_card",
    "device_missing",
    "ip_missing",
]

FEATURES = NUMERIC + CATEGORICAL


def prepare_matrix(g):
    X = g[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    y = g["is_fraud"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 4. Temporal split  (train on the past, test on the future)
# ---------------------------------------------------------------------------
def temporal_split(g, train_frac=0.8):
    cut = g["timestamp"].quantile(train_frac)
    train = g[g["timestamp"] <= cut]
    test = g[g["timestamp"] > cut]
    print(f"Split at {cut} | train={len(train):,} test={len(test):,}")
    print(
        f"Fraud rate  train={train.is_fraud.mean():.4%}  "
        f"test={test.is_fraud.mean():.4%}"
    )
    return train, test


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
def train_model(X_tr, y_tr):
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=neg / max(pos, 1),  # imbalance handling
        objective="binary",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_tr,
        y_tr,
        categorical_feature=CATEGORICAL,
    )
    return model


# ---------------------------------------------------------------------------
# 6. Evaluate  (the metrics that actually matter for imbalanced fraud)
# ---------------------------------------------------------------------------
def precision_at_k(y_true, scores, k):
    """Precision among the top-k highest-scored transactions —
    mirrors an analyst reviewing the k riskiest alerts per day."""
    order = np.argsort(scores)[::-1][:k]
    return y_true.iloc[order].mean()


def pick_threshold_by_cost(y_true, scores, cost_fn=100.0, cost_fp=2.0):
    """Choose the probability cutoff that minimizes expected cost.
    cost_fn = cost of MISSING a fraud; cost_fp = cost of a false alarm."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    best_t, best_cost = 0.5, np.inf
    n_pos = y_true.sum()
    n = len(y_true)
    for t in thr:
        pred = scores >= t
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = n_pos - tp
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost


def evaluate(model, X_te, y_te):
    scores = model.predict_proba(X_te)[:, 1]

    pr_auc = average_precision_score(y_te, scores)
    roc = roc_auc_score(y_te, scores)
    print(f"\nPR-AUC (avg precision): {pr_auc:.4f}")
    print(f"ROC-AUC               : {roc:.4f}")

    for k in (100, 500, 1000):
        k = min(k, len(y_te))
        print(f"Precision@{k:<5}: {precision_at_k(y_te, scores, k):.4f}")

    t, cost = pick_threshold_by_cost(y_te, scores)
    print(f"\nCost-optimal threshold: {t:.4f} (expected cost={cost:,.0f})")

    pred = (scores >= t).astype(int)
    print("\nConfusion matrix @ cost-optimal threshold:")
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
# main
# ---------------------------------------------------------------------------
def main(path):
    df = load(path)
    g = build_features(df)
    print(g.groupby("is_fraud")["distance_to_merchant"].describe())

#     train, test = temporal_split(g)

#     X_tr, y_tr = prepare_matrix(train)
#     X_te, y_te = prepare_matrix(test)

#     model = train_model(X_tr, y_tr)
#     evaluate(model, X_te, y_te)
#     show_importances(model)
#     return model


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "transactions.csv")