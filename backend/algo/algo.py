import pandas as pd
import numpy as np

def load(path) -> pd.DataFrame:
    """Load csv to df"""
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
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    return df

def velocity(df, window="5min", threshold=5):
    """Count transactions per card in a trailing time window."""
    out = df.copy()

    def _count(g):
        s = g.set_index("timestamp")["amount"]
        # trailing rolling count over time, inclusive of current row
        return s.rolling(window).count()

    out["tx_in_window"] = (
        out.set_index("timestamp")
        .groupby("card_id")["amount"]
        .transform(lambda s: s.rolling(window).count())
        .values
    )
    return out[out["tx_in_window"] >= threshold]


def geo_anomalies(df, min_minutes=10):
    g = df.sort_values("timestamp").copy()
    grp = g.groupby("card_id")

    g["prev_ts"] = grp["timestamp"].shift()
    g["prev_country"] = grp["merchant_country"].shift()
    g["prev_dist"] = grp["distance_to_merchant"].shift()
    g["minutes_apart"] = (g["timestamp"] - g["prev_ts"]).dt.total_seconds() / 60

    # a) merchant country changed within a short window
    country_hop = (
        (g["merchant_country"] != g["prev_country"])
        & g["prev_country"].notna()
        & (g["minutes_apart"] < min_minutes)
    )

    # b) cardholder transacting far from home in a foreign merchant country
    cross_border = (
        (g["cardholder_country"] != g["merchant_country"])
        & (g["distance_to_merchant"] > 0.95)  # normalized distance, high end
    )

    g["geo_flag"] = country_hop | cross_border
    return g[g["geo_flag"]]


def amount_anomalies(df):
    a = df["amount"]
    just_below = (
        ((a >= 99.50) & (a < 100.00))
        | ((a >= 499.50) & (a < 500.00))
    )
    card_test = a.isin([1.00, 5.00, 10.00])
    # generic round-dollar test (cheap card-testing tell)
    round_small = (a <= 10) & (a == a.round(0))
    return df[just_below | card_test | round_small]


def merchant_spikes(df, lookback_hours=168, ratio=3.0):
    h = df.copy()
    h["hour_bucket"] = h["timestamp"].dt.floor("h")

    hourly = (
        h.groupby(["merchant_name", "hour_bucket"])
        .agg(unique_cards=("card_id", "nunique"),
             total_tx=("transaction_id", "count"),
             total_amount=("amount", "sum"))
        .reset_index()
        .sort_values("hour_bucket")
    )

    # rolling avg of unique cards over prior week, per merchant
    hourly["rolling_avg"] = (
        hourly.groupby("merchant_name")["unique_cards"]
        .transform(lambda s: s.shift().rolling(lookback_hours, min_periods=4).mean())
    )
    hourly["spike_ratio"] = hourly["unique_cards"] / hourly["rolling_avg"].replace(0, np.nan)
    return hourly[hourly["spike_ratio"] > ratio].sort_values("spike_ratio", ascending=False)


def off_hours(df, min_history_days=60, min_count=2):
    # build each card's "normal" active hour band (hours with >= min_count tx)
    pat = (
        df.groupby(["card_id", "hour_of_day"])
        .size()
        .reset_index(name="c")
    )
    pat = pat[pat["c"] >= min_count]
    band = pat.groupby("card_id")["hour_of_day"].agg(
        earliest="min", latest="max"
    ).reset_index()

    m = df.merge(band, on="card_id", how="inner")
    flagged = m[
        (m["hour_of_day"] < m["earliest"]) | (m["hour_of_day"] > m["latest"])
    ]
    return flagged


def add_window_features(df):
    g = df.sort_values("timestamp").copy()
    grp = g.groupby("card_id")

    g["time_since_last"] = grp["timestamp"].diff().dt.total_seconds()
    g["merchant_change"] = (
        g["merchant_name"] != grp["merchant_name"].shift()
    ).astype(int)
    g["device_change"] = (g["device_id"] != grp["device_id"].shift()).astype(int)

    # rolling 24h spend per card
    g["running_24h_amount"] = (
        g.set_index("timestamp")
        .groupby("card_id")["amount"]
        .transform(lambda s: s.rolling("24h").sum())
        .values
    )
    g["tx_of_day"] = g.groupby(["card_id", "date"]).cumcount() + 1
    return g



for w, t in [("1min", 3), ("5min", 5), ("1h", 10)]:
    hits = velocity(df, w, t)
    print(w, "→", len(hits), "flagged rows")

w = add_window_features(df)
card_testing = w[
    (w["tx_of_day"] >= 5)
    & (w["time_since_last"] < 60)
    & (w["merchant_change"] == 1)
]

def score(df):
    w = add_window_features(df)
    w["s_velocity"] = (w["tx_of_day"] >= 5) & (w["time_since_last"] < 60)
    w["s_amount"] = w["transaction_id"].isin(amount_anomalies(df)["transaction_id"])
    w["s_geo"] = w["transaction_id"].isin(geo_anomalies(df)["transaction_id"])
    w["s_offhours"] = w["transaction_id"].isin(off_hours(df)["transaction_id"])

    flags = ["s_velocity", "s_amount", "s_geo", "s_offhours"]
    w["fraud_score"] = w[flags].sum(axis=1)
    return w.sort_values("fraud_score", ascending=False)