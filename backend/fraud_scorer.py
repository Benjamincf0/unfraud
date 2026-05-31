"""Production fraud scorer for the challenge dataset (no FastAPI dependency)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

# Challenge dataset ~1% fraud — keep the reviewer queue focused.
FRAUD_SCORE_THRESHOLD = 0.55


def _top_values(series: pd.Series, limit: int = 4) -> List[str]:
    values = series.dropna().astype(str)
    if values.empty:
        return []
    return values.value_counts().head(limit).index.tolist()


def _build_card_amount_series(df: pd.DataFrame, points: int = 12) -> Dict[str, List[List[Dict[str, Any]]]]:
    series_map: Dict[str, List[List[Dict[str, Any]]]] = {}
    for card_id, card_df in df.groupby("card_id", sort=False):
        card_df = card_df.sort_values("timestamp_dt")
        running: List[Dict[str, Any]] = []
        snapshots: List[List[Dict[str, Any]]] = []
        for _, row in card_df.iterrows():
            running.append(
                {
                    "timestamp": row["timestamp_dt"].isoformat(),
                    "amount": float(row["amount"]),
                    "risk_score": round(float(row["fraud_score"]), 4),
                }
            )
            snapshots.append(running[-points:].copy())
        series_map[str(card_id)] = snapshots
    return series_map


def simple_fraud_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card anomaly scoring + cross-card aggregation + explainable output."""
    working = df.copy()
    if working.empty:
        for column, default_value in [
            ("is_fraud", False),
            ("fraud_score", 0.0),
            ("fraud_reasons", ""),
            ("score_breakdown", "[]"),
            ("card_baseline_json", "{}"),
            ("cross_card_signals_json", "{}"),
            ("graph_features_json", "{}"),
            ("card_amount_series_json", "[]"),
        ]:
            working[column] = default_value
        return working

    if "device_id" not in working.columns:
        working["device_id"] = pd.NA
    if "ip_address" not in working.columns:
        working["ip_address"] = pd.NA

    working["device_id"] = working["device_id"].replace("", pd.NA)
    working["ip_address"] = working["ip_address"].replace("", pd.NA)
    working["timestamp_dt"] = pd.to_datetime(working["timestamp"], errors="coerce")
    if working["timestamp_dt"].isna().any():
        raise ValueError("Invalid timestamp format in CSV")

    working = working.sort_values(["timestamp_dt", "transaction_id"]).reset_index(drop=True)

    card_group = working.groupby("card_id", sort=False)
    working["card_tx_index"] = card_group.cumcount()
    working["card_median_before"] = card_group["amount"].transform(
        lambda s: s.shift().expanding().median()
    )
    working["card_mean_before"] = card_group["amount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    working["card_std_before"] = card_group["amount"].transform(
        lambda s: s.shift().expanding().std()
    )
    working["card_median_before"] = working["card_median_before"].fillna(working["amount"])
    working["card_mean_before"] = working["card_mean_before"].fillna(working["amount"])
    working["card_std_before"] = working["card_std_before"].fillna(0.0)

    working["amount_ratio"] = (
        working["amount"] / working["card_median_before"].replace(0, pd.NA)
    ).fillna(1.0)
    working["amount_zscore"] = (
        (working["amount"] - working["card_mean_before"])
        / working["card_std_before"].replace(0, pd.NA)
    ).fillna(0.0)

    card_cat_group = working.groupby(["card_id", "merchant_category"], sort=False)
    working["card_cat_median_before"] = card_cat_group["amount"].transform(
        lambda s: s.shift().expanding().median()
    )
    working["card_cat_mean_before"] = card_cat_group["amount"].transform(
        lambda s: s.shift().expanding().mean()
    )
    working["card_cat_std_before"] = card_cat_group["amount"].transform(
        lambda s: s.shift().expanding().std()
    )
    working["card_cat_median_before"] = working["card_cat_median_before"].fillna(
        working["card_median_before"]
    )
    working["card_cat_mean_before"] = working["card_cat_mean_before"].fillna(
        working["card_mean_before"]
    )
    working["card_cat_std_before"] = working["card_cat_std_before"].fillna(working["card_std_before"])

    working["amount_ratio_category"] = (
        working["amount"] / working["card_cat_median_before"].replace(0, pd.NA)
    ).fillna(1.0)
    working["amount_zscore_category"] = (
        (working["amount"] - working["card_cat_mean_before"])
        / working["card_cat_std_before"].replace(0, pd.NA)
    ).fillna(0.0)

    category_seen_before = working.groupby(["card_id", "merchant_category"], sort=False).cumcount()
    device_seen_before = working.groupby(["card_id", "device_id"], sort=False).cumcount()
    ip_seen_before = working.groupby(["card_id", "ip_address"], sort=False).cumcount()

    history_nonzero = working["card_tx_index"].replace(0, pd.NA)
    working["category_seen_rate"] = (category_seen_before / history_nonzero).fillna(1.0)
    working["novel_category"] = (
        (working["card_tx_index"] >= 3) & (category_seen_before == 0)
    ).astype(int)
    working["novel_device"] = (
        (working["channel"] == "online")
        & (working["card_tx_index"] >= 3)
        & working["device_id"].notna()
        & (device_seen_before == 0)
    ).astype(int)
    working["novel_ip"] = (
        (working["channel"] == "online")
        & (working["card_tx_index"] >= 3)
        & working["ip_address"].notna()
        & (ip_seen_before == 0)
    ).astype(int)

    working["foreign_country"] = (
        (working["cardholder_country"] != working["merchant_country"])
        & working["cardholder_country"].notna()
        & working["merchant_country"].notna()
    ).astype(int)
    country_seen_before = working.groupby(["card_id", "merchant_country"], sort=False).cumcount()
    working["novel_country"] = (
        (working["card_tx_index"] >= 4) & (country_seen_before == 0)
    ).astype(int)

    working["device_card_fanout"] = (
        working.groupby("device_id")["card_id"].transform("nunique").fillna(1)
    )
    working.loc[working["device_id"].isna(), "device_card_fanout"] = 1
    working["ip_card_fanout"] = (
        working.groupby("ip_address")["card_id"].transform("nunique").fillna(1)
    )
    working.loc[working["ip_address"].isna(), "ip_card_fanout"] = 1

    merchant_tx_30m = pd.Series(1.0, index=working.index)
    merchant_unique_cards_2h = pd.Series(1.0, index=working.index)
    card_codes = pd.Categorical(working["card_id"]).codes.astype(float)
    working["card_code"] = card_codes
    for merchant, merchant_df in working.groupby("merchant_name", sort=False):
        sorted_merchant = merchant_df.sort_values("timestamp_dt")
        merchant_view = sorted_merchant.set_index("timestamp_dt")
        merchant_tx_30m.loc[sorted_merchant.index] = (
            merchant_view["transaction_id"].rolling("30min").count().values
        )
        merchant_unique_cards_2h.loc[sorted_merchant.index] = merchant_view["card_code"].rolling("2h").apply(
            lambda values: len(set(values)), raw=True
        ).values
        _ = merchant

    working["merchant_tx_30m"] = merchant_tx_30m
    working["merchant_unique_cards_2h"] = merchant_unique_cards_2h

    amount_risk = ((working["amount_ratio"] - 1.6) / 5.0).clip(lower=0, upper=1)
    amount_z_risk = ((working["amount_zscore"].abs() - 2.2) / 3.0).clip(lower=0, upper=1)
    cat_z_capped = working["amount_zscore_category"].abs().clip(upper=8.0)
    category_amount_z_risk = ((cat_z_capped - 2.5) / 3.5).clip(lower=0, upper=1)
    category_amount_ratio_risk = (
        (working["amount_ratio_category"] - 3.0) / 5.0
    ).clip(lower=0, upper=1)
    category_risk = (1 - working["category_seen_rate"]).clip(lower=0, upper=1) * working["novel_category"]
    device_novelty_risk = working["novel_device"] * 1.0
    ip_novelty_risk = working["novel_ip"] * 1.0
    corroborated = (
        (working["amount_zscore"].abs() >= 2.0)
        | (working["amount_zscore_category"].abs() >= 2.5)
        | (working["amount_ratio"] >= 2.5)
        | (working["amount_ratio_category"] >= 3.0)
        | (working["novel_device"] == 1)
        | (working["novel_ip"] == 1)
        | (working["novel_category"] == 1)
    )
    country_risk = (
        working["novel_country"].astype(float)
        + working["foreign_country"].astype(float) * corroborated.astype(float) * 0.35
    ).clip(0, 1)
    device_reuse_risk = ((working["device_card_fanout"] - 1) / 4.0).clip(lower=0, upper=1)
    ip_reuse_risk = ((working["ip_card_fanout"] - 1) / 5.0).clip(lower=0, upper=1)
    merchant_burst_risk = (
        ((working["merchant_tx_30m"] - 3) / 8.0).clip(lower=0, upper=1) * 0.4
        + ((working["merchant_unique_cards_2h"] - 2) / 6.0).clip(lower=0, upper=1) * 0.6
    ).clip(lower=0, upper=1)

    components = {
        "amount_outlier": 0.18 * amount_risk + 0.08 * amount_z_risk,
        "category_amount": 0.14 * category_amount_z_risk + 0.12 * category_amount_ratio_risk,
        "category_shift": 0.07 * category_risk,
        "device_shift": 0.10 * device_novelty_risk,
        "ip_shift": 0.09 * ip_novelty_risk,
        "geo_shift": 0.07 * country_risk,
        "device_reuse_cross_card": 0.11 * device_reuse_risk,
        "ip_reuse_cross_card": 0.09 * ip_reuse_risk,
        "merchant_burst_cross_card": 0.14 * merchant_burst_risk,
    }

    score = sum(components.values()).clip(lower=0, upper=1)
    has_category_history = category_seen_before >= 1
    high_confidence_rule = (
        (working["amount_ratio"] >= 7.0)
        | (
            has_category_history
            & (working["amount_ratio_category"] >= 6.0)
            & (cat_z_capped >= 3.5)
        )
        | (working["ip_card_fanout"] >= 4)
        | (working["merchant_unique_cards_2h"] >= 7)
    )
    working["fraud_score"] = score.round(4)
    working["is_fraud"] = (working["fraud_score"] >= FRAUD_SCORE_THRESHOLD) | high_confidence_rule

    top_categories = {
        card_id: _top_values(group["merchant_category"])
        for card_id, group in working.groupby("card_id", sort=False)
    }
    top_countries = {
        card_id: _top_values(group["merchant_country"])
        for card_id, group in working.groupby("card_id", sort=False)
    }
    top_devices = {
        card_id: _top_values(group["device_id"])
        for card_id, group in working.groupby("card_id", sort=False)
    }
    top_ips = {
        card_id: _top_values(group["ip_address"])
        for card_id, group in working.groupby("card_id", sort=False)
    }

    rows_breakdown: List[List[Dict[str, Any]]] = []
    rows_baseline: List[Dict[str, Any]] = []
    rows_cross_card: List[Dict[str, Any]] = []
    rows_graph_features: List[Dict[str, float]] = []
    rows_reason_labels: List[str] = []

    for _, row in working.iterrows():
        breakdown: List[Dict[str, Any]] = []

        def add_reason(
            code: str,
            label: str,
            detail: str,
            contribution: float,
            signal_type: str,
            value: Optional[float] = None,
            baseline: Optional[float] = None,
        ):
            if contribution < 0.03:
                return
            breakdown.append(
                {
                    "code": code,
                    "label": label,
                    "detail": detail,
                    "weight": round(float(contribution), 4),
                    "signal_type": signal_type,
                    "value": None if value is None else round(float(value), 4),
                    "baseline": None if baseline is None else round(float(baseline), 4),
                }
            )

        add_reason(
            "amount_outlier",
            "Amount anomaly",
            (
                f"Amount is {row['amount_ratio']:.2f}× this card's historical median "
                f"({row['card_median_before']:.2f})."
            ),
            components["amount_outlier"].loc[row.name],
            "per_card",
            row["amount"],
            row["card_median_before"],
        )

        add_reason(
            "category_shift",
            "Atypical category",
            (
                f"Category '{row['merchant_category']}' is uncommon for this card "
                f"(seen rate {row['category_seen_rate']:.2f})."
            ),
            components["category_shift"].loc[row.name],
            "per_card",
            row["category_seen_rate"],
            1.0,
        )

        add_reason(
            "category_amount",
            "Category amount anomaly",
            (
                f"Amount is {row['amount_ratio_category']:.2f}× this card's median at "
                f"'{row['merchant_category']}' ({row['card_cat_median_before']:.2f}), "
                f"z={row['amount_zscore_category']:.2f} within category."
            ),
            components["category_amount"].loc[row.name],
            "per_card",
            row["amount"],
            row["card_cat_median_before"],
        )

        if row["novel_device"] == 1:
            add_reason(
                "device_shift",
                "New device for card",
                f"Online transaction from unseen device '{row['device_id']}'.",
                components["device_shift"].loc[row.name],
                "per_card",
            )
        if row["novel_ip"] == 1:
            add_reason(
                "ip_shift",
                "New IP for card",
                f"Online transaction from unseen IP '{row['ip_address']}'.",
                components["ip_shift"].loc[row.name],
                "per_card",
            )
        if components["geo_shift"].loc[row.name] >= 0.03:
            if row["novel_country"] == 1:
                geo_detail = (
                    f"First time this card transacts in merchant country "
                    f"'{row['merchant_country']}'."
                )
            else:
                geo_detail = (
                    f"Cross-border ({row['cardholder_country']}→{row['merchant_country']}) "
                    f"combined with amount or identity anomaly."
                )
            add_reason(
                "geo_shift",
                "Location deviation",
                geo_detail,
                components["geo_shift"].loc[row.name],
                "per_card",
            )

        add_reason(
            "device_reuse_cross_card",
            "Device shared across cards",
            f"Device appears on {int(row['device_card_fanout'])} distinct cards.",
            components["device_reuse_cross_card"].loc[row.name],
            "cross_card",
            row["device_card_fanout"],
            1.0,
        )
        add_reason(
            "ip_reuse_cross_card",
            "IP shared across cards",
            f"IP appears on {int(row['ip_card_fanout'])} distinct cards.",
            components["ip_reuse_cross_card"].loc[row.name],
            "cross_card",
            row["ip_card_fanout"],
            1.0,
        )
        add_reason(
            "merchant_burst_cross_card",
            "Merchant burst across cards",
            (
                f"Merchant has {int(row['merchant_tx_30m'])} tx in 30m and "
                f"{int(row['merchant_unique_cards_2h'])} cards in 2h."
            ),
            components["merchant_burst_cross_card"].loc[row.name],
            "cross_card",
            row["merchant_unique_cards_2h"],
            1.0,
        )

        breakdown = sorted(breakdown, key=lambda signal: signal["weight"], reverse=True)
        if not breakdown and row["fraud_score"] > 0:
            breakdown = [
                {
                    "code": "baseline_risk",
                    "label": "Composite anomaly score",
                    "detail": "Combined per-card and cross-card signals elevated risk.",
                    "weight": round(float(row["fraud_score"]), 4),
                    "signal_type": "composite",
                    "value": round(float(row["fraud_score"]), 4),
                    "baseline": 0.0,
                }
            ]

        rows_breakdown.append(breakdown)
        rows_reason_labels.append("; ".join(reason["label"] for reason in breakdown))
        rows_baseline.append(
            {
                "history_count": int(row["card_tx_index"]),
                "typical_amount": round(float(row["card_median_before"]), 2),
                "amount_ratio": round(float(row["amount_ratio"]), 4),
                "amount_zscore": round(float(row["amount_zscore"]), 4),
                "category_typical_amount": round(float(row["card_cat_median_before"]), 2),
                "amount_ratio_category": round(float(row["amount_ratio_category"]), 4),
                "amount_zscore_category": round(float(row["amount_zscore_category"]), 4),
                "usual_categories": top_categories.get(row["card_id"], []),
                "usual_countries": top_countries.get(row["card_id"], []),
                "usual_devices": top_devices.get(row["card_id"], []),
                "usual_ips": top_ips.get(row["card_id"], []),
            }
        )
        rows_cross_card.append(
            {
                "device_card_fanout": int(row["device_card_fanout"]),
                "ip_card_fanout": int(row["ip_card_fanout"]),
                "merchant_tx_30m": int(row["merchant_tx_30m"]),
                "merchant_unique_cards_2h": int(row["merchant_unique_cards_2h"]),
            }
        )
        rows_graph_features.append(
            {
                "amount_ratio": round(float(row["amount_ratio"]), 4),
                "amount_zscore": round(float(row["amount_zscore"]), 4),
                "amount_ratio_category": round(float(row["amount_ratio_category"]), 4),
                "amount_zscore_category": round(float(row["amount_zscore_category"]), 4),
                "card_tx_index": float(row["card_tx_index"]),
                "device_card_fanout": float(row["device_card_fanout"]),
                "ip_card_fanout": float(row["ip_card_fanout"]),
                "merchant_tx_30m": float(row["merchant_tx_30m"]),
                "merchant_unique_cards_2h": float(row["merchant_unique_cards_2h"]),
            }
        )

    working["fraud_reasons"] = rows_reason_labels
    working["score_breakdown"] = [json.dumps(value) for value in rows_breakdown]
    working["card_baseline_json"] = [json.dumps(value) for value in rows_baseline]
    working["cross_card_signals_json"] = [json.dumps(value) for value in rows_cross_card]
    working["graph_features_json"] = [json.dumps(value) for value in rows_graph_features]

    card_series_map = _build_card_amount_series(working)
    card_counters: Dict[str, int] = {}
    series_column: List[str] = []
    for _, row in working.iterrows():
        card_id = str(row["card_id"])
        position = card_counters.get(card_id, 0)
        series = card_series_map[card_id][position]
        series_column.append(json.dumps(series))
        card_counters[card_id] = position + 1
    working["card_amount_series_json"] = series_column

    return working.drop(columns=["timestamp_dt", "card_code"])
