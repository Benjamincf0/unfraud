"""LightGBM hyperparameter defaults and JSON load/save for optional offline tuning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

OPS_DIR = Path(__file__).resolve().parent / "ops"
DEFAULT_BEST_PARAMS_PATH = OPS_DIR / "best_lgbm_params.json"
DEFAULT_OPTUNA_DB_PATH = OPS_DIR / "optuna.db"

# Tunable knobs (scale_pos_weight is swept offline in tune_lgbm; pipeline uses natural weight).
DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_bin": 127,
}

# Set by train_model / pipeline integration — not passed to LGBMClassifier.
TRAIN_ONLY_KEYS = frozenset({"scale_pos_weight", "objective", "n_jobs", "random_state"})


def natural_scale_pos_weight(y) -> float:
    """Inverse class frequency: neg / pos (same as LightGBM ``is_unbalance=True``)."""
    pos = int(np.asarray(y).sum())
    neg = len(y) - pos
    return neg / max(pos, 1)


def merge_lgbm_params(
    overrides: Optional[Dict[str, Any]] = None,
    *,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge overrides onto defaults; ignores train-only keys in overrides."""
    merged = dict(base or DEFAULT_LGBM_PARAMS)
    if overrides:
        for key, value in overrides.items():
            if key not in TRAIN_ONLY_KEYS:
                merged[key] = value
    return merged


def load_lgbm_params(path: Path | str = DEFAULT_BEST_PARAMS_PATH) -> Dict[str, Any]:
    """Load tunable params from a tuning run artifact (``params`` field or flat dict)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No tuned params at {path} — run tune_lgbm first or use defaults.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("params"), dict):
        return merge_lgbm_params(data["params"])
    return merge_lgbm_params(data)


def save_best_params(
    params: Dict[str, Any],
    path: Path | str = DEFAULT_BEST_PARAMS_PATH,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write best params + metadata for reproducibility and later pipeline integration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": 1,
        "metric": "val_pr_auc",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "params": merge_lgbm_params(params),
    }
    if metadata:
        payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
