"""
Offline LightGBM hyperparameter search with Optuna (not used by FraudDetectionPipeline).

Optimizes validation PR-AUC under the same temporal split as ``algo.temporal_split``.
Writes ``ops/best_lgbm_params.json``; optional SQLite study DB for resume.

Run from ``backend/``:

    uv sync --extra tune
    uv run --extra tune python -m algo.tune_lgbm fraudTrain_part1.csv --n-trials 40

Later pipeline integration (manual):

    from algo.lgbm_params import load_lgbm_params
    from algo.algo import train_model
    model = train_model(X_tr, y_tr, params=load_lgbm_params())
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from sklearn.metrics import average_precision_score

from algo.algo import (
    CATEGORICAL,
    apply_rule_guardrails,
    build_features,
    load,
    prepare_matrix,
    shrink,
    temporal_split,
    train_model,
    validate_dataset_labels,
)
from algo.lgbm_params import (
    DEFAULT_BEST_PARAMS_PATH,
    DEFAULT_OPTUNA_DB_PATH,
    merge_lgbm_params,
    save_best_params,
)

STUDY_NAME = "lgbm_val_pr_auc"


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def prepare_temporal_matrices(
    csv_path: Path,
    *,
    train_frac: float,
    val_frac: float,
) -> Tuple[Any, Any, Any, Any, Dict[str, int]]:
    """Load CSV, build features once, return train/val matrices and split sizes."""
    df = shrink(load(str(csv_path)))
    validate_dataset_labels(df, verbose=False)
    g = apply_rule_guardrails(build_features(df))
    train, val, _test = temporal_split(g, train_frac=train_frac, val_frac=val_frac)
    X_tr, y_tr = prepare_matrix(train)
    X_val, y_val = prepare_matrix(val)
    sizes = {"train": len(train), "val": len(val), "test": len(_test)}
    return X_tr, y_tr, X_val, y_val, sizes


def suggest_lgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, step=10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
    }


def run_study(
    X_tr,
    y_tr,
    X_val,
    y_val,
    *,
    n_trials: int,
    storage_url: str | None,
    study_name: str,
    show_progress: bool,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        model = train_model(X_tr, y_tr, params=params)
        scores = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, scores))

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_url,
        load_if_exists=storage_url is not None,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )
    return study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optuna search for LightGBM params (maximize validation PR-AUC)."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Training CSV with is_fraud labels (e.g. fraudTrain_part1.csv)",
    )
    parser.add_argument("--n-trials", type=int, default=40, help="Number of Optuna trials")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BEST_PARAMS_PATH,
        help="Where to write best_lgbm_params.json",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_OPTUNA_DB_PATH,
        help="SQLite path for Optuna study resume (use --no-storage to disable)",
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Ephemeral study (no optuna.db); trials are not resumed",
    )
    parser.add_argument("--study-name", default=STUDY_NAME)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    print(f"Loading and featurizing {csv_path.name} ...")
    X_tr, y_tr, X_val, y_val, sizes = prepare_temporal_matrices(
        csv_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    feat_sec = time.perf_counter() - t0
    print(f"Feature pipeline: {feat_sec:.1f}s | train={sizes['train']:,} val={sizes['val']:,}")

    storage_url = None
    if not args.no_storage:
        args.storage.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{args.storage.resolve().as_posix()}"

    print(f"Starting study ({args.n_trials} trials, metric=val PR-AUC) ...")
    t1 = time.perf_counter()
    study = run_study(
        X_tr,
        y_tr,
        X_val,
        y_val,
        n_trials=args.n_trials,
        storage_url=storage_url,
        study_name=args.study_name,
        show_progress=not args.no_progress,
    )
    tune_sec = time.perf_counter() - t1
    best_params = merge_lgbm_params(study.best_trial.params)

    out_path = save_best_params(
        best_params,
        args.output,
        metadata={
            "best_value": round(float(study.best_value), 6),
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "n_trials_requested": args.n_trials,
            "dataset_path": str(csv_path),
            "dataset_sha256": _file_sha256(csv_path),
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "split_sizes": sizes,
            "study_name": args.study_name,
            "storage": storage_url,
            "feature_seconds": round(feat_sec, 2),
            "tune_seconds": round(tune_sec, 2),
        },
    )

    print(f"\nBest validation PR-AUC: {study.best_value:.4f} (trial {study.best_trial.number})")
    print("Best params:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")
    print(f"\nWrote {out_path}")
    if storage_url:
        print(f"Study DB: {args.storage.resolve()}")
    print(
        "\nPipeline integration (when needed):\n"
        "  from algo.lgbm_params import load_lgbm_params\n"
        "  from algo.algo import train_model\n"
        "  model = train_model(X_tr, y_tr, params=load_lgbm_params())"
    )
    print(f"\nTotal: {feat_sec + tune_sec:.1f}s (features {feat_sec:.1f}s + trials {tune_sec:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
