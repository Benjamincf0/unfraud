import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from algo.lgbm_params import (
    DEFAULT_LGBM_PARAMS,
    load_lgbm_params,
    merge_lgbm_params,
    save_best_params,
)


def test_merge_lgbm_params_ignores_train_only_keys():
    merged = merge_lgbm_params({"n_estimators": 100, "scale_pos_weight": 99})
    assert merged["n_estimators"] == 100
    assert "scale_pos_weight" not in merged


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "best_lgbm_params.json")
        save_best_params(
            {"num_leaves": 48, "learning_rate": 0.05},
            path,
            metadata={"best_value": 0.91, "n_trials": 10},
        )
        loaded = load_lgbm_params(path)
        assert loaded["num_leaves"] == 48
        assert loaded["learning_rate"] == 0.05
        assert loaded["n_estimators"] == DEFAULT_LGBM_PARAMS["n_estimators"]
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["best_value"] == 0.91
        assert data["metric"] == "val_pr_auc"


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_lgbm_params("/nonexistent/best_lgbm_params.json")
