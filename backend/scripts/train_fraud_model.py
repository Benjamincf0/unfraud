"""Train the hybrid fraud model and write ``algo/ops/fraud_model.pkl``."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from algo.algo import DEFAULT_MODEL_PATH, FraudDetectionPipeline


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(_BACKEND / "fraudTrain_part1.csv")
    pipeline = FraudDetectionPipeline()
    pipeline.fit(csv_path)
    saved = pipeline.save(DEFAULT_MODEL_PATH)
    print(f"Saved model artifact to {saved}")


if __name__ == "__main__":
    main()
