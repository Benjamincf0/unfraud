from pathlib import Path

import pandas as pd

from fraud_scorer import simple_fraud_detection


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "transactions.csv"
OUTPUT_PATH = ROOT / "analyzed_transactions.csv"


def main() -> None:
    transactions = pd.read_csv(INPUT_PATH)
    analyzed = simple_fraud_detection(transactions)
    analyzed["review_decision"] = ""
    analyzed["reviewer_notes"] = ""
    analyzed["reviewed_at"] = ""
    analyzed.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {len(analyzed)} rows")


if __name__ == "__main__":
    main()
