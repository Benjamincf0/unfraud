from pathlib import Path

import pandas as pd

from hybrid_scorer import apply_hybrid_decision, build_hybrid_scored_transactions

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "transactions.csv"
OUTPUT_PATH = ROOT / "analyzed_transactions.csv"

# Backward-compatible aliases for tests and imports.
_apply_hybrid_decision = apply_hybrid_decision
build_analyzed_transactions = build_hybrid_scored_transactions


def main() -> None:
    transactions = pd.read_csv(INPUT_PATH)
    analyzed = build_hybrid_scored_transactions(transactions)
    analyzed["review_decision"] = ""
    analyzed["reviewer_notes"] = ""
    analyzed["reviewed_at"] = ""
    analyzed.to_csv(OUTPUT_PATH, index=False)
    flagged_count = int(analyzed["is_fraud"].sum())
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {len(analyzed)} rows "
        f"({flagged_count} flagged)"
    )


if __name__ == "__main__":
    main()
