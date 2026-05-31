# Product Requirements

## User

The primary user is a trust and safety reviewer at a payments company. They are responsible for quickly triaging suspected card fraud without blocking legitimate customer activity.

## Problem

The reviewer cannot inspect every transaction manually. They need a short, high-signal queue that explains why each transaction is suspicious and supports fast decisions: approve, dismiss, or escalate.

## Goals

- Ingest the provided `transactions.csv` and process all 1,000 transactions.
- Rank suspicious transactions by risk score.
- Explain every alert in human-readable language.
- Let a reviewer work the queue with keyboard controls and undo.
- Preserve reviewer decisions in an audit trail and CSV export.
- Keep the app runnable and understandable from a clean clone.

## Success Measures

- A reviewer can upload the CSV and start triage without reading source code.
- Flagged transactions include at least one concrete reason.
- The queue is short enough to review quickly while still catching diverse fraud patterns.
- Review decisions are visible in the session and included in exported results.
- Backend tests and frontend build pass from documented commands.

## Non-Goals

- Building a production identity, permissions, or case-management system.
- Persisting data beyond the current backend process.
- Training a supervised production model on unrelated public fraud datasets.
- Claiming precision/recall/F1 without access to the hidden answer key.
- Blocking payments automatically.
