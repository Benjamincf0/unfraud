import type { TransactionFlag } from '../types'

export function mergeTransactionMaps(
  current: Map<string, TransactionFlag>,
  items: TransactionFlag[],
): Map<string, TransactionFlag> {
  if (items.length === 0) {
    return current
  }

  const next = new Map(current)

  for (const item of items) {
    const existing = next.get(item.transactionId)
    next.set(
      item.transactionId,
      existing
        ? {
            ...existing,
            ...item,
            decision: existing.decision,
            reviewedAt: existing.reviewedAt ?? item.reviewedAt,
            reviewerNotes: existing.reviewerNotes ?? item.reviewerNotes,
          }
        : item,
    )
  }

  return next
}

export function applyModelScores(
  heuristic: TransactionFlag,
  model: TransactionFlag,
): TransactionFlag {
  return {
    ...heuristic,
    score: model.score,
    label: model.label,
    isFraud: model.isFraud,
    reasons: model.reasons,
  }
}
