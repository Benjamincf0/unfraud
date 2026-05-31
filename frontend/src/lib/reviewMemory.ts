import type { TransactionFlag } from '../types'

const GENERIC_REASON_DETAILS = new Set([
  'Backend detector returned a positive score without a specific reason.',
  'Backend detector returned this signal.',
  'Flag returned by the backend fraud detector.',
])

export function hasDetailedReasons(transaction: TransactionFlag) {
  return transaction.reasons.some(
    (reason) => !GENERIC_REASON_DETAILS.has(reason.detail),
  )
}

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
    const preserveScoring =
      existing !== undefined &&
      hasDetailedReasons(existing) &&
      !hasDetailedReasons(item)

    next.set(
      item.transactionId,
      existing
        ? {
            ...existing,
            ...item,
            decision: existing.decision,
            reviewedAt: existing.reviewedAt ?? item.reviewedAt,
            reviewerNotes: existing.reviewerNotes ?? item.reviewerNotes,
            ...(preserveScoring
              ? {
                  cardContext: existing.cardContext,
                  isFraud: existing.isFraud,
                  label: existing.label,
                  reasons: existing.reasons,
                  score: existing.score,
                }
              : {}),
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
