import type { DecisionFeedback, TransactionFlag } from '../types'

export function getFeedbackReasonCode(
  reason: TransactionFlag['reasons'][number],
  index: number,
) {
  const parts = reason.id.split('-')
  return reason.code ?? parts[parts.length - 1] ?? `reason_${index}`
}

export function defaultDecisionFeedback(
  transaction: TransactionFlag,
): DecisionFeedback {
  const firstReason = transaction.reasons[0]

  return {
    reasonCodes: firstReason ? [getFeedbackReasonCode(firstReason, 0)] : [],
    reasoning: '',
  }
}
