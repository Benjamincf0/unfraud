import type { MlScoringInfo, TransactionFlag } from '../types'

export type ScoringTier = 'queued' | 'elevated' | 'low'

export type { MlScoringInfo }

type BackendMlFields = {
  model_score?: number | null
  model_threshold?: number | null
  flagged_by_model?: boolean | null
  flagged_by_alert?: boolean | null
  rule_guardrail?: boolean | null
}

export function mlScoringFromBackend(
  fields: BackendMlFields,
): MlScoringInfo | undefined {
  if (fields.model_score == null && fields.flagged_by_model == null) {
    return undefined
  }

  return {
    modelScore: fields.model_score ?? null,
    modelThreshold: fields.model_threshold ?? null,
    flaggedByModel: Boolean(fields.flagged_by_model),
    flaggedByAlert: Boolean(fields.flagged_by_alert),
    ruleGuardrail: Boolean(fields.rule_guardrail),
  }
}

export function getScoringTier(transaction: TransactionFlag): ScoringTier {
  if (transaction.isFraud) {
    return 'queued'
  }

  if (transaction.mlScoring?.ruleGuardrail || transaction.score >= 0.35) {
    return 'elevated'
  }

  return 'low'
}

export function queueCauseLabel(transaction: TransactionFlag): string | null {
  const ml = transaction.mlScoring
  if (!transaction.isFraud || !ml) {
    return transaction.isFraud ? 'In review queue' : null
  }

  if (ml.flaggedByModel && ml.flaggedByAlert) {
    return 'Model + alert rule'
  }
  if (ml.flaggedByModel) {
    return 'Model threshold'
  }
  if (ml.flaggedByAlert) {
    return 'Alert rule'
  }
  if (!ml.flaggedByModel && !ml.flaggedByAlert) {
    return 'Strong heuristic score'
  }
  return 'In review queue'
}

export type QueueCauseFilter = 'all' | 'model' | 'alert' | 'both'

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

export function passesQueueCauseFilter(
  transaction: TransactionFlag,
  filter: QueueCauseFilter,
): boolean {
  if (filter === 'all') {
    return true
  }

  const ml = transaction.mlScoring
  if (!transaction.isFraud || !ml) {
    return false
  }

  if (filter === 'model') {
    return ml.flaggedByModel && !ml.flaggedByAlert
  }
  if (filter === 'alert') {
    return ml.flaggedByAlert && !ml.flaggedByModel
  }
  return ml.flaggedByModel && ml.flaggedByAlert
}
