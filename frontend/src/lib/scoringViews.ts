import type { ReviewDecision, TransactionFlag } from '../types'

export type RiskSortMode = 'active' | 'heuristic' | 'model'

export type ScoringSnapshot = {
  allItems: TransactionFlag[]
  items: TransactionFlag[]
}

export type DualReviewDataResult = {
  fileHash: string
  heuristic: ScoringSnapshot
  mlModelAvailable: boolean
  model: ScoringSnapshot | null
  source: 'cache' | 'upload'
}

export type RiskTuningState = {
  falsePositiveCost: number
  riskThreshold: number
}

export type RiskTuningByMode = Record<RiskSortMode, RiskTuningState>

export const defaultRiskTuning: RiskTuningByMode = {
  active: { falsePositiveCost: 5, riskThreshold: 0 },
  heuristic: { falsePositiveCost: 5, riskThreshold: 0 },
  model: { falsePositiveCost: 5, riskThreshold: 0 },
}

const neutralFalsePositiveCost = 5

export function getEffectiveRiskThreshold(
  riskThreshold: number,
  falsePositiveCost: number,
) {
  return Math.min(
    95,
    Math.max(
      0,
      riskThreshold + (falsePositiveCost - neutralFalsePositiveCost) * 5,
    ),
  )
}

export function buildTransactionIndex(items: TransactionFlag[]) {
  return new Map(items.map((transaction) => [transaction.transactionId, transaction]))
}

export function mergeScoringWithDecisions(
  scoringItems: TransactionFlag[],
  decisions: Map<string, ReviewDecision>,
): TransactionFlag[] {
  return scoringItems.map((transaction) => {
    const decision = decisions.get(transaction.transactionId)

    return decision === undefined
      ? transaction
      : { ...transaction, decision }
  })
}

export function collectDecisions(transactions: TransactionFlag[]) {
  return new Map(
    transactions.map((transaction) => [
      transaction.transactionId,
      transaction.decision,
    ]),
  )
}

export function sortTransactionsByScore(
  transactions: TransactionFlag[],
  scoreById: Map<string, number>,
) {
  return [...transactions].sort((first, second) => {
    const firstScore = scoreById.get(first.transactionId) ?? first.score
    const secondScore = scoreById.get(second.transactionId) ?? second.score

    return secondScore - firstScore
  })
}

export function resolveRiskSortMode(
  sortMode: RiskSortMode,
  useModel: boolean,
): Exclude<RiskSortMode, 'active'> {
  if (sortMode === 'heuristic' || sortMode === 'model') {
    return sortMode
  }

  return useModel ? 'model' : 'heuristic'
}

export function getScoreSource(
  data: DualReviewDataResult,
  mode: Exclude<RiskSortMode, 'active'>,
): ScoringSnapshot | null {
  if (mode === 'model') {
    return data.model
  }

  return data.heuristic
}
