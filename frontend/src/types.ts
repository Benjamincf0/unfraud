export type ReviewDecision = 'pending' | 'approved' | 'dismissed' | 'escalated'

export type SearchFieldKey =
  | 'transaction_id'
  | 'timestamp'
  | 'card_id'
  | 'amount'
  | 'merchant_name'
  | 'merchant_category'
  | 'channel'
  | 'cardholder_country'
  | 'merchant_country'
  | 'device_id'
  | 'ip_address'

export type MlScoringInfo = {
  modelScore: number | null
  modelThreshold: number | null
  flaggedByModel: boolean
  flaggedByAlert: boolean
  ruleGuardrail: boolean
}

export type RiskReason = {
  id: string
  code?: string
  label: string
  detail: string
  weight: number
  signalType?: 'per_card' | 'cross_card' | 'composite' | string
}

export type DecisionFeedback = {
  reasonCodes: string[]
  reasoning: string
}

export type TransactionFlag = {
  transactionId: string
  timestamp: string
  cardId: string
  amount: number
  merchantName: string
  merchantCategory: string
  channel: 'online' | 'in_person' | 'atm'
  cardholderCountry: string
  merchantCountry: string
  deviceId?: string
  ipAddress?: string
  score: number
  label: 'critical' | 'high' | 'medium' | 'low'
  decision: ReviewDecision
  reviewedAt?: string
  reviewerNotes?: string
  reasons: RiskReason[]
  isFraud: boolean
  mlScoring?: MlScoringInfo
  cardContext: {
    medianAmount: number
    usualCountries: string[]
    usualCategories: string[]
    previousTransactions: number
  }
}

export type ReviewLogEntry = {
  transactionId: string
  action: Exclude<ReviewDecision, 'pending'>
  reviewerNotes?: string
  feedbackEffects: ReviewFeedbackEffect[]
  reviewedAt: string
}

export type ReviewFeedbackEffect = {
  type: string
  signalCode: string
  signalLabel: string
  direction: string
  previousMultiplier: number
  nextMultiplier: number
  summary: string
}

export type CardTransaction = {
  transactionId: string
  timestamp: string
  amount: number
  merchantName: string
  merchantCategory: string
  channel: 'online' | 'in_person' | 'atm'
  cardholderCountry: string
  merchantCountry: string
  deviceId?: string
  ipAddress?: string
  score: number
  isFraud: boolean
  reasons: RiskReason[]
}

export type CardAnalysis = {
  cardId: string
  transactions: CardTransaction[]
  summary: {
    totalSpend: number
    medianAmount: number
    highestAmount: number
    fraudCount: number
    transactionCount: number
    categories: string[]
    countries: string[]
  }
}

export type DecisionAction = {
  transactionId: string
  previousDecision: ReviewDecision
  nextDecision: Exclude<ReviewDecision, 'pending'>
  actedAt: string
}
