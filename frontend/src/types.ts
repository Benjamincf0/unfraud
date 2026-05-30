export type ReviewDecision = 'pending' | 'approved' | 'dismissed' | 'escalated'

export type RiskReason = {
  id: string
  label: string
  detail: string
  weight: number
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
  reasons: RiskReason[]
  cardContext: {
    medianAmount: number
    usualCountries: string[]
    usualCategories: string[]
    previousTransactions: number
  }
}

export type AuditEntry = {
  id: string
  transactionId: string
  decision: Exclude<ReviewDecision, 'pending'>
  timestamp: string
}

export type DecisionAction = {
  transactionId: string
  previousDecision: ReviewDecision
  nextDecision: Exclude<ReviewDecision, 'pending'>
}
