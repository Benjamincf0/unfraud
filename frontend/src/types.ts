export type ReviewDecision = 'pending' | 'approved' | 'dismissed' | 'escalated'

export type RiskReason = {
  id: string
  label: string
  detail: string
  weight: number
  signalType?: 'per_card' | 'cross_card' | 'composite' | string
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
  cardContext: {
    medianAmount: number
    usualCountries: string[]
    usualCategories: string[]
    previousTransactions: number
  }
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
