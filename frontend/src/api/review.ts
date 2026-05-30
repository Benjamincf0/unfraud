import { mockReviewItems } from '../data/mockReviewItems'
import type { TransactionFlag } from '../types'

export type ReviewDataResult = {
  items: TransactionFlag[]
  source: 'api' | 'sample'
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? ''

export async function loadReviewItems(): Promise<ReviewDataResult> {
  try {
    const response = await fetch(`${apiBaseUrl}/api/review/flags`)

    if (!response.ok) {
      throw new Error(`Review API returned ${response.status}`)
    }

    const payload = (await response.json()) as unknown

    if (!isTransactionFlagList(payload)) {
      throw new Error('Review API returned an invalid payload')
    }

    return { items: payload, source: 'api' }
  } catch {
    return { items: mockReviewItems, source: 'sample' }
  }
}

function isTransactionFlagList(payload: unknown): payload is TransactionFlag[] {
  return (
    Array.isArray(payload) &&
    payload.every((item) => {
      if (!item || typeof item !== 'object') {
        return false
      }

      const candidate = item as Partial<TransactionFlag>

      return (
        typeof candidate.transactionId === 'string' &&
        typeof candidate.timestamp === 'string' &&
        typeof candidate.cardId === 'string' &&
        typeof candidate.amount === 'number' &&
        typeof candidate.merchantName === 'string' &&
        typeof candidate.score === 'number' &&
        Array.isArray(candidate.reasons)
      )
    })
  )
}
