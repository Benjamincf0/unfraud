import type { TransactionFlag } from '../types'

export type ReviewDataResult = {
  items: TransactionFlag[]
  source: 'upload'
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? ''

export async function uploadTransactionsCsv(file: File): Promise<ReviewDataResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${apiBaseUrl}/api/review/upload`, {
    body: formData,
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(`Upload failed with status ${response.status}`)
  }

  const payload = (await response.json()) as unknown
  const items = normalizeReviewPayload(payload)

  if (!items) {
    throw new Error('Upload returned an invalid review payload')
  }

  return { items, source: 'upload' }
}

function normalizeReviewPayload(payload: unknown) {
  if (isTransactionFlagList(payload)) {
    return payload
  }

  if (payload && typeof payload === 'object' && 'items' in payload) {
    const candidate = (payload as { items: unknown }).items
    return isTransactionFlagList(candidate) ? candidate : null
  }

  return null
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
