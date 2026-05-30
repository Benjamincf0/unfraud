export type ReviewSession = {
  fileHash: string
  label: string
  uploadedAt: string
}

const storageKey = 'fraud-hunter-review-sessions'
const activeStorageKey = 'fraud-hunter-active-review-session'

export function loadReviewSessions(): ReviewSession[] {
  try {
    const rawValue = window.localStorage.getItem(storageKey)
    const parsedValue = rawValue ? (JSON.parse(rawValue) as unknown) : []

    return Array.isArray(parsedValue)
      ? parsedValue.filter(isReviewSession)
      : []
  } catch {
    return []
  }
}

export function saveReviewSession(session: ReviewSession) {
  const sessions = [
    session,
    ...loadReviewSessions().filter((item) => item.fileHash !== session.fileHash),
  ]

  window.localStorage.setItem(storageKey, JSON.stringify(sessions))
  saveActiveReviewSession(session.fileHash)

  return sessions
}

export function loadActiveReviewSession() {
  return window.localStorage.getItem(activeStorageKey)
}

export function saveActiveReviewSession(fileHash: string) {
  window.localStorage.setItem(activeStorageKey, fileHash)
}

function isReviewSession(value: unknown): value is ReviewSession {
  if (!value || typeof value !== 'object') {
    return false
  }

  const candidate = value as Partial<ReviewSession>

  return (
    typeof candidate.fileHash === 'string' &&
    typeof candidate.label === 'string' &&
    typeof candidate.uploadedAt === 'string'
  )
}
