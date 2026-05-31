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
  const fileHash = window.localStorage.getItem(activeStorageKey)

  if (!isNonEmptyString(fileHash)) {
    clearActiveReviewSession()
    return null
  }

  return fileHash
}

export function saveActiveReviewSession(fileHash: string) {
  if (!isNonEmptyString(fileHash)) {
    clearActiveReviewSession()
    return
  }

  window.localStorage.setItem(activeStorageKey, fileHash)
}

export function clearActiveReviewSession() {
  window.localStorage.removeItem(activeStorageKey)
}

function isReviewSession(value: unknown): value is ReviewSession {
  if (!value || typeof value !== 'object') {
    return false
  }

  const candidate = value as Partial<ReviewSession>

  return (
    isNonEmptyString(candidate.fileHash) &&
    typeof candidate.label === 'string' &&
    typeof candidate.uploadedAt === 'string'
  )
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}
