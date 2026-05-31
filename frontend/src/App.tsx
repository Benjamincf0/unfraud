import { useCallback, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ReviewQueue } from './components/ReviewQueue'
import { UploadCsv } from './components/UploadCsv'
import {
  openReviewSession,
  uploadTransactionsCsv,
  type ReviewSessionData,
} from './api/review'
import {
  clearActiveReviewSession,
  loadReviewSessions,
  saveActiveReviewSession,
  saveReviewSession,
} from './lib/reviewSessions'

function App() {
  const [session, setSession] = useState<ReviewSessionData | null>(null)
  const [sessions, setSessions] = useState(loadReviewSessions)
  const [activeFileHash, setActiveFileHash] = useState<string | null>(null)
  const [isUploadMode, setIsUploadMode] = useState(true)
  const sessionQuery = useQuery({
    enabled:
      !isUploadMode &&
      Boolean(activeFileHash) &&
      session?.fileHash !== activeFileHash,
    queryFn: () => openReviewSession(activeFileHash ?? ''),
    queryKey: ['review-session', activeFileHash],
  })
  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadTransactionsCsv(file),
    onSuccess: (data, file) => {
      setSessions(
        saveReviewSession({
          fileHash: data.fileHash,
          label: file.name,
          uploadedAt: new Date().toISOString(),
        }),
      )
      setActiveFileHash(data.fileHash)
      setSession(data)
      setIsUploadMode(false)
    },
  })
  const activeSession =
    session?.fileHash === activeFileHash
      ? session
      : sessionQuery.data ?? null

  const uploadCsv = (file: File) => {
    uploadMutation.mutate(file)
  }

  const selectSession = useCallback((fileHash: string) => {
    saveActiveReviewSession(fileHash)
    setActiveFileHash(fileHash)
    setSession(null)
    setIsUploadMode(false)
  }, [])

  const showUploadScreen = useCallback(() => {
    clearActiveReviewSession()
    setSession(null)
    setActiveFileHash(null)
    setIsUploadMode(true)
  }, [])

  if (!activeSession) {
    return (
      <UploadCsv
        error={
          sessionQuery.error instanceof Error
            ? sessionQuery.error.message
            : uploadMutation.error instanceof Error
            ? uploadMutation.error.message
            : null
        }
        isUploading={uploadMutation.isPending || sessionQuery.isFetching}
        onUpload={uploadCsv}
      />
    )
  }

  return (
    <ReviewQueue
      activeFileHash={activeSession.fileHash}
      onReset={showUploadScreen}
      onSelectSession={selectSession}
      session={activeSession}
      sessions={sessions}
    />
  )
}

export default App
