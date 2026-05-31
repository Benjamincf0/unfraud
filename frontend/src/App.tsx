import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ReviewQueue } from './components/ReviewQueue'
import { UploadCsv } from './components/UploadCsv'
import {
  openReviewSession,
  uploadTransactionsCsv,
  type ReviewSessionData,
} from './api/review'
import {
  clearActiveReviewSession,
  loadActiveReviewSession,
  loadReviewSessions,
  saveActiveReviewSession,
  saveReviewSession,
} from './lib/reviewSessions'

function App() {
  const queryClient = useQueryClient()
  const [session, setSession] = useState<ReviewSessionData | null>(null)
  const [sessions, setSessions] = useState(loadReviewSessions)
  const [activeFileHash, setActiveFileHash] = useState<string | null>(
    loadActiveReviewSession,
  )
  const [isUploadMode, setIsUploadMode] = useState(() => !activeFileHash)
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
      queryClient.removeQueries({ queryKey: ['review-session', data.fileHash] })
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
    queryClient.removeQueries({ queryKey: ['review-session'] })
    setSession(null)
    setActiveFileHash(null)
    setIsUploadMode(true)
  }, [queryClient])

  useEffect(() => {
    if (session?.fileHash === activeFileHash) {
      return
    }

    if (!activeFileHash || !sessionQuery.isError || sessionQuery.isFetching) {
      return
    }

    const errorMessage =
      sessionQuery.error instanceof Error ? sessionQuery.error.message : ''

    if (errorMessage.toLowerCase().includes('file not found')) {
      showUploadScreen()
    }
  }, [
    activeFileHash,
    session?.fileHash,
    sessionQuery.error,
    sessionQuery.isError,
    sessionQuery.isFetching,
    showUploadScreen,
  ])

  if (!activeSession) {
    return (
      <UploadCsv
        error={
          uploadMutation.error instanceof Error
            ? uploadMutation.error.message
            : isUploadMode || session?.fileHash === activeFileHash
            ? null
            : sessionQuery.error instanceof Error
            ? sessionQuery.error.message
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
