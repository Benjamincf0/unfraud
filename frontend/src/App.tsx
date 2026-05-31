import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ReviewQueue } from './components/ReviewQueue'
import { UploadCsv } from './components/UploadCsv'
import {
  fetchReviewDataByHash,
  uploadTransactionsCsv,
  type ReviewDataResult,
} from './api/review'
import {
  clearActiveReviewSession,
  loadActiveReviewSession,
  loadReviewSessions,
  saveActiveReviewSession,
  saveReviewSession,
} from './lib/reviewSessions'

function App() {
  const [reviewData, setReviewData] = useState<ReviewDataResult | null>(null)
  const [sessions, setSessions] = useState(loadReviewSessions)
  const [activeFileHash, setActiveFileHash] = useState(
    loadActiveReviewSession,
  )
  const [isUploadMode, setIsUploadMode] = useState(() => !activeFileHash)
  const cachedResultQuery = useQuery({
    enabled:
      !isUploadMode &&
      Boolean(activeFileHash) &&
      reviewData?.fileHash !== activeFileHash,
    queryFn: () => fetchReviewDataByHash(activeFileHash ?? ''),
    queryKey: ['review-data', activeFileHash],
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
      setReviewData(data)
      setIsUploadMode(false)
    },
  })
  const activeReviewData =
    reviewData?.fileHash === activeFileHash
      ? reviewData
      : cachedResultQuery.data ?? null

  const uploadCsv = (file: File) => {
    uploadMutation.mutate(file)
  }

  const selectSession = useCallback((fileHash: string) => {
    saveActiveReviewSession(fileHash)
    setActiveFileHash(fileHash)
    setReviewData(null)
    setIsUploadMode(false)
  }, [])

  const showUploadScreen = useCallback(() => {
    clearActiveReviewSession()
    setReviewData(null)
    setActiveFileHash(null)
    setIsUploadMode(true)
  }, [])

  useEffect(() => {
    if (!isUploadMode && !activeFileHash && sessions.length > 0) {
      selectSession(sessions[0].fileHash)
    }
  }, [activeFileHash, isUploadMode, selectSession, sessions])

  if (!activeReviewData) {
    return (
      <UploadCsv
        error={
          cachedResultQuery.error instanceof Error
            ? cachedResultQuery.error.message
            : uploadMutation.error instanceof Error
            ? uploadMutation.error.message
            : null
        }
        isUploading={uploadMutation.isPending || cachedResultQuery.isFetching}
        onUpload={uploadCsv}
      />
    )
  }

  return (
    <ReviewQueue
      activeFileHash={activeReviewData.fileHash}
      onReset={showUploadScreen}
      onSelectSession={selectSession}
      reviewData={activeReviewData}
      sessions={sessions}
    />
  )
}

export default App
