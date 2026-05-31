import { useCallback, useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ReviewQueue } from './components/ReviewQueue'
import { UploadCsv } from './components/UploadCsv'
import {
  fetchReviewDataByHash,
  fetchScoringStatus,
  uploadTransactionsCsv,
  type ReviewDataResult,
} from './api/review'
import {
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
  const scoringStatusQuery = useQuery({
    enabled: isUploadMode,
    queryFn: fetchScoringStatus,
    queryKey: ['scoring-status'],
  })
  const activeSession = sessions.find((session) => session.fileHash === activeFileHash)
  const activeUseModel = activeSession?.useModel ?? reviewData?.useModel ?? false
  const cachedResultQuery = useQuery({
    enabled:
      !isUploadMode &&
      Boolean(activeFileHash) &&
      reviewData?.fileHash !== activeFileHash,
    queryFn: () => fetchReviewDataByHash(activeFileHash ?? '', activeUseModel),
    queryKey: ['review-data', activeFileHash, activeUseModel],
  })
  const uploadMutation = useMutation({
    mutationFn: ({ file, useModel }: { file: File; useModel: boolean }) =>
      uploadTransactionsCsv(file, useModel),
    onSuccess: (data, variables) => {
      setSessions(
        saveReviewSession({
          fileHash: data.fileHash,
          label: variables.file.name,
          uploadedAt: new Date().toISOString(),
          useModel: data.useModel,
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

  const uploadCsv = (file: File, useModel: boolean) => {
    uploadMutation.mutate({ file, useModel })
  }

  const selectSession = useCallback((fileHash: string) => {
    saveActiveReviewSession(fileHash)
    setActiveFileHash(fileHash)
    setReviewData(null)
    setIsUploadMode(false)
  }, [])

  const showUploadScreen = useCallback(() => {
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
        mlModelAvailable={scoringStatusQuery.data?.ml_model_available ?? false}
        onUpload={uploadCsv}
      />
    )
  }

  return (
    <ReviewQueue
      activeFileHash={activeReviewData.fileHash}
      fileHash={activeReviewData.fileHash}
      items={activeReviewData.items}
      onReset={showUploadScreen}
      onSelectSession={selectSession}
      sessions={sessions}
      useModel={activeReviewData.useModel}
    />
  )
}

export default App
