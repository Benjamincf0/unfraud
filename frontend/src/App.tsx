import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ReviewQueue } from './components/ReviewQueue'
import { UploadCsv } from './components/UploadCsv'
import { uploadTransactionsCsv, type ReviewDataResult } from './api/review'

function App() {
  const [reviewData, setReviewData] = useState<ReviewDataResult | null>(null)
  const uploadMutation = useMutation({
    mutationFn: uploadTransactionsCsv,
    onSuccess: setReviewData,
  })

  const uploadCsv = (file: File) => {
    uploadMutation.mutate(file)
  }

  if (!reviewData) {
    return (
      <UploadCsv
        error={
          uploadMutation.error instanceof Error
            ? uploadMutation.error.message
            : null
        }
        isUploading={uploadMutation.isPending}
        onUpload={uploadCsv}
      />
    )
  }

  return (
    <ReviewQueue
      fileHash={reviewData.fileHash}
      items={reviewData.items}
      onReset={() => setReviewData(null)}
    />
  )
}

export default App
