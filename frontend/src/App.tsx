import { useState } from 'react'
import { ReviewQueue } from './components/ReviewQueue'
import { UploadCsv } from './components/UploadCsv'
import { uploadTransactionsCsv, type ReviewDataResult } from './api/review'

function App() {
  const [reviewData, setReviewData] = useState<ReviewDataResult | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const uploadCsv = async (file: File) => {
    setIsUploading(true)
    setUploadError(null)

    try {
      const data = await uploadTransactionsCsv(file)
      setReviewData(data)
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : 'The upload could not be processed',
      )
    } finally {
      setIsUploading(false)
    }
  }

  if (!reviewData) {
    return (
      <UploadCsv
        error={uploadError}
        isUploading={isUploading}
        onUpload={uploadCsv}
      />
    )
  }

  return (
    <ReviewQueue
      items={reviewData.items}
      onReset={() => setReviewData(null)}
    />
  )
}

export default App
