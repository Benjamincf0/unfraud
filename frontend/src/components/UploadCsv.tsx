import { useEffect, useRef, useState } from 'react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader } from './ui/card'

type UploadCsvProps = {
  error: string | null
  isModelStatusLoading: boolean
  isUploading: boolean
  mlModelAvailable: boolean
  modelPath?: string
  modelStatusError: string | null
  onUpload: (file: File, useModel: boolean) => void
}

export function UploadCsv({
  error,
  isModelStatusLoading,
  isUploading,
  mlModelAvailable,
  modelPath,
  modelStatusError,
  onUpload,
}: UploadCsvProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [useModel, setUseModel] = useState(false)
  const modelOptionDisabled =
    isUploading || isModelStatusLoading || !mlModelAvailable

  useEffect(() => {
    if (!mlModelAvailable) {
      setUseModel(false)
    }
  }, [mlModelAvailable])

  const submitUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile, useModel)
    }
  }

  return (
    <main className="upload-page">
      <Card className="upload-card">
        <CardHeader>
          <div>
            <h1>Upload Transactions</h1>
          </div>
        </CardHeader>
        <CardContent>
          <button
            className="file-drop"
            onClick={() => inputRef.current?.click()}
            type="button"
          >
            <span className="file-drop-title">
              {selectedFile ? selectedFile.name : 'Choose transactions.csv'}
            </span>
            <span className="file-drop-meta">
              {selectedFile
                ? `${Math.ceil(selectedFile.size / 1024)} KB selected`
                : 'CSV files only'}
            </span>
          </button>

          <input
            accept=".csv,text/csv"
            className="visually-hidden"
            disabled={isUploading}
            onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            ref={inputRef}
            type="file"
          />

          {error ? (
            <div className="upload-error" role="alert">
              {error}
            </div>
          ) : null}

          <label className="upload-option">
            <input
              checked={useModel}
              disabled={modelOptionDisabled}
              onChange={(event) => setUseModel(event.target.checked)}
              type="checkbox"
            />
            <span>
              Use trained ML model
              {isModelStatusLoading ? ' (checking backend model...)' : ''}
              {!isModelStatusLoading && modelStatusError
                ? ' (could not check backend model)'
                : ''}
              {!isModelStatusLoading && !modelStatusError && !mlModelAvailable
                ? ` (not available at ${modelPath ?? 'backend/algo/ops/fraud_model.pkl'})`
                : ''}
            </span>
          </label>

          <div className="upload-actions">
            <Button
              disabled={!selectedFile || isUploading}
              onClick={submitUpload}
            >
              {isUploading ? 'Analyzing' : 'Upload CSV'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  )
}
