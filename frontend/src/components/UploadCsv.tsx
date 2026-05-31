import { useRef, useState } from 'react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader } from './ui/card'

type UploadCsvProps = {
  error: string | null
  isUploading: boolean
  mlModelAvailable: boolean
  onUpload: (file: File, useModel: boolean) => void
}

export function UploadCsv({
  error,
  isUploading,
  mlModelAvailable,
  onUpload,
}: UploadCsvProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [useModel, setUseModel] = useState(false)

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
            <p>Send a CSV to the detector, then review the flagged transactions.</p>
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
              disabled={isUploading || !mlModelAvailable}
              onChange={(event) => setUseModel(event.target.checked)}
              type="checkbox"
            />
            <span>
              Use trained ML model
              {!mlModelAvailable ? ' (not available — train the backend model first)' : ''}
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
