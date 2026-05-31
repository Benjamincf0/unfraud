import { useRef, useState } from 'react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader } from './ui/card'

type UploadCsvProps = {
  error: string | null
  isUploading: boolean
  onUpload: (file: File) => void
}

export function UploadCsv({ error, isUploading, onUpload }: UploadCsvProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const submitUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile)
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
