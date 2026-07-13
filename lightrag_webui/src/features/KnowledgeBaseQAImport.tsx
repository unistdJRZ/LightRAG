import { type ChangeEvent, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { AlertCircleIcon, CheckCircle2Icon, FileSpreadsheetIcon, UploadIcon, XIcon } from 'lucide-react'

import { importKnowledgeBaseQA, KnowledgeBaseQAImportResponse } from '@/api/lightrag'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Progress from '@/components/ui/Progress'
import Badge from '@/components/ui/Badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { errorMessage } from '@/lib/utils'

const columns = [
  {
    name: 'kbqa_id',
    required: false,
    description: 'Stable QA id. Leave blank to let the backend generate one.'
  },
  {
    name: 'question',
    required: true,
    description: 'Preset question. This text is embedded for retrieval.'
  },
  {
    name: 'answer',
    required: true,
    description: 'Preset answer returned by QA retrieval.'
  },
  {
    name: 'metadata',
    required: false,
    description: 'JSON object string, for example {"category":"statistics"}.'
  }
]

export default function KnowledgeBaseQAImport() {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isImporting, setIsImporting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<KnowledgeBaseQAImportResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const selectedFileSize = useMemo(() => {
    if (!selectedFile) {
      return ''
    }
    if (selectedFile.size < 1024 * 1024) {
      return `${Math.max(1, Math.round(selectedFile.size / 1024))} KB`
    }
    return `${(selectedFile.size / 1024 / 1024).toFixed(1)} MB`
  }, [selectedFile])

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null
    setResult(null)
    setError(null)
    setProgress(0)

    if (!file) {
      setSelectedFile(null)
      return
    }
    if (!file.name.toLowerCase().endsWith('.xlsx')) {
      setSelectedFile(null)
      event.target.value = ''
      setError(t('knowledgeBaseQAImport.invalidFile', 'Only .xlsx files are supported.'))
      return
    }
    setSelectedFile(file)
  }

  const clearFile = () => {
    setSelectedFile(null)
    setProgress(0)
    setResult(null)
    setError(null)
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  const handleImport = async () => {
    if (!selectedFile || isImporting) {
      return
    }

    setIsImporting(true)
    setError(null)
    setResult(null)
    setProgress(0)
    try {
      const response = await importKnowledgeBaseQA(selectedFile, setProgress)
      setResult(response)
      toast.success(
        t('knowledgeBaseQAImport.successToast', 'Knowledge-base QA imported successfully.')
      )
    } catch (importError) {
      const message = errorMessage(importError)
      setError(message)
      toast.error(t('knowledgeBaseQAImport.errorToast', 'Knowledge-base QA import failed.'))
    } finally {
      setIsImporting(false)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-4">
      <section className="flex flex-col gap-2 border-b pb-4">
        <div className="flex items-center gap-2">
          <FileSpreadsheetIcon className="size-5 text-emerald-500" aria-hidden="true" />
          <h1 className="text-xl font-semibold">
            {t('knowledgeBaseQAImport.title', 'Knowledge-base QA Import')}
          </h1>
        </div>
        <p className="max-w-3xl text-sm text-muted-foreground">
          {t(
            'knowledgeBaseQAImport.description',
            'Import workspace-level preset QA pairs from an Excel workbook. Questions are stored and indexed into the knowledge-base QA vector store.'
          )}
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-3">
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t('knowledgeBaseQAImport.columnName', 'Column')}</TableHead>
                  <TableHead>{t('knowledgeBaseQAImport.required', 'Required')}</TableHead>
                  <TableHead>{t('knowledgeBaseQAImport.notes', 'Notes')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {columns.map((column) => (
                  <TableRow key={column.name}>
                    <TableCell className="font-mono text-xs">{column.name}</TableCell>
                    <TableCell>
                      <Badge variant={column.required ? 'default' : 'secondary'}>
                        {column.required
                          ? t('knowledgeBaseQAImport.yes', 'Yes')
                          : t('knowledgeBaseQAImport.no', 'No')}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {column.description}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <Alert>
            <AlertCircleIcon className="size-4" aria-hidden="true" />
            <AlertTitle>{t('knowledgeBaseQAImport.sheetFormat', 'Excel format')}</AlertTitle>
            <AlertDescription>
              {t(
                'knowledgeBaseQAImport.sheetFormatDescription',
                'Use .xlsx. The first row must be the header. A sheet named knowledge_base_qa is preferred; otherwise the active sheet is used. Chinese headers are also accepted: 问题编号, 预设问题, 预设答案, 元数据.'
              )}
            </AlertDescription>
          </Alert>
        </div>

        <div className="flex flex-col gap-3 rounded-md border p-4">
          <label className="text-sm font-medium" htmlFor="kbqa-import-file">
            {t('knowledgeBaseQAImport.file', 'Excel file')}
          </label>
          <Input
            ref={inputRef}
            id="kbqa-import-file"
            type="file"
            accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            onChange={handleFileChange}
            disabled={isImporting}
          />

          {selectedFile && (
            <div className="flex items-center justify-between rounded-md bg-muted px-3 py-2 text-sm">
              <div className="min-w-0">
                <div className="truncate font-medium">{selectedFile.name}</div>
                <div className="text-xs text-muted-foreground">{selectedFileSize}</div>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                tooltip={t('knowledgeBaseQAImport.removeFile', 'Remove file')}
                onClick={clearFile}
                disabled={isImporting}
              >
                <XIcon className="size-4" aria-hidden="true" />
              </Button>
            </div>
          )}

          {isImporting && (
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <div className="text-xs text-muted-foreground">{progress}%</div>
            </div>
          )}

          <Button
            type="button"
            onClick={handleImport}
            disabled={!selectedFile || isImporting}
            className="w-full"
          >
            <UploadIcon className="size-4" aria-hidden="true" />
            {isImporting
              ? t('knowledgeBaseQAImport.importing', 'Importing...')
              : t('knowledgeBaseQAImport.import', 'Import')}
          </Button>

          {result && (
            <Alert className="border-emerald-500/50">
              <CheckCircle2Icon className="size-4 text-emerald-500" aria-hidden="true" />
              <AlertTitle>{t('knowledgeBaseQAImport.imported', 'Imported')}</AlertTitle>
              <AlertDescription>
                {t(
                  'knowledgeBaseQAImport.importedDescription',
                  'Imported {{count}} QA rows into workspace {{workspace}}.',
                  {
                    count: result.imported_count,
                    workspace: result.workspace
                  }
                )}
              </AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircleIcon className="size-4" aria-hidden="true" />
              <AlertTitle>{t('knowledgeBaseQAImport.failed', 'Import failed')}</AlertTitle>
              <AlertDescription className="break-words">{error}</AlertDescription>
            </Alert>
          )}
        </div>
      </section>
    </div>
  )
}
