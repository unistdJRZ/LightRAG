import { useCallback, useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { ImageIcon, FileTextIcon, RefreshCwIcon } from 'lucide-react'

import { getChunksPaginated, ChunkPreview, PaginationInfo } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useSettingsStore } from '@/stores/settings'
import PaginationControls from '@/components/ui/PaginationControls'
import EmptyCard from '@/components/ui/EmptyCard'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

const DEFAULT_PAGINATION: PaginationInfo = {
  page: 1,
  page_size: 20,
  total_count: 0,
  total_pages: 0,
  has_next: false,
  has_prev: false
}

const IMAGE_PLACEHOLDER_MIME = 'image/png'

function isImageChunk(chunk: ChunkPreview | null): boolean {
  return (chunk?.content_type ?? '').trim().toLowerCase() === 'image'
}

function getImagePayload(chunk: ChunkPreview | null): string | null {
  if (!chunk || !isImageChunk(chunk)) {
    return null
  }

  const payload = (chunk.image_base64 ?? chunk.content ?? '').trim()
  return payload || null
}

function buildImageSrc(content: string): string {
  if (content.startsWith('data:image/')) {
    return content
  }
  return `data:${IMAGE_PLACEHOLDER_MIME};base64,${content}`
}

function truncateMiddle(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value
  }
  const head = Math.ceil(maxLength * 0.65)
  const tail = Math.max(8, maxLength - head - 3)
  return `${value.slice(0, head)}...${value.slice(-tail)}`
}

function getDisplayFileName(filePath: string): string {
  const segments = filePath.split(/[\\/]/)
  return segments[segments.length - 1] || filePath
}

export default function ChunksExplorer() {
  const { t } = useTranslation()
  const workspace = useSettingsStore.use.workspace()
  const [chunks, setChunks] = useState<ChunkPreview[]>([])
  const [pagination, setPagination] = useState<PaginationInfo>(DEFAULT_PAGINATION)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)

  const fetchChunks = useCallback(async (page: number, pageSize: number) => {
    setIsLoading(true)
    try {
      const response = await getChunksPaginated({
        page,
        page_size: pageSize,
        sort_direction: 'desc',
        workspace
      })

      setChunks(response.chunks)
      setPagination(response.pagination)
      setSelectedChunkId((previous) => {
        if (previous && response.chunks.some((chunk) => chunk.chunk_id === previous)) {
          return previous
        }
        return response.chunks[0]?.chunk_id ?? null
      })
    } catch (error) {
      toast.error(
        t('chunksPanel.loadError', {
          defaultValue: 'Failed to load chunks: {{error}}',
          error: errorMessage(error)
        })
      )
      setChunks([])
      setPagination((previous) => ({
        ...previous,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      }))
      setSelectedChunkId(null)
    } finally {
      setIsLoading(false)
    }
  }, [t, workspace])

  useEffect(() => {
    fetchChunks(pagination.page, pagination.page_size)
  }, [fetchChunks, pagination.page, pagination.page_size, workspace])

  const selectedChunk = useMemo(
    () => chunks.find((chunk) => chunk.chunk_id === selectedChunkId) ?? chunks[0] ?? null,
    [chunks, selectedChunkId]
  )

  const selectedChunkImagePayload = useMemo(() => getImagePayload(selectedChunk), [selectedChunk])

  const selectedChunkImageSrc = useMemo(() => {
    if (!selectedChunkImagePayload) {
      return null
    }
    return buildImageSrc(selectedChunkImagePayload)
  }, [selectedChunkImagePayload])

  const selectedChunkMetadata = useMemo(() => {
    if (!selectedChunk) {
      return []
    }

    return [
      {
        label: t('chunksPanel.meta.chunkId', 'Chunk ID'),
        value: selectedChunk.chunk_id,
        mono: true
      },
      {
        label: t('chunksPanel.meta.docId', 'Document ID'),
        value: selectedChunk.doc_id,
        mono: true
      },
      {
        label: t('chunksPanel.meta.file', 'File'),
        value: selectedChunk.file_path
      },
      {
        label: t('chunksPanel.meta.type', 'Type'),
        value: selectedChunk.content_type || 'text'
      },
      {
        label: t('chunksPanel.meta.order', 'Order'),
        value: selectedChunk.chunk_order_index ?? '-'
      },
      {
        label: t('chunksPanel.meta.tokens', 'Tokens'),
        value: selectedChunk.tokens ?? '-'
      },
      {
        label: t('chunksPanel.meta.page', 'Page'),
        value: selectedChunk.page_id ?? '-'
      },
      {
        label: t('chunksPanel.meta.ocrChunkId', 'OCR Chunk ID'),
        value: selectedChunk.ocr_chunk_id ?? '-',
        mono: true
      },
      {
        label: t('chunksPanel.meta.bbox', 'BBox'),
        value: selectedChunk.bbox ? JSON.stringify(selectedChunk.bbox) : '-',
        mono: true
      },
      {
        label: t('chunksPanel.meta.size', 'Size'),
        value: selectedChunk.content.length.toLocaleString()
      }
    ]
  }, [selectedChunk, t])

  const handlePageChange = useCallback((page: number) => {
    setPagination((previous) => ({ ...previous, page }))
  }, [])

  const handlePageSizeChange = useCallback((pageSize: number) => {
    setPagination((previous) => ({ ...previous, page: 1, page_size: pageSize }))
  }, [])

  const handleRefresh = useCallback(() => {
    fetchChunks(pagination.page, pagination.page_size)
  }, [fetchChunks, pagination.page, pagination.page_size])

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      <Card className="border-border/60 bg-background/70">
        <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle>{t('chunksPanel.title', 'Chunks')}</CardTitle>
            <CardDescription>
              {t(
                'chunksPanel.description',
                'Browse stored chunk records and preview their full content, including image chunks.'
              )}
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isLoading}
          >
            <RefreshCwIcon className={cn('mr-2 size-4', isLoading && 'animate-spin')} />
            {t('chunksPanel.refresh', 'Refresh')}
          </Button>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
          <span>{t('chunksPanel.total', { defaultValue: 'Total chunks: {{count}}', count: pagination.total_count })}</span>
          <span>{t('chunksPanel.workspace', { defaultValue: 'Workspace: {{name}}', name: workspace || 'default' })}</span>
        </CardContent>
      </Card>

      {chunks.length === 0 && !isLoading ? (
        <EmptyCard
          title={t('chunksPanel.emptyTitle', 'No chunks')}
          description={t('chunksPanel.emptyDescription', 'No chunk records are available in the current workspace.')}
        />
      ) : (
        <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(360px,0.9fr)]">
          <Card className="min-h-0 min-w-0 overflow-hidden">
            <CardHeader>
              <CardTitle>{t('chunksPanel.listTitle', 'Chunk List')}</CardTitle>
              <CardDescription>
                {t('chunksPanel.listDescription', 'Select a chunk to inspect its stored payload and metadata.')}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
              <div className="min-h-0 flex-1 overflow-hidden rounded-md border">
                <Table className="table-fixed">
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[120px]">{t('chunksPanel.columns.type', 'Type')}</TableHead>
                      <TableHead className="w-[190px]">{t('chunksPanel.columns.chunkId', 'Chunk ID')}</TableHead>
                      <TableHead>{t('chunksPanel.columns.file', 'File')}</TableHead>
                      <TableHead className="w-[70px]">{t('chunksPanel.columns.page', 'Page')}</TableHead>
                      <TableHead className="w-[96px] text-right">{t('chunksPanel.columns.size', 'Size')}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {chunks.map((chunk) => {
                      const selected = chunk.chunk_id === selectedChunk?.chunk_id
                      const image = isImageChunk(chunk)
                      return (
                        <TableRow
                          key={chunk.chunk_id}
                          className={cn('cursor-pointer', selected && 'bg-emerald-500/10')}
                          onClick={() => setSelectedChunkId(chunk.chunk_id)}
                        >
                          <TableCell className="w-[120px]">
                            <span className="inline-flex max-w-full items-center gap-2 text-xs font-medium">
                              {image ? <ImageIcon className="size-4" /> : <FileTextIcon className="size-4" />}
                              <span className="truncate">{chunk.content_type || 'text'}</span>
                            </span>
                          </TableCell>
                          <TableCell className="w-[190px] font-mono text-xs" title={chunk.chunk_id}>
                            <span className="block truncate">{truncateMiddle(chunk.chunk_id, 28)}</span>
                          </TableCell>
                          <TableCell title={chunk.file_path}>
                            <span className="block truncate">{getDisplayFileName(chunk.file_path)}</span>
                          </TableCell>
                          <TableCell className="w-[70px] whitespace-nowrap">{chunk.page_id ?? '-'}</TableCell>
                          <TableCell className="w-[96px] whitespace-nowrap text-right font-mono text-xs">
                            {chunk.content.length.toLocaleString()}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </div>

              <PaginationControls
                currentPage={pagination.page}
                totalPages={Math.max(1, pagination.total_pages)}
                pageSize={pagination.page_size}
                totalCount={pagination.total_count}
                onPageChange={handlePageChange}
                onPageSizeChange={handlePageSizeChange}
                isLoading={isLoading}
              />
            </CardContent>
          </Card>

          <Card className="min-h-0 min-w-0 overflow-hidden">
            <CardHeader>
              <CardTitle>{t('chunksPanel.previewTitle', 'Preview')}</CardTitle>
              <CardDescription>
                {selectedChunk
                  ? t('chunksPanel.previewDescription', {
                    defaultValue: 'Inspect chunk {{chunkId}} from {{file}}.',
                    chunkId: selectedChunk.chunk_id,
                    file: getDisplayFileName(selectedChunk.file_path)
                  })
                  : t('chunksPanel.previewEmpty', 'Select a chunk to preview its content.')}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex min-h-0 min-w-0 flex-1 flex-col gap-4">
              {selectedChunk ? (
                <>
                  <div className="grid gap-3 rounded-md border bg-muted/20 p-3 text-sm sm:grid-cols-2">
                    {selectedChunkMetadata.map((item) => (
                      <div key={item.label} className="min-w-0">
                        <div className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
                          {item.label}
                        </div>
                        <div className={cn('break-words', item.mono && 'font-mono text-xs')}>
                          {String(item.value)}
                        </div>
                      </div>
                    ))}
                  </div>

                  {isImageChunk(selectedChunk) && selectedChunkImageSrc && (
                    <div className="overflow-hidden rounded-md border bg-black/5 p-3">
                      <img
                        src={selectedChunkImageSrc}
                        alt={selectedChunk.chunk_id}
                        className="mx-auto max-h-[320px] rounded object-contain"
                      />
                    </div>
                  )}

                  {selectedChunk.image_text && (
                    <div className="rounded-md border bg-muted/10 p-3">
                      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {t('chunksPanel.previewFields.imageText', 'Image Text')}
                      </div>
                      <pre className="whitespace-pre-wrap break-words text-xs leading-5">
                        {selectedChunk.image_text}
                      </pre>
                    </div>
                  )}

                  {!isImageChunk(selectedChunk) && (
                    <div className="min-h-0 flex-1 overflow-auto rounded-md border bg-muted/10 p-3">
                      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {t('chunksPanel.previewFields.content', 'Content')}
                      </div>
                      <pre className="whitespace-pre-wrap break-words text-xs leading-5">
                        {selectedChunk.content}
                      </pre>
                    </div>
                  )}
                </>
              ) : (
                <EmptyCard
                  title={t('chunksPanel.previewEmptyTitle', 'Nothing selected')}
                  description={t('chunksPanel.previewEmptyDescription', 'Choose a chunk from the list to preview its content.')}
                />
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
