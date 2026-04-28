'use client';
import { getDocumentStatusColor } from '@/app/workspace/collections/tools';
import { FormatDate } from '@/components/format-date';
import { buildDocumentAssetUrl, Markdown } from '@/components/markdown';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { buildDocumentObjectUrl } from '@/features/document/client-api';
import type { Document, DocumentPreview } from '@/features/document/types';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import {
  ArrowLeft,
  Download,
  FileText,
  ImageIcon,
  LoaderCircle,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

const PDFDocument = dynamic(() => import('react-pdf').then((r) => r.Document), {
  ssr: false,
});
const PDFPage = dynamic(() => import('react-pdf').then((r) => r.Page), {
  ssr: false,
});

const IMAGE_EXTENSIONS = new Set([
  'png',
  'jpg',
  'jpeg',
  'webp',
  'gif',
  'bmp',
  'tif',
  'tiff',
]);

const MARKDOWN_TEXT_EXTENSIONS = new Set(['md', 'markdown', 'txt']);

const getExtension = (filename?: string | null) =>
  filename?.split('.').pop()?.toLowerCase() ?? '';

const formatFileSize = (size?: number | null) => {
  const kb = Number(size || 0) / 1000;
  if (kb < 1000) return `${kb.toFixed(2)} KB`;
  return `${(kb / 1000).toFixed(2)} MB`;
};

const buildDocumentDownloadUrl = (collectionId: string, documentId: string) =>
  `${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}/api/v2/collections/${collectionId}/documents/${documentId}/download`;

export const DocumentDetail = ({
  document,
  documentPreview,
}: {
  document: Document;
  documentPreview: DocumentPreview;
}) => {
  const { collection } = useCollectionContext();
  const page_documents = useTranslations('page_documents');
  const documentText = page_documents as unknown as (
    key: string,
    values?: Record<string, string>,
  ) => string;
  const [numPages, setNumPages] = useState<number>(0);

  const collectionId = collection.id ?? '';
  const documentId = document.id ?? '';
  const filename = documentPreview.doc_filename || document.name || '';
  const extension = getExtension(filename);
  const originalObjectPath = documentPreview.doc_object_path;
  const originalObjectUrl =
    collectionId && documentId && originalObjectPath
      ? buildDocumentObjectUrl(collectionId, documentId, originalObjectPath)
      : undefined;
  const convertedPdfUrl =
    collectionId && documentId && documentPreview.converted_pdf_object_path
      ? buildDocumentObjectUrl(
          collectionId,
          documentId,
          documentPreview.converted_pdf_object_path,
        )
      : undefined;
  const downloadUrl =
    collectionId && documentId
      ? buildDocumentDownloadUrl(collectionId, documentId)
      : originalObjectUrl;

  const isPdf = extension === 'pdf';
  const isImage = IMAGE_EXTENSIONS.has(extension);
  const isMarkdownText = MARKDOWN_TEXT_EXTENSIONS.has(extension);
  const markdownContent = documentPreview.markdown_content?.trim();
  const hasParsedText = Boolean(markdownContent);
  const visionChunks = documentPreview.vision_chunks || [];
  const hasVisionPreview = visionChunks.length > 0;
  const pdfPreviewUrl = isPdf
    ? originalObjectUrl || convertedPdfUrl
    : undefined;
  const hasOriginalPreview = Boolean(
    pdfPreviewUrl || (isImage && originalObjectUrl),
  );
  const defaultTab =
    hasOriginalPreview || !isMarkdownText
      ? 'original'
      : hasParsedText
        ? 'parsed'
        : 'original';

  useEffect(() => {
    const loadPDF = async () => {
      const { pdfjs } = await import('react-pdf');

      pdfjs.GlobalWorkerOptions.workerSrc = new URL(
        'pdfjs-dist/build/pdf.worker.min.mjs',
        import.meta.url,
      ).toString();
    };
    loadPDF();
  }, []);

  useEffect(() => {
    setNumPages(0);
  }, [pdfPreviewUrl]);

  return (
    <Tabs
      defaultValue={defaultTab}
      className="border-border/70 bg-card gap-0 overflow-hidden rounded-xl border shadow-sm"
    >
      <div className="grid gap-4 border-b p-4 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="flex min-w-0 flex-row items-start gap-3">
          <Button asChild variant="outline" size="icon" className="shrink-0">
            <Link href={`/workspace/collections/${collection.id}/documents`}>
              <ArrowLeft className="size-4" />
            </Link>
          </Button>
          <div className="min-w-0">
            <div className={cn('truncate text-base font-medium')}>
              {filename}
            </div>
            <div className="text-muted-foreground mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="font-mono tabular-nums">
                {formatFileSize(document.size)}
              </span>
              {document.updated ? (
                <>
                  <Separator
                    orientation="vertical"
                    className="data-[orientation=vertical]:h-3"
                  />
                  <FormatDate datetime={new Date(document.updated)} />
                </>
              ) : null}
              {document.status ? (
                <>
                  <Separator
                    orientation="vertical"
                    className="data-[orientation=vertical]:h-3"
                  />
                  <Badge
                    variant="outline"
                    className={cn(
                      'bg-secondary rounded-sm border-transparent font-mono text-[10px] uppercase',
                      getDocumentStatusColor(document.status),
                    )}
                  >
                    {_.capitalize(document.status)}
                  </Badge>
                </>
              ) : null}
            </div>
          </div>
        </div>

        <TabsList className="w-fit justify-start">
          <TabsTrigger value="original">
            {documentText('preview_original')}
          </TabsTrigger>
          {hasParsedText && (
            <TabsTrigger value="parsed">
              {documentText('preview_parsed_text')}
            </TabsTrigger>
          )}
          {hasVisionPreview && (
            <TabsTrigger value="vision">
              {documentText('preview_image_analysis')}
            </TabsTrigger>
          )}
        </TabsList>
      </div>

      <TabsContent value="original" className="bg-background/50 m-0 p-4">
        {pdfPreviewUrl ? (
          <PDFDocument
            file={pdfPreviewUrl}
            onLoadSuccess={({ numPages }: { numPages: number }) => {
              setNumPages(numPages);
            }}
            loading={
              <div className="flex flex-col py-12">
                <LoaderCircle className="size-10 animate-spin self-center opacity-50" />
              </div>
            }
            className="flex flex-col justify-center gap-1"
          >
            {_.times(numPages).map((index) => {
              return (
                <div key={index} className="text-center">
                  <Card className="border-border/70 inline-block overflow-hidden p-0 shadow-sm">
                    <PDFPage pageNumber={index + 1} className="bg-muted" />
                  </Card>
                </div>
              );
            })}
          </PDFDocument>
        ) : isImage && originalObjectUrl ? (
          <div className="flex justify-center">
            <img
              src={originalObjectUrl}
              alt={filename || documentText('preview_original')}
              className="border-border/70 bg-background max-h-[75vh] max-w-full rounded-lg border object-contain shadow-sm"
            />
          </div>
        ) : (
          <Card className="border-border/70 py-0 shadow-sm">
            <CardContent className="text-muted-foreground flex min-h-72 flex-col items-center justify-center gap-4 p-8 text-center text-sm">
              <FileText className="size-10" />
              <div className="space-y-1">
                <div className="text-foreground text-sm font-medium">
                  {documentText('original_preview_unavailable')}
                </div>
                {hasParsedText ? (
                  <div>{documentText('parsed_text_available_hint')}</div>
                ) : null}
              </div>
              {downloadUrl ? (
                <Button asChild variant="outline">
                  <a href={downloadUrl} download={filename || undefined}>
                    <Download className="size-4" />
                    {documentText('download_original')}
                  </a>
                </Button>
              ) : null}
            </CardContent>
          </Card>
        )}
      </TabsContent>

      {hasParsedText && (
        <TabsContent value="parsed" className="bg-background/50 m-0 p-4">
          <Card className="border-border/70 py-0 shadow-sm">
            <CardContent className="space-y-4 p-5">
              <div className="text-muted-foreground text-xs">
                {documentText('parsed_text_notice')}
              </div>
              <Markdown>{markdownContent}</Markdown>
            </CardContent>
          </Card>
        </TabsContent>
      )}

      {hasVisionPreview && (
        <TabsContent value="vision" className="bg-background/50 m-0 p-4">
          <div className="grid gap-4 lg:grid-cols-2">
            {visionChunks.map((chunk, index) => {
              const imageUrl = chunk.asset_id
                ? buildDocumentAssetUrl(
                    `asset://${chunk.asset_id}?collection_id=${collection.id}&document_id=${document.id}`,
                    {
                      collectionId: collection.id ?? '',
                      documentId: document.id ?? '',
                      mode: 'workspace',
                    },
                  )
                : undefined;
              const pageIdx =
                typeof chunk.metadata === 'object' &&
                chunk.metadata &&
                'page_idx' in chunk.metadata
                  ? Number(chunk.metadata.page_idx) + 1
                  : undefined;

              return (
                <Card
                  key={chunk.id || chunk.asset_id || index}
                  className="border-border/70 gap-0 overflow-hidden py-0 shadow-sm"
                >
                  <CardContent className="space-y-4 p-4">
                    {imageUrl ? (
                      <img
                        src={imageUrl}
                        alt={documentText('image_alt', {
                          number: String(index + 1),
                        })}
                        className="border-border/70 bg-background w-full rounded-lg border"
                      />
                    ) : (
                      <div className="text-muted-foreground bg-muted flex min-h-48 items-center justify-center rounded-lg">
                        <ImageIcon className="size-8" />
                      </div>
                    )}
                    <div className="space-y-2">
                      <div className="text-sm font-medium">
                        {pageIdx
                          ? documentText('page_label', {
                              number: String(pageIdx),
                            })
                          : documentText('image_label', {
                              number: String(index + 1),
                            })}
                      </div>
                      <div className="text-muted-foreground text-sm whitespace-pre-wrap">
                        {chunk.text ||
                          documentText('image_summary_unavailable')}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>
      )}
    </Tabs>
  );
};
