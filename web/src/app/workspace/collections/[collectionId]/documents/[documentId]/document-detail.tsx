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
import { ArrowLeft, FileText, ImageIcon, LoaderCircle } from 'lucide-react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

const PDFDocument = dynamic(() => import('react-pdf').then((r) => r.Document), {
  ssr: false,
});
const PDFPage = dynamic(() => import('react-pdf').then((r) => r.Page), {
  ssr: false,
});

const formatFileSize = (size?: number | null) => {
  const kb = Number(size || 0) / 1000;
  if (kb < 1000) return `${kb.toFixed(2)} KB`;
  return `${(kb / 1000).toFixed(2)} MB`;
};

export const DocumentDetail = ({
  document,
  documentPreview,
}: {
  document: Document;

  documentPreview: DocumentPreview;
}) => {
  const { collection } = useCollectionContext();
  const [numPages, setNumPages] = useState<number>(0);

  const isPdf = useMemo(() => {
    return Boolean(documentPreview.doc_filename?.match(/\.pdf/));
  }, [documentPreview.doc_filename]);

  const hasPdfPreview = Boolean(
    isPdf && documentPreview.converted_pdf_object_path,
  );
  const visionChunks = documentPreview.vision_chunks || [];
  const hasVisionPreview = visionChunks.length > 0;
  const defaultTab = hasPdfPreview ? 'pdf' : 'markdown';

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

  return (
    <>
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
                {documentPreview.doc_filename}
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
            {hasPdfPreview && <TabsTrigger value="pdf">PDF</TabsTrigger>}
            {hasVisionPreview && (
              <TabsTrigger value="vision">Images</TabsTrigger>
            )}
            <TabsTrigger value="markdown">Markdown</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="markdown" className="bg-background/50 m-0 p-4">
          <Card className="border-border/70 py-0 shadow-sm">
            <CardContent className="p-5">
              {documentPreview.markdown_content?.trim() ? (
                <Markdown>{documentPreview.markdown_content}</Markdown>
              ) : (
                <div className="text-muted-foreground flex min-h-56 flex-col items-center justify-center gap-3 py-6 text-center text-sm">
                  <FileText className="size-8" />
                  Markdown preview is unavailable for this document.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

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
                          alt={`Document image ${index + 1}`}
                          className="border-border/70 bg-background w-full rounded-lg border"
                        />
                      ) : (
                        <div className="text-muted-foreground bg-muted flex min-h-48 items-center justify-center rounded-lg">
                          <ImageIcon className="size-8" />
                        </div>
                      )}
                      <div className="space-y-2">
                        <div className="text-sm font-medium">
                          {pageIdx ? `Page ${pageIdx}` : `Image ${index + 1}`}
                        </div>
                        <div className="text-muted-foreground text-sm whitespace-pre-wrap">
                          {chunk.text ||
                            'No extracted image summary available.'}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>
        )}

        {hasPdfPreview && (
          <TabsContent value="pdf" className="bg-background/50 m-0 p-4">
            <PDFDocument
              file={buildDocumentObjectUrl(
                collection.id ?? '',
                document.id ?? '',
                documentPreview.converted_pdf_object_path ?? '',
              )}
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
          </TabsContent>
        )}
      </Tabs>
    </>
  );
};
