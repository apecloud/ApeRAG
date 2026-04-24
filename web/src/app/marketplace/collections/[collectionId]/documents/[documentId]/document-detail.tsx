'use client';
import type { DocumentPreview } from '@/features/document/types';
import { buildDocumentAssetUrl, Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { ArrowLeft, FileText, ImageIcon, LoaderCircle } from 'lucide-react';
import dynamic from 'next/dynamic';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

const PDFDocument = dynamic(() => import('react-pdf').then((r) => r.Document), {
  ssr: false,
});
const PDFPage = dynamic(() => import('react-pdf').then((r) => r.Page), {
  ssr: false,
});

export const DocumentDetail = ({
  documentPreview,
}: {
  documentPreview: DocumentPreview;
}) => {
  const [numPages, setNumPages] = useState<number>(0);
  const { documentId, collectionId } = useParams();
  const page_marketplace = useTranslations('page_marketplace');

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
      <Tabs defaultValue={defaultTab} className="gap-4">
        <div className="border-border/70 bg-card grid gap-4 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
          <div className="flex min-w-0 items-center gap-3">
            <Button asChild variant="outline" size="icon">
              <Link href={`/marketplace/collections/${collectionId}/documents`}>
                <ArrowLeft className="size-4" />
              </Link>
            </Button>
            <div className="bg-accent-soft text-accent-ink flex size-10 shrink-0 items-center justify-center rounded-lg">
              <FileText className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
                {page_marketplace('document_preview_label')}
              </div>
              <div className={cn('mt-1 truncate text-base font-medium')}>
                {documentPreview.doc_filename}
              </div>
            </div>
          </div>

          <div className="flex justify-start lg:justify-end">
            <TabsList className="bg-muted rounded-xl">
              {hasPdfPreview && (
                <TabsTrigger value="pdf">
                  {page_marketplace('preview_pdf')}
                </TabsTrigger>
              )}
              {hasVisionPreview && (
                <TabsTrigger value="vision">
                  {page_marketplace('preview_images')}
                </TabsTrigger>
              )}
              <TabsTrigger value="markdown">
                {page_marketplace('preview_markdown')}
              </TabsTrigger>
            </TabsList>
          </div>
        </div>

        <TabsContent value="markdown">
          <Card className="rounded-xl border-border/70">
            <CardContent className="p-5 md:p-6">
              {documentPreview.markdown_content?.trim() ? (
                <Markdown>{documentPreview.markdown_content}</Markdown>
              ) : (
                <div className="text-muted-foreground py-6 text-sm">
                  {page_marketplace('markdown_unavailable')}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {hasVisionPreview && (
          <TabsContent value="vision">
            <div className="grid gap-4 lg:grid-cols-2">
              {visionChunks.map((chunk, index) => {
                const collectionIdValue =
                  typeof collectionId === 'string' ? collectionId : collectionId?.[0];
                const documentIdValue =
                  typeof documentId === 'string' ? documentId : documentId?.[0];
                const imageUrl =
                  chunk.asset_id && collectionIdValue && documentIdValue
                    ? buildDocumentAssetUrl(
                        `asset://${chunk.asset_id}?collection_id=${collectionIdValue}&document_id=${documentIdValue}`,
                        {
                          collectionId: collectionIdValue,
                          documentId: documentIdValue,
                          mode: 'marketplace',
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
                    className="rounded-xl border-border/70"
                  >
                    <CardContent className="space-y-4 p-4">
                      {imageUrl ? (
                        <img
                          src={imageUrl}
                          alt={page_marketplace('image_alt', {
                            number: String(index + 1),
                          })}
                          className="w-full rounded-lg border"
                        />
                      ) : null}
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm font-medium">
                          <ImageIcon className="text-muted-foreground size-4" />
                          {pageIdx
                            ? page_marketplace('page_label', {
                                number: String(pageIdx),
                              })
                            : page_marketplace('image_label', {
                                number: String(index + 1),
                              })}
                        </div>
                        <div className="text-muted-foreground whitespace-pre-wrap text-sm">
                          {chunk.text ||
                            page_marketplace('image_summary_unavailable')}
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
          <TabsContent value="pdf">
            <PDFDocument
              file={`${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v1/marketplace/collections/${collectionId}/documents/${documentId}/object?path=${documentPreview.converted_pdf_object_path}`}
              onLoadSuccess={({ numPages }: { numPages: number }) => {
                setNumPages(numPages);
              }}
              loading={
                <div className="flex flex-col py-8">
                  <LoaderCircle className="text-muted-foreground size-10 animate-spin self-center opacity-50" />
                </div>
              }
              className="flex flex-col justify-center gap-1"
            >
              {_.times(numPages).map((index) => {
                return (
                  <div key={index} className="text-center">
                    <Card className="inline-block overflow-hidden rounded-xl border-border/70 p-0">
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
