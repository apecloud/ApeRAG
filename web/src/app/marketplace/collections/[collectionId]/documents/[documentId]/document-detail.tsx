'use client';
import type { DocumentPreview } from '@/features/document/types';
import { buildDocumentAssetUrl, Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { ArrowLeft, LoaderCircle } from 'lucide-react';
import dynamic from 'next/dynamic';
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
        <div className="flex flex-row items-center justify-between gap-2">
          <div className="flex flex-row items-center gap-4">
            <Button asChild variant="ghost" size="icon">
              <Link href={`/marketplace/collections/${collectionId}/documents`}>
                <ArrowLeft />
              </Link>
            </Button>
            <div className={cn('max-w-80 truncate')}>
              {documentPreview.doc_filename}
            </div>
          </div>

          <div className="flex flex-row gap-6">
            <div className="text-muted-foreground flex flex-row items-center gap-4 text-sm"></div>
            <TabsList>
              {hasPdfPreview && <TabsTrigger value="pdf">PDF</TabsTrigger>}
              {hasVisionPreview && (
                <TabsTrigger value="vision">Images</TabsTrigger>
              )}
              <TabsTrigger value="markdown">Markdown</TabsTrigger>
            </TabsList>
          </div>
        </div>

        <TabsContent value="markdown">
          <Card>
            <CardContent>
              {documentPreview.markdown_content?.trim() ? (
                <Markdown>{documentPreview.markdown_content}</Markdown>
              ) : (
                <div className="text-muted-foreground py-6 text-sm">
                  Markdown preview is unavailable for this document.
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
                  <Card key={chunk.id || chunk.asset_id || index}>
                    <CardContent className="space-y-4">
                      {imageUrl ? (
                        <img
                          src={imageUrl}
                          alt={`Document image ${index + 1}`}
                          className="w-full rounded-md border"
                        />
                      ) : null}
                      <div className="space-y-2">
                        <div className="text-sm font-medium">
                          {pageIdx ? `Page ${pageIdx}` : `Image ${index + 1}`}
                        </div>
                        <div className="text-muted-foreground whitespace-pre-wrap text-sm">
                          {chunk.text || 'No extracted image summary available.'}
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
                  <LoaderCircle className="size-10 animate-spin self-center opacity-50" />
                </div>
              }
              className="flex flex-col justify-center gap-1"
            >
              {_.times(numPages).map((index) => {
                return (
                  <div key={index} className="text-center">
                    <Card className="inline-block overflow-hidden p-0">
                      <PDFPage pageNumber={index + 1} className="bg-accent" />
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
