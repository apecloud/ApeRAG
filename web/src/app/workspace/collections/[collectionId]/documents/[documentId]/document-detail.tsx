'use client';
import { Collection, Document, DocumentPreview } from '@/api';
import { getDocumentStatusColor } from '@/app/workspace/collections/tools';
import { FormatDate } from '@/components/format-date';
import { Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { ArrowLeft, LoaderCircle } from 'lucide-react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

const PDFDocument = dynamic(() => import('react-pdf').then((r) => r.Document), {
  ssr: false,
});
const PDFPage = dynamic(() => import('react-pdf').then((r) => r.Page), {
  ssr: false,
});

export const DocumentDetail = ({
  collection,
  document,
  documentPreview,
}: {
  document: Document;
  collection: Collection;
  documentPreview: DocumentPreview;
}) => {
  const [numPages, setNumPages] = useState<number>(0);

  const isPdf = useMemo(() => {
    return Boolean(documentPreview.doc_filename?.match(/\.pdf/));
  }, [documentPreview.doc_filename]);

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

  const icon = useMemo(() => {
    const extension =
      document.name?.split('.').pop()?.toLowerCase() ||
      ('unknow' as keyof typeof defaultStyles);
    const iconProps = _.get(defaultStyles, extension);
    return (
      <FileIcon
        color="var(--muted-foreground)"
        extension={extension}
        {...iconProps}
      />
    );
  }, [document.name]);

  return (
    <>
      <Tabs defaultValue="markdown" className="gap-4">
        <div className="flex flex-row items-center justify-between gap-2">
          <div className="flex flex-row items-center gap-4">
            <Button asChild variant="outline" size="icon">
              <Link href={`/workspace/collections/${collection.id}/documents`}>
                <ArrowLeft />
              </Link>
            </Button>
            <div className="flex flex-row items-center gap-2">
              <div className="h-8 w-6">{icon}</div>
              <div>
                <div className={cn('max-w-60 truncate')}>{document.name}</div>
              </div>
            </div>
          </div>

          <div className="flex flex-row gap-6">
            <div className="text-muted-foreground flex flex-row items-center gap-4 text-sm">
              <div>{(Number(document.size || 0) / 1000).toFixed(2)} KB</div>
              <Separator
                orientation="vertical"
                className="data-[orientation=vertical]:h-6"
              />
              {document.updated ? (
                <>
                  <div>
                    <FormatDate datetime={new Date(document.updated)} />
                  </div>
                  <Separator
                    orientation="vertical"
                    className="data-[orientation=vertical]:h-6"
                  />
                </>
              ) : null}
              <div className={getDocumentStatusColor(document.status)}>
                {_.capitalize(document.status)}
              </div>
            </div>
            <TabsList>
              <TabsTrigger value="markdown">Markdown</TabsTrigger>
              {isPdf && <TabsTrigger value="pdf">PDF</TabsTrigger>}
            </TabsList>
          </div>
        </div>

        <TabsContent value="markdown">
          <Card>
            <CardContent>
              <Markdown>{documentPreview.markdown_content}</Markdown>
            </CardContent>
          </Card>
        </TabsContent>

        {isPdf && (
          <TabsContent value="pdf">
            <PDFDocument
              file={`/api/v1/collections/${collection.id}/documents/${document.id}/object?path=${documentPreview.converted_pdf_object_path}`}
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
