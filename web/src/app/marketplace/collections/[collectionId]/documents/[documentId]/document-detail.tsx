'use client';
import { Markdown } from '@/components/markdown';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import type { DocumentPreview } from '@/features/document/types';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import {
  ArrowLeft,
  Download,
  FileText,
  LoaderCircle,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

// See workspace mirror (../../../../workspace/.../document-detail.tsx)
// for the full PDF.js options rationale — same cMapUrl + standard
// font URL config so non-Latin glyphs render in the marketplace
// preview too.
const PDF_OPTIONS = {
  cMapUrl: `${BASE_PATH}/cmaps/`,
  cMapPacked: true,
  standardFontDataUrl: `${BASE_PATH}/standard_fonts/`,
} as const;

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

// See workspace mirror for the rationale — .txt / .md / .markdown
// uploads have their original content surfaced via
// ``DocumentPreview.markdown_content`` and should still display in
// the preview pane after PR #1815 hid the parsed-text tab (which was
// only meant to suppress poor PDF parses).
const TEXT_PREVIEW_EXTENSIONS = new Set(['txt', 'md', 'markdown']);

const getExtension = (filename?: string | null) =>
  filename?.split('.').pop()?.toLowerCase() ?? '';

const buildMarketplaceDocumentObjectUrl = (
  collectionId: string,
  documentId: string,
  objectPath: string,
) => {
  const query = new URLSearchParams({ path: objectPath });
  return `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/marketplace/collections/${collectionId}/documents/${documentId}/object?${query.toString()}`;
};

export const DocumentDetail = ({
  documentPreview,
}: {
  documentPreview: DocumentPreview;
}) => {
  const [numPages, setNumPages] = useState<number>(0);
  const { documentId, collectionId } = useParams();
  const page_marketplace = useTranslations('page_marketplace');
  const marketplaceText = page_marketplace as unknown as (
    key: string,
    values?: Record<string, string>,
  ) => string;

  const collectionIdValue =
    typeof collectionId === 'string' ? collectionId : collectionId?.[0] || '';
  const documentIdValue =
    typeof documentId === 'string' ? documentId : documentId?.[0] || '';
  const filename = documentPreview.doc_filename || '';
  const extension = getExtension(filename);
  const originalObjectPath = documentPreview.doc_object_path;
  const originalObjectUrl =
    collectionIdValue && documentIdValue && originalObjectPath
      ? buildMarketplaceDocumentObjectUrl(
          collectionIdValue,
          documentIdValue,
          originalObjectPath,
        )
      : undefined;
  const convertedPdfUrl =
    collectionIdValue &&
    documentIdValue &&
    documentPreview.converted_pdf_object_path
      ? buildMarketplaceDocumentObjectUrl(
          collectionIdValue,
          documentIdValue,
          documentPreview.converted_pdf_object_path,
        )
      : undefined;
  const isPdf = extension === 'pdf';
  const isImage = IMAGE_EXTENSIONS.has(extension);
  const isTextPreview = TEXT_PREVIEW_EXTENSIONS.has(extension);
  const markdownContent = documentPreview.markdown_content?.trim();
  const pdfPreviewUrl = isPdf
    ? originalObjectUrl || convertedPdfUrl
    : undefined;

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
    <div className="flex flex-col gap-4">
      <div className="border-border/70 bg-card grid gap-4 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="flex min-w-0 items-center gap-3">
          <Button asChild variant="outline" size="icon">
            <Link
              href={`/marketplace/collections/${collectionIdValue}/documents`}
            >
              <ArrowLeft className="size-4" />
            </Link>
          </Button>
          <div className="bg-accent-soft text-accent-ink flex size-10 shrink-0 items-center justify-center rounded-lg">
            <FileText className="size-5" />
          </div>
          <div className="min-w-0">
            <div className="text-muted-foreground font-mono text-[11px] tracking-[0.12em] uppercase">
              {marketplaceText('document_preview_label')}
            </div>
            <div className={cn('mt-1 truncate text-base font-medium')}>
              {filename}
            </div>
          </div>
        </div>
      </div>

      {/* Parsed-text tab removed per earayu2 directive msg=153f4b85 —
          parsed markdown stays available for retrieval / agent /
          graph backends; only the user-facing preview tab is hidden. */}
      <div>
        {pdfPreviewUrl ? (
          <PDFDocument
            file={pdfPreviewUrl}
            options={PDF_OPTIONS}
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
                  <Card className="border-border/70 inline-block overflow-hidden rounded-xl p-0">
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
              alt={filename || marketplaceText('preview_original')}
              className="border-border/70 bg-background max-h-[75vh] max-w-full rounded-lg border object-contain shadow-sm"
            />
          </div>
        ) : isTextPreview && markdownContent ? (
          <Card className="border-border/70 rounded-xl">
            <CardContent className="p-5 md:p-6">
              <Markdown>{markdownContent}</Markdown>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-border/70 rounded-xl">
            <CardContent className="text-muted-foreground flex min-h-72 flex-col items-center justify-center gap-4 p-8 text-center text-sm">
              <FileText className="size-10" />
              <div className="space-y-1">
                <div className="text-foreground text-sm font-medium">
                  {marketplaceText('original_preview_unavailable')}
                </div>
              </div>
              {originalObjectUrl ? (
                <Button asChild variant="outline">
                  <a href={originalObjectUrl} download={filename || undefined}>
                    <Download className="size-4" />
                    {marketplaceText('download_original')}
                  </a>
                </Button>
              ) : null}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};
