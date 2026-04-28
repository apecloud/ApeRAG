'use client';
import { getDocumentStatusColor } from '@/app/workspace/collections/tools';
import { FormatDate } from '@/components/format-date';
import { Markdown } from '@/components/markdown';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { buildDocumentObjectUrl } from '@/features/document/client-api';
import type { Document, DocumentPreview } from '@/features/document/types';
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
import { useEffect, useState } from 'react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Plain-text / markdown extensions whose ``markdown_content`` field on
// the BE-side ``DocumentPreview`` IS the original file content (the
// indexing pipeline copies the upload through verbatim — there's no
// "parse" step to hide). Keep this surface so .txt / .md / .markdown
// uploads still display in the original-preview pane after PR #1815
// removed the parsed-text tab (parsed-text was only suppressed for
// PDFs, where the parsed render quality was poor).
const TEXT_PREVIEW_EXTENSIONS = new Set(['txt', 'md', 'markdown']);

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

// PDF.js asset URLs for non-Latin glyph rendering. ``cMapUrl`` is the
// directory PDF.js fetches per-encoding character maps from (Adobe-GB1
// / Adobe-CNS1 / Adobe-Japan1 / etc.) at runtime — without it,
// non-ASCII text in the source PDF (CJK, Cyrillic, Arabic, …) renders
// as blanks because PDF.js cannot reverse-resolve the embedded
// glyph IDs to Unicode. ``standardFontDataUrl`` provides Adobe Type 1
// base font data (Helvetica / Times / Courier) for PDFs that
// reference those fonts without embedding them. Both directories are
// copied from ``node_modules/pdfjs-dist/`` into ``web/public/`` by
// ``scripts/copy-pdf-assets.mjs`` (wired into the ``dev`` / ``build``
// / ``postinstall`` scripts), so they're served as Next.js static
// assets at the same origin — no CDN dependency, works offline / in
// CN / private-cloud deploys.
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
    <div className="border-border/70 bg-card flex flex-col gap-0 overflow-hidden rounded-xl border shadow-sm">
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
      </div>

      {/* Parsed-text tab removed per earayu2 directive msg=153f4b85
          ("把解析文档从前端隐藏，目前效果不好，干脆别展示了") — the
          parsed markdown is still produced + persisted by the BE
          indexing pipeline and consumed by retrieval / agent / graph
          flows; only the user-facing preview tab is hidden. The
          original document (PDF / image / fallback download CTA)
          remains the single content surface here. */}
      <div className="bg-background/50 m-0 p-4">
        {pdfPreviewUrl ? (
          <PDFDocument
            file={pdfPreviewUrl}
            options={PDF_OPTIONS}
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
        ) : isTextPreview && markdownContent ? (
          <Card className="border-border/70 py-0 shadow-sm">
            <CardContent className="p-5">
              <Markdown>{markdownContent}</Markdown>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-border/70 py-0 shadow-sm">
            <CardContent className="text-muted-foreground flex min-h-72 flex-col items-center justify-center gap-4 p-8 text-center text-sm">
              <FileText className="size-10" />
              <div className="space-y-1">
                <div className="text-foreground text-sm font-medium">
                  {documentText('original_preview_unavailable')}
                </div>
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
      </div>
    </div>
  );
};
