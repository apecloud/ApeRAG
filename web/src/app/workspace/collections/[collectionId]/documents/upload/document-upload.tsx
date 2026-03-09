'use client';

import { UploadDocumentResponseStatusEnum } from '@/api';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import async from 'async';
import { Bs1CircleFill, Bs2CircleFill } from 'react-icons/bs';
import { TextImport } from './import/text-import';
import { UrlImport } from './import/url-import';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { Checkbox } from '@/components/ui/checkbox';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  FileUpload,
  FileUploadClear,
  FileUploadDropzone,
  FileUploadTrigger,
} from '@/components/ui/file-upload';
import { Progress } from '@/components/ui/progress';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import {
  ColumnDef,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import _ from 'lodash';
import {
  BrushCleaning,
  ChevronRight,
  EllipsisVertical,
  FileText,
  Globe,
  LoaderCircle,
  Save,
  Trash,
  Upload,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';
import { toast } from 'sonner';

/**
 * A staging-area entry. Two sources:
 *   1. Loaded from DB (GET /staged) — `file` is undefined, document_id always set.
 *   2. An in-progress file upload  — `file` holds the real File object, document_id
 *      is set once the upload completes.
 */
type DocumentsWithFile = {
  /** Present only for in-progress file uploads. */
  file?: File;
  filename: string;
  size: number;
  progress: number;
  progress_status: 'pending' | 'uploading' | 'success' | 'failed';
  document_id?: string;
  status?: UploadDocumentResponseStatusEnum;
};

type AsyncTask = (callback: (error?: Error | null) => void) => void;

let uploadController: AbortController | undefined;

export const DocumentUpload = () => {
  const { collection } = useCollectionContext();
  const page_documents = useTranslations('page_documents');
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentsWithFile[]>([]);
  const [step, setStep] = useState<number>(1);
  const [urlDialogOpen, setUrlDialogOpen] = useState(false);
  const [textDialogOpen, setTextDialogOpen] = useState(false);
  const [rowSelection, setRowSelection] = useState({});
  const [isUploading, setIsUploading] = useState(false);
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 20 });
  const uploadingFilesRef = useRef<Set<string>>(new Set());

  // ── Staged document helpers ──────────────────────────────────────────────

  /**
   * Load UPLOADED documents from the DB and merge them with any currently
   * in-progress local uploads. DB records are the source of truth for
   * completed items; in-progress uploads (no document_id yet) are kept as-is.
   */
  const refreshStaged = useCallback(async () => {
    if (!collection.id) return;
    try {
      const res =
        await apiClient.defaultApi.collectionsCollectionIdDocumentsStagedGet({
          collectionId: collection.id,
        });
      const staged: DocumentsWithFile[] = res.data.documents.map((doc) => ({
        filename: doc.filename,
        size: doc.size,
        document_id: doc.document_id,
        status: doc.status as UploadDocumentResponseStatusEnum,
        progress: 100,
        progress_status: 'success' as const,
      }));
      setDocuments((prev) => {
        // Keep uploads that are still in progress (no document_id assigned yet)
        const inProgress = prev.filter((d) => d.file && !d.document_id);
        return [...staged, ...inProgress];
      });
    } catch (err) {
      console.error('Failed to load staged documents', err);
    }
  }, [collection.id]);

  // Load staged documents when the page opens
  useEffect(() => {
    refreshStaged();
  }, [refreshStaged]);

  // ── Import success callbacks ─────────────────────────────────────────────

  const handleUrlImportSuccess = useCallback(
    (
      results: {
        url: string;
        fetch_status: 'success' | 'error';
        document_id?: string;
        filename?: string;
        size?: number;
        status?: string;
        error?: string;
      }[],
    ) => {
      const succeeded = results.filter(
        (r) => r.fetch_status === 'success' && r.document_id,
      );
      const failed = results.filter((r) => r.fetch_status === 'error');

      if (succeeded.length > 0) {
        toast.success(
          page_documents('import_url_success', {
            count: String(succeeded.length),
          }),
        );
        refreshStaged();
      }
      if (failed.length > 0) {
        toast.error(
          page_documents('import_url_partial', {
            succeeded: String(succeeded.length),
            failed: String(failed.length),
          }),
        );
      }
    },
    [page_documents, refreshStaged],
  );

  const handleTextImportSuccess = useCallback(() => {
    toast.success(page_documents('import_text_success'));
    refreshStaged();
  }, [page_documents, refreshStaged]);

  // ── Confirm (save to collection) ─────────────────────────────────────────

  const handleSaveToCollection = useCallback(async () => {
    if (!collection.id) return;
    const res =
      await apiClient.defaultApi.collectionsCollectionIdDocumentsConfirmPost({
        collectionId: collection.id,
        confirmDocumentsRequest: {
          document_ids: documents
            .map((doc) => doc.document_id || '')
            .filter((id) => !_.isEmpty(id)),
        },
      });
    if (res.status === 200) {
      toast.success('Document added successfully');
      router.push(`/workspace/collections/${collection.id}/documents`);
    }
  }, [collection.id, documents, router]);

  // ── Upload machinery ─────────────────────────────────────────────────────

  const stopUpload = useCallback(() => {
    setIsUploading(false);
    uploadController?.abort();
  }, []);

  useEffect(() => stopUpload, [stopUpload]);

  const startUpload = useCallback(
    (docs: DocumentsWithFile[]) => {
      const filesToUpload = docs.filter((doc) => {
        if (!doc.file) return false;
        const fileKey = `${doc.file.name}-${doc.file.size}-${doc.file.lastModified}`;
        return (
          doc.progress_status === 'pending' &&
          !doc.document_id &&
          !uploadingFilesRef.current.has(fileKey)
        );
      });

      if (filesToUpload.length === 0) return;

      filesToUpload.forEach((doc) => {
        const fileKey = `${doc.file!.name}-${doc.file!.size}-${doc.file!.lastModified}`;
        uploadingFilesRef.current.add(fileKey);
      });

      const tasks: AsyncTask[] = filesToUpload.map((_doc) => async (callback) => {
        const file = _doc.file!;
        if (!collection?.id) {
          callback();
          return;
        }

        const networkSimulation = async () => {
          const totalChunks = 100;
          let uploadedChunks = 0;
          for (let i = 0; i < totalChunks; i++) {
            await new Promise((resolve) =>
              setTimeout(resolve, Math.random() * 5 + 5),
            );
            uploadedChunks++;
            const progress = (uploadedChunks / totalChunks) * 99;
            setDocuments((docs) => {
              const doc = docs.find((d) => d.file && _.isEqual(d.file, file));
              if (doc) {
                doc.progress = Number(progress.toFixed(0));
                doc.progress_status = 'uploading';
              }
              return [...docs];
            });
          }
        };

        try {
          const [res] = await Promise.all([
            apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost(
              { collectionId: collection.id, file },
              { timeout: 1000 * 30 },
            ),
            networkSimulation(),
          ]);

          setDocuments((docs) => {
            const doc = docs.find((d) => d.file && _.isEqual(d.file, file));
            if (doc && res.data.document_id) {
              Object.assign(doc, {
                ...res.data,
                progress: 100,
                progress_status: 'success',
              });
            }
            return [...docs];
          });
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
        } catch (err) {
          setDocuments((docs) => {
            const doc = docs.find((d) => d.file && _.isEqual(d.file, file));
            if (doc) {
              Object.assign(doc, { progress: 0, progress_status: 'failed' });
            }
            return [...docs];
          });
        } finally {
          const fileKey = `${file.name}-${file.size}-${file.lastModified}`;
          uploadingFilesRef.current.delete(fileKey);
        }
        callback(null);
      });

      setIsUploading(true);
      uploadController = new AbortController();
      async.eachLimit(
        tasks,
        3,
        (task, callback) => {
          if (uploadController?.signal.aborted) {
            setIsUploading(false);
            callback(new Error('stop upload'));
          } else {
            task(callback);
          }
        },
        (err) => {
          if (err) console.error('Upload error:', err);
          else console.log('Upload completed');
          setIsUploading(false);
        },
      );
    },
    [collection.id],
  );

  const handleRemoveFile = useCallback(
    (item: DocumentsWithFile) => {
      setDocuments((docs) =>
        docs.filter((doc) =>
          item.file
            ? !_.isEqual(doc.file, item.file)
            : doc.document_id !== item.document_id,
        ),
      );
    },
    [],
  );

  // ── DataGrid columns ─────────────────────────────────────────────────────

  const columns: ColumnDef<DocumentsWithFile>[] = useMemo(
    () => [
      {
        id: 'select',
        header: ({ table }) => (
          <div className="flex items-center justify-center">
            <Checkbox
              checked={
                table.getIsAllPageRowsSelected() ||
                (table.getIsSomePageRowsSelected() && 'indeterminate')
              }
              onCheckedChange={(value) =>
                table.toggleAllPageRowsSelected(!!value)
              }
              aria-label="Select all"
            />
          </div>
        ),
        cell: ({ row }) => (
          <div className="flex items-center justify-center">
            <Checkbox
              checked={row.getIsSelected()}
              onCheckedChange={(value) => row.toggleSelected(!!value)}
              aria-label="Select row"
            />
          </div>
        ),
      },
      {
        accessorKey: 'filename',
        header: page_documents('filename'),
        cell: ({ row }) => {
          const { filename, file, size } = row.original;
          const mimeType = file?.type ?? '';
          const extension = _.last(mimeType.split('/')) || _.last(filename.split('.')) || '';
          return (
            <div className="flex w-full flex-row items-center gap-2">
              <div className="size-6">
                <FileIcon
                  color="var(--primary)"
                  extension={extension}
                  {..._.get(defaultStyles, extension)}
                />
              </div>
              <div>
                <div className="max-w-md truncate">{filename}</div>
                <div className="text-muted-foreground text-sm">
                  {(size / 1000).toFixed(0) + ' KB'}
                </div>
              </div>
            </div>
          );
        },
      },
      {
        header: page_documents('file_type'),
        cell: ({ row }) => {
          const { file, filename } = row.original;
          return file?.type ?? _.last(filename.split('.')) ?? '—';
        },
      },
      {
        header: page_documents('upload_progress'),
        cell: ({ row }) => (
          <div className="flex w-50 flex-col">
            <Progress
              value={row.original.progress}
              className="h-1.5 transition-all"
            />
            <div className="text-muted-foreground flex flex-row justify-between text-xs">
              <div>{row.original.progress}%</div>
              <div
                data-status={row.original.progress_status}
                className="data-[status=failed]:text-red-600 data-[status=success]:text-emerald-600 data-[status=uploading]:text-amber-500"
              >
                {row.original.progress_status}
              </div>
            </div>
          </div>
        ),
      },
      {
        id: 'actions',
        cell: ({ row }) => (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="data-[state=open]:bg-muted text-muted-foreground flex size-8"
                size="icon"
              >
                <EllipsisVertical />
                <span className="sr-only">Open menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-32">
              <DropdownMenuItem
                variant="destructive"
                onClick={() => handleRemoveFile(row.original)}
              >
                <Trash /> {page_documents('remove_file')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ],
    [handleRemoveFile, page_documents],
  );

  const table = useReactTable({
    data: documents,
    columns,
    state: { rowSelection, pagination },
    getRowId: (row) =>
      String(row.document_id ?? (row.file ? `${row.file.name}-${row.file.lastModified}` : row.filename)),
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  });

  const onFileReject = useCallback((file: File, message: string) => {
    toast.error(message, {
      description: `"${file.name.length > 20 ? `${file.name.slice(0, 20)}...` : file.name}" has been rejected`,
    });
  }, []);

  const onFileValidate = useCallback(
    (file: File): string | null => {
      const exists = documents.some(
        (doc) =>
          doc.filename === file.name &&
          (doc.file
            ? doc.file.size === file.size &&
              doc.file.lastModified === file.lastModified
            : true),
      );
      if (exists) return 'File already exists.';
      return null;
    },
    [documents],
  );

  useEffect(() => {
    if (
      documents.length === 0 ||
      documents.some((d) => !d.document_id || d.progress_status !== 'success')
    ) {
      setStep(1);
    } else {
      setStep(2);
    }
  }, [documents]);

  const tabBtnClass = cn(
    'flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition-colors',
    'text-muted-foreground hover:text-foreground hover:bg-accent cursor-pointer',
  );

  // Only real File objects are passed to FileUpload (for its internal dedup)
  const realFiles = useMemo(
    () => documents.filter((d) => d.file).map((d) => d.file!),
    [documents],
  );

  return (
    <>
      <FileUpload
        maxFiles={1000}
        maxSize={100 * 1024 * 1024}
        className="w-full gap-4"
        accept=".pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xls,.xlsx"
        value={realFiles}
        onValueChange={(files) => {
          const newDocs: DocumentsWithFile[] = [];
          const newFilesToUpload: DocumentsWithFile[] = [];

          files.forEach((file) => {
            const existingDoc = documents.find(
              (doc) => doc.file && _.isEqual(doc.file, file),
            );
            if (existingDoc) {
              newDocs.push(existingDoc);
            } else {
              const newDoc: DocumentsWithFile = {
                file,
                filename: file.name,
                size: file.size,
                progress_status: 'pending',
                progress: 0,
              };
              newDocs.push(newDoc);
              newFilesToUpload.push(newDoc);
            }
          });

          // Preserve DB-loaded staged docs; replace in-progress list
          const dbDocs = documents.filter((d) => !d.file);
          setDocuments([...dbDocs, ...newDocs]);

          if (newFilesToUpload.length > 0) {
            startUpload(newFilesToUpload);
          }
        }}
        onFileReject={onFileReject}
        onFileValidate={onFileValidate}
        multiple
        disabled={isUploading}
      >
        {/* Toolbar */}
        <div className="flex flex-row items-center justify-between text-sm">
          <div className="text-muted-foreground flex h-9 flex-row items-center gap-2">
            <div
              className={cn(
                'flex flex-row items-center gap-1',
                step === 1 ? 'text-primary' : '',
              )}
            >
              <Bs1CircleFill className="size-5" />
              <div>{page_documents('browse_files')}</div>
            </div>
            <ChevronRight className="size-4" />
            <div
              className={cn(
                'flex flex-row items-center gap-1',
                step === 2 ? 'text-primary' : '',
              )}
            >
              <Bs2CircleFill className="size-5" />
              <div>{page_documents('add_documents')}</div>
            </div>
          </div>

          <div className="flex flex-row gap-2">
            {documents.length > 0 && (
              <FileUploadClear asChild disabled={isUploading}>
                <Button variant="outline" className="cursor-pointer">
                  <BrushCleaning />
                  <span className="hidden lg:inline">
                    {page_documents('clear_files')}
                  </span>
                </Button>
              </FileUploadClear>
            )}

            {isUploading ? (
              <Button className="cursor-pointer" onClick={stopUpload}>
                <LoaderCircle className="animate-spin" />
                <span className="hidden lg:inline">Stop</span>
              </Button>
            ) : (
              <Button
                className="cursor-pointer"
                onClick={handleSaveToCollection}
                disabled={step !== 2}
              >
                <Save />
                <span className="hidden lg:inline">
                  {page_documents('add_documents')}
                </span>
              </Button>
            )}
          </div>
        </div>

        {/* Main content: drop zone or file list */}
        {documents.length === 0 ? (
          <FileUploadDropzone className="cursor-pointer rounded-lg border p-16">
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="flex items-center justify-center rounded-full border p-2.5">
                <Upload className="text-muted-foreground size-6" />
              </div>
              <p className="text-sm font-medium">
                {page_documents('drag_drop_files_here')}
              </p>
              <p className="text-muted-foreground text-xs">
                {page_documents('or_click_to_browse_files')}
              </p>
            </div>
          </FileUploadDropzone>
        ) : (
          <>
            <DataGrid table={table} />
            <DataGridPagination table={table} />
          </>
        )}

        {/* Source picker — always anchored at the bottom */}
        <div className="flex items-center gap-1 rounded-lg border bg-muted/30 px-4 py-2">
          {documents.length > 0 && (
            <span className="text-muted-foreground mr-2 text-xs">
              {page_documents('add_more_sources')}
            </span>
          )}
          <FileUploadTrigger asChild>
            <button className={tabBtnClass}>
              <Upload className="size-3.5" />
              {page_documents('import_source_file')}
            </button>
          </FileUploadTrigger>
          <button
            className={tabBtnClass}
            onClick={() => setUrlDialogOpen(true)}
          >
            <Globe className="size-3.5" />
            {page_documents('import_source_url')}
          </button>
          <button
            className={tabBtnClass}
            onClick={() => setTextDialogOpen(true)}
          >
            <FileText className="size-3.5" />
            {page_documents('import_source_text')}
          </button>
        </div>
      </FileUpload>

      {/* URL import dialog */}
      <Dialog open={urlDialogOpen} onOpenChange={setUrlDialogOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{page_documents('import_url_title')}</DialogTitle>
          </DialogHeader>
          <UrlImport
            onSuccess={(results) => {
              handleUrlImportSuccess(results);
              // Only auto-close when every URL succeeded; keep open on partial failure
              // so the user can read the error details before dismissing.
              const hasFailures = results.some((r) => r.fetch_status === 'error');
              if (!hasFailures) setUrlDialogOpen(false);
            }}
          />
        </DialogContent>
      </Dialog>

      {/* Text import dialog */}
      <Dialog open={textDialogOpen} onOpenChange={setTextDialogOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{page_documents('import_text_title')}</DialogTitle>
          </DialogHeader>
          <TextImport
            onSuccess={() => {
              handleTextImportSuccess();
              setTextDialogOpen(false);
            }}
          />
        </DialogContent>
      </Dialog>
    </>
  );
};
