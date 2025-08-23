'use client';

import { UploadDocumentResponseStatusEnum } from '@/api';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import async from 'async';
import { Bs1CircleFill, Bs2CircleFill, Bs3CircleFill } from 'react-icons/bs';

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
  CloudUpload,
  EllipsisVertical,
  FolderSearch,
  LoaderCircle,
  Save,
  Trash,
  Upload,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';
import { toast } from 'sonner';

type DocumentsWithFile = {
  file: File;
  progress: number;
  progress_status: 'pending' | 'uploading' | 'success' | 'failed';

  document_id?: string;
  filename?: string;
  size?: number;
  status?: UploadDocumentResponseStatusEnum;
};

type AsyncTask = (callback: (error?: Error | null) => void) => void;

let uploadController: AbortController | undefined;

export const DocumentUpload = () => {
  const { collection } = useCollectionContext();
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentsWithFile[]>([]);
  const [step, setStep] = useState<number>(1);
  const [rowSelection, setRowSelection] = useState({});
  const [isUploading, setIsUploading] = useState(false);
  const [pagination, setPagination] = useState({
    pageIndex: 0,
    pageSize: 20,
  });

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

  const stopUpload = useCallback(() => {
    uploadController?.abort();
  }, []);

  /**
   * stop upload after page unmount
   */
  useEffect(() => stopUpload, [stopUpload]);

  const startUpload = useCallback(() => {
    const tasks: AsyncTask[] = documents
      .filter((doc) => !doc.document_id)
      .map((_doc) => async (callback) => {
        const file = _doc.file;
        if (!collection?.id) {
          callback();
          return;
        }
        const totalChunks = 20;
        let uploadedChunks = 0;
        for (let i = 0; i < totalChunks; i++) {
          // Simulate network delay (20-40ms per chunk)
          await new Promise((resolve) =>
            setTimeout(resolve, Math.random() * 20 + 20),
          );
          // Update progress for this specific file
          uploadedChunks++;
          const progress = (uploadedChunks / totalChunks) * 85;
          setDocuments((docs) => {
            const doc = docs.find((doc) => _.isEqual(doc.file, file));
            if (doc) {
              doc.progress = Number(progress.toFixed(0));
              doc.progress_status = 'uploading';
            }
            return [...docs];
          });
        }

        try {
          const res =
            await apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost(
              {
                collectionId: collection.id,
                file: _doc.file,
              },
              {
                timeout: 1000 * 30,
              },
            );
          setDocuments((docs) => {
            const doc = docs.find((doc) => _.isEqual(doc.file, file));
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
            const doc = docs.find((doc) => _.isEqual(doc.file, file));
            if (doc) {
              Object.assign(doc, {
                progress: 0,
                progress_status: 'failed',
              });
            }
            return [...docs];
          });
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
        if (err) {
          console.error('Error:', err);
        } else {
          console.log('upload complated');
        }
        setIsUploading(false);
      },
    );
  }, [collection.id, documents]);

  const handleRemoveFile = useCallback((item: DocumentsWithFile) => {
    setDocuments((docs) =>
      docs.filter((doc) => !_.isEqual(doc.file, item.file)),
    );
  }, []);

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
        header: 'Filename',
        cell: ({ row }) => {
          const file = row.original.file;
          const extension = _.last(file.type.split('/')) || '';
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
                <div className="max-w-md truncate">{file.name}</div>
                <div className="text-muted-foreground text-sm">
                  {(row.original.file.size / 1000).toFixed(0) + ' KB'}
                </div>
              </div>
            </div>
          );
        },
      },
      {
        header: 'Type',
        cell: ({ row }) => {
          return row.original.file.type;
        },
      },
      {
        header: 'Progress',
        cell: ({ row }) => {
          return (
            <div className="flex w-50 flex-col">
              <Progress value={row.original.progress} className="h-1.5" />
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
          );
        },
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
                <Trash /> Remove
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ],
    [handleRemoveFile],
  );

  const table = useReactTable({
    data: documents,
    columns,
    state: {
      rowSelection,
      pagination,
    },
    getRowId: (row) => String(row.document_id || row.file.name),
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
      const doc = documents.some(
        (doc) =>
          doc.file.name === file.name &&
          doc.file.size === file.size &&
          doc.file.lastModified === file.lastModified &&
          doc.file.type === file.type,
      );
      if (doc) {
        return 'File already exists.';
      }
      return null;
    },
    [documents],
  );

  useEffect(() => {
    if (documents.length === 0) {
      setStep(1);
    } else if (
      documents.filter((doc) => doc.progress_status === 'success').length !==
      documents.length
    ) {
      setStep(2);
    } else {
      setStep(3);
    }
  }, [documents]);

  return (
    <>
      <FileUpload
        maxFiles={1000}
        maxSize={5 * 1024 * 1024}
        className="w-full gap-4"
        accept=".pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xls,.xlsx"
        value={documents.map((f) => f.file)}
        onValueChange={(files) => {
          setDocuments((docs) => {
            const data: DocumentsWithFile[] = [];
            files.forEach((file) => {
              const doc = docs.find((doc) => _.isEqual(doc.file, file));
              data.push({
                file,
                progress_status: 'pending',
                progress: 0,
                ...doc,
              });
            });
            return data;
          });
        }}
        onFileReject={onFileReject}
        onFileValidate={onFileValidate}
        multiple
        disabled={isUploading}
      >
        <div className="flex flex-row items-center justify-between text-sm">
          <div className="text-muted-foreground flex h-9 flex-row items-center gap-2">
            <div
              className={cn(
                'flex flex-row items-center gap-1',
                step === 1 ? 'text-primary' : '',
              )}
            >
              <Bs1CircleFill className="size-5" />
              <div>Browse Files</div>
            </div>
            <ChevronRight className="size-4" />
            <div
              className={cn(
                'flex flex-row items-center gap-1',
                step === 2 ? 'text-primary' : '',
              )}
            >
              <Bs2CircleFill className="size-5" />
              <div>Upload</div>
            </div>
            <ChevronRight className="size-4" />
            <div
              className={cn(
                'flex flex-row items-center gap-1',
                step === 3 ? 'text-primary' : '',
              )}
            >
              <Bs3CircleFill className="size-5" />
              <div>Save to colelction</div>
            </div>
          </div>
          <div className="flex flex-row gap-2">
            <FileUploadClear asChild disabled={isUploading}>
              <Button variant="outline" className="cursor-pointer">
                <BrushCleaning />
                <span className="hidden lg:inline">Clear</span>
              </Button>
            </FileUploadClear>

            {documents.length > 0 && (
              <FileUploadTrigger asChild disabled={isUploading}>
                <Button variant="outline" className="cursor-pointer">
                  <FolderSearch />
                  <span className="hidden lg:inline">Browse files</span>
                </Button>
              </FileUploadTrigger>
            )}

            {step === 2 &&
              (isUploading ? (
                <Button
                  className="w-28 cursor-pointer"
                  onClick={() => stopUpload()}
                >
                  <LoaderCircle className="animate-spin" />
                  <span className="hidden lg:inline">Stop</span>
                </Button>
              ) : (
                <Button
                  className="w-28 cursor-pointer"
                  onClick={() => startUpload()}
                >
                  <CloudUpload />
                  <span className="hidden lg:inline">Upload</span>
                </Button>
              ))}
            {step === 3 && (
              <Button
                className="cursor-pointer"
                onClick={handleSaveToCollection}
              >
                <Save />
                <span className="hidden lg:inline">Save to Collection</span>
              </Button>
            )}
          </div>
        </div>

        {documents.length === 0 ? (
          <FileUploadDropzone className="cursor-pointer p-26">
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="flex items-center justify-center rounded-full border p-2.5">
                <Upload className="text-muted-foreground size-6" />
              </div>
              <p className="text-sm font-medium">Drag & drop files here</p>
              <p className="text-muted-foreground text-xs">
                Or click to browse (max 5MB for each file)
              </p>
            </div>
          </FileUploadDropzone>
        ) : (
          <>
            <DataGrid table={table} />
            <DataGridPagination table={table} />
          </>
        )}
      </FileUpload>
    </>
  );
};
