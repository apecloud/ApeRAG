'use client';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import {
  ColumnDef,
  ColumnFiltersState,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  VisibilityState,
} from '@tanstack/react-table';
import * as React from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';

import { DOCUMENT_INDEX_TYPES, type Document } from '@/features/document/types';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { Badge } from '@/components/ui/badge';
import { cn, objectKeys, parsePageParams } from '@/lib/utils';
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  FolderSync,
  Plus,
  Search,
  Trash,
} from 'lucide-react';

import { getDocumentStatusColor } from '@/app/workspace/collections/tools';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { isVisibleDocumentIndexType } from '../../feature-visibility';
import { DocumentDelete } from './document-delete';
import { DocumentIndexStatus } from './document-index-status';
import { DocumentReBuildFailedIndex } from './document-rebuild-failed-index';
import { DocumentReBuildIndex } from './document-rebuild-index';

const FAST_POLL_INTERVAL_MS = 5_000;
const FAST_BACKGROUND_POLL_INTERVAL_MS = 30_000;
const SLOW_POLL_INTERVAL_MS = 60_000;
const SLOW_BACKGROUND_POLL_INTERVAL_MS = 120_000;

const NON_TERMINAL_DOCUMENT_STATUSES = new Set([
  'UPLOADED',
  'PENDING',
  'RUNNING',
  'DELETING',
]);

const NON_TERMINAL_INDEX_STATUSES = new Set(['PENDING', 'RUNNING']);

const DOCUMENT_STATUS_CLASS: Record<string, string> = {
  COMPLETE: 'bg-accent-soft text-accent-ink border-accent-soft',
  RUNNING: 'bg-secondary text-foreground border-transparent',
  PENDING: 'bg-secondary text-muted-foreground border-transparent',
  UPLOADED: 'bg-secondary text-muted-foreground border-transparent',
  FAILED: 'bg-destructive/10 text-destructive border-destructive/20',
  EXPIRED: 'bg-secondary text-muted-foreground border-transparent line-through',
  DELETED: 'bg-secondary text-muted-foreground border-transparent line-through',
  DELETING:
    'bg-secondary text-muted-foreground border-transparent line-through',
};

const formatFileSize = (size?: number | null) => {
  const kb = Number(size || 0) / 1000;
  if (kb < 1000) return `${kb.toFixed(2)} KB`;
  return `${(kb / 1000).toFixed(2)} MB`;
};

const formatStatus = (status: string) =>
  status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();

const DocumentStatusBadge = ({ status }: { status?: string | null }) => {
  if (!status) return null;
  return (
    <Badge
      variant="outline"
      className={cn(
        'rounded-sm border font-mono text-[10px] uppercase tracking-normal',
        DOCUMENT_STATUS_CLASS[status] ||
          'bg-secondary text-muted-foreground border-transparent',
      )}
    >
      {formatStatus(status)}
    </Badge>
  );
};

const getEnabledIndexStatusKeys = (
  config?: {
    enable_fulltext?: boolean | null;
    enable_knowledge_graph?: boolean | null;
    enable_vector?: boolean | null;
  } | null,
) => {
  const keys: Array<keyof Document> = [];
  if (config?.enable_fulltext) keys.push('fulltext_index_status');
  if (config?.enable_knowledge_graph) keys.push('graph_index_status');
  if (config?.enable_vector) keys.push('vector_index_status');
  return keys;
};

const hasNonTerminalDocumentState = (
  document: Document,
  enabledIndexStatusKeys: Array<keyof Document>,
) => {
  if (document.status && NON_TERMINAL_DOCUMENT_STATUSES.has(document.status)) {
    return true;
  }

  return enabledIndexStatusKeys.some((key) => {
    const status = document[key];
    return (
      typeof status === 'string' && NON_TERMINAL_INDEX_STATUSES.has(status)
    );
  });
};

export function DocumentsTable({
  data,
  pageCount,
}: {
  data: Document[];
  pageCount?: number;
}) {
  const { collection } = useCollectionContext();
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const page_documents = useTranslations('page_documents');
  const page_collections = useTranslations('page_collections');
  const [searchValue, setSearchValue] = React.useState<string>('');
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();
  const [isPageVisible, setIsPageVisible] = React.useState(
    () =>
      typeof document === 'undefined' || document.visibilityState === 'visible',
  );

  const query = React.useMemo(() => {
    return {
      ...parsePageParams({
        page: searchParams.get('page'),
        pageSize: searchParams.get('pageSize'),
      }),
      search: searchParams.get('search'),
    };
  }, [searchParams]);

  React.useEffect(() => {
    setSearchValue(query.search || '');
  }, [query]);

  React.useEffect(() => {
    const handleVisibilityChange = () => {
      setIsPageVisible(document.visibilityState === 'visible');
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const handleSearch = React.useCallback(
    (params: { page?: number; pageSize?: number; search?: string }) => {
      const urlSearchParams = new URLSearchParams();
      const data = { ...query, ...params };
      objectKeys(data).forEach((key) => {
        const value = data[key];
        if (value !== null && value !== undefined && value !== '') {
          urlSearchParams.set(key, String(value));
        }
      });
      router.push(`${pathname}?${urlSearchParams.toString()}`);
    },
    [query, router, pathname],
  );

  const enabledIndexStatusKeys = React.useMemo(
    () => getEnabledIndexStatusKeys(collection.config),
    [collection.config],
  );

  const hasNonTerminalRows = React.useMemo(
    () =>
      data.some((document) =>
        hasNonTerminalDocumentState(document, enabledIndexStatusKeys),
      ),
    [data, enabledIndexStatusKeys],
  );

  const hasDirtySearchInput = searchValue !== (query.search || '');
  const [, startRefreshTransition] = React.useTransition();

  const pollIntervalMs = React.useMemo(() => {
    if (hasDirtySearchInput) return null;

    if (hasNonTerminalRows) {
      return isPageVisible
        ? FAST_POLL_INTERVAL_MS
        : FAST_BACKGROUND_POLL_INTERVAL_MS;
    }

    return isPageVisible
      ? SLOW_POLL_INTERVAL_MS
      : SLOW_BACKGROUND_POLL_INTERVAL_MS;
  }, [hasDirtySearchInput, hasNonTerminalRows, isPageVisible]);

  React.useEffect(() => {
    if (!pollIntervalMs) return;

    const refreshTimer = window.setTimeout(() => {
      startRefreshTransition(() => {
        router.refresh();
      });
    }, pollIntervalMs);

    return () => {
      window.clearTimeout(refreshTimer);
    };
  }, [pollIntervalMs, router, startRefreshTransition]);

  const columns: ColumnDef<Document>[] = React.useMemo(() => {
    const indexCols: ColumnDef<Document>[] = [];
    DOCUMENT_INDEX_TYPES.filter(isVisibleDocumentIndexType).map((key) => {
      const accessorKey = key.toLowerCase() + '_index_status';

      const config = collection.config;
      let enabled: boolean | undefined;
      switch (key) {
        case 'FULLTEXT':
          enabled = Boolean(config?.enable_fulltext);
          break;
        case 'GRAPH':
          enabled = Boolean(config?.enable_knowledge_graph);
          break;
        case 'VECTOR':
          enabled = Boolean(config?.enable_vector);
          break;
        default:
          enabled = false;
      }
      if (enabled) {
        indexCols.push({
          accessorKey,
          header: page_collections(`index_type_${key}.title`),
          cell: ({ row }) => (
            <DocumentIndexStatus
              document={row.original}
              accessorKey={accessorKey}
            />
          ),
        });
      }
    });

    const cols: ColumnDef<Document>[] = [
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
        accessorKey: 'name',
        header: page_documents('file'),
        cell: ({ row }) => {
          const extension =
            row.original.name?.split('.').pop()?.toLowerCase() ||
            ('unknow' as keyof typeof defaultStyles);
          const iconProps =
            defaultStyles[extension as keyof typeof defaultStyles];
          const icon = (
            <FileIcon
              color="var(--primary)"
              extension={extension}
              {...iconProps}
            />
          );
          return (
            <div className="flex min-w-0 flex-row items-center gap-3">
              <div className="bg-muted flex h-10 w-9 shrink-0 items-center justify-center rounded-lg p-1.5">
                {icon}
              </div>
              <div className="min-w-0">
                <div className="max-w-80 truncate text-sm font-medium">
                  {row.original.vector_index_status === 'ACTIVE' ? (
                    <Link
                      href={`/workspace/collections/${collection.id}/documents/${row.original.id}`}
                      className={cn('hover:text-primary transition-colors')}
                    >
                      {row.original.name}
                    </Link>
                  ) : (
                    <span
                      className={getDocumentStatusColor(row.original.status)}
                    >
                      {row.original.name}
                    </span>
                  )}
                </div>
                <div className="text-muted-foreground text-xs">
                  {formatFileSize(row.original.size)}
                </div>
              </div>
            </div>
          );
        },
      },
      {
        accessorKey: 'status',
        header: page_documents('upload_progress'),
        cell: ({ row }) => <DocumentStatusBadge status={row.original.status} />,
      },
      ...indexCols,
      {
        accessorKey: 'updated',
        header: page_documents('last_updated'),
        cell: ({ row }) => {
          return row.original.updated ? (
            <span className="text-muted-foreground text-xs">
              <FormatDate datetime={new Date(row.original.updated)} />
            </span>
          ) : (
            ''
          );
        },
      },
      {
        id: 'actions',
        enableHiding: false,
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
            <DropdownMenuContent align="end" className="w-42">
              <DocumentReBuildIndex file={row.original}>
                <DropdownMenuItem>
                  <FolderSync /> {page_documents('index_rebuild')}
                </DropdownMenuItem>
              </DocumentReBuildIndex>
              <DropdownMenuSeparator />
              <DocumentDelete document={row.original}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> {page_documents('delete_document')}
                </DropdownMenuItem>
              </DocumentDelete>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [collection.config, collection.id, page_collections, page_documents]);

  const table = useReactTable({
    data,
    columns,
    manualPagination: true,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination: {
        pageIndex: query.page - 1,
        pageSize: query.pageSize,
      },
    },
    getRowId: (row) => String(row.id),
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    pageCount,
    // TanStack Table's `onPaginationChange` handler receives an
    // `Updater<PaginationState>` — either a `PaginationState` value or a
    // `(old: PaginationState) => PaginationState` function. Treating it as
    // only a function (as the previous `@ts-expect-error` version did) would
    // throw at runtime whenever TanStack dispatches a plain value, silently
    // eating pagination clicks. Handle both shapes explicitly.
    onPaginationChange: (updater) => {
      const current = {
        pageIndex: query.page - 1,
        pageSize: query.pageSize,
      };
      const next = typeof updater === 'function' ? updater(current) : updater;
      handleSearch({
        page: next.pageIndex + 1,
        pageSize: next.pageSize,
      });
    },
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="border-border/70 bg-card grid gap-3 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="relative max-w-xl lg:max-w-none">
          <Search className="text-muted-foreground pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2" />
          <Input
            className="bg-background/70 h-10 rounded-lg pl-9"
            placeholder={page_documents('search_document')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleSearch({
                  page: 1,
                  search: e.currentTarget.value,
                });
              }
            }}
          />
        </div>
        <div className="grid grid-cols-[1fr_auto_auto] items-center gap-2 sm:flex sm:flex-wrap sm:justify-end">
          <Button asChild className="min-w-0 cursor-pointer">
            <Link
              href={`/workspace/collections/${collection.id}/documents/upload`}
              className="justify-center"
            >
              <Plus />
              <span className="truncate">
                {page_documents('add_documents')}
              </span>
            </Link>
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Columns3 />
                <ChevronDown />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              {table
                .getAllColumns()
                .filter(
                  (column) =>
                    typeof column.accessorFn !== 'undefined' &&
                    column.getCanHide(),
                )
                .map((column) => {
                  return (
                    <DropdownMenuCheckboxItem
                      key={column.id}
                      className="capitalize"
                      checked={column.getIsVisible()}
                      onCheckedChange={(value) =>
                        column.toggleVisibility(!!value)
                      }
                    >
                      {String(column.columnDef.header)}
                    </DropdownMenuCheckboxItem>
                  );
                })}
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="icon" variant="outline">
                <EllipsisVertical />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-45">
              <DocumentReBuildFailedIndex>
                <DropdownMenuItem>
                  <FolderSync /> {page_documents('index_rebuild_failed')}
                </DropdownMenuItem>
              </DocumentReBuildFailedIndex>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      <DataGrid
        table={table}
        className="border-border/70 bg-card overflow-x-auto shadow-sm [&_table]:min-w-[760px]"
      />
      <DataGridPagination table={table} />
    </div>
  );
}
