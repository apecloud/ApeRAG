'use client';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
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
import type { SharedCollection } from '@/features/marketplace/types';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { cn, objectKeys, parsePageParams } from '@/lib/utils';
import { ChevronDown, Columns3, FileText, Search } from 'lucide-react';

import { getDocumentStatusColor } from '@/app/workspace/collections/tools';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { DocumentIndexStatus } from './document-index-status';

export function DocumentsTable({
  collection,
  data,
  pageCount,
}: {
  collection: SharedCollection;
  data: Document[];
  pageCount?: number;
}) {
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const page_documents = useTranslations('page_documents');
  const page_collections = useTranslations('page_collections');
  const page_marketplace = useTranslations('page_marketplace');
  const [sorting, setSorting] = React.useState<SortingState>([]);

  const [searchValue, setSearchValue] = React.useState<string>('');
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();

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

  const columns: ColumnDef<Document>[] = React.useMemo(() => {
    const indexCols: ColumnDef<Document>[] = [];
    DOCUMENT_INDEX_TYPES.map((key) => {
      const accessorKey = key.toLowerCase() + '_index_status';

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
            <div className="flex flex-row items-center gap-2">
              <div className="h-8 w-6">{icon}</div>
              <div>
                <div className="max-w-60 truncate">
                  {row.original.vector_index_status === 'ACTIVE' ? (
                    <Link
                      href={`/marketplace/collections/${collection.id}/documents/${row.original.id}`}
                      className={cn('hover:text-primary')}
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
                  {(Number(row.original.size || 0) / 1000).toFixed(2)} KB
                </div>
              </div>
            </div>
          );
        },
      },
      ...indexCols,
      {
        accessorKey: 'updated',
        header: page_documents('last_updated'),
        cell: ({ row }) => {
          return row.original.updated ? (
            <FormatDate datetime={new Date(row.original.updated)} />
          ) : (
            ''
          );
        },
      },
    ];
    return cols;
  }, [collection.id, page_collections, page_documents]);

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
    onPaginationChange: (fn) => {
      // @ts-expect-error onPaginationChange
      const { pageIndex, pageSize } = fn({
        pageIndex: query.page - 1,
        pageSize: query.pageSize,
      });
      handleSearch({
        page: pageIndex + 1,
        pageSize,
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
    <div className="border-border/70 bg-card flex flex-col gap-4 rounded-xl border p-4 shadow-sm">
      <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="flex min-w-0 items-center gap-3">
          <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
            <FileText className="size-4" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-medium">
              {page_documents('metadata.title')}
            </div>
            <div className="text-muted-foreground text-xs">
              {page_marketplace('document_count', {
                count: data.length,
              })}
            </div>
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="relative min-w-64">
            <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
            <Input
              className="bg-background/70 h-10 rounded-lg pl-9"
              placeholder={page_documents('search_document')}
              value={searchValue}
              onChange={(e) => setSearchValue(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSearch({
                    search: e.currentTarget.value,
                  });
                }
              }}
            />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="h-10">
                <Columns3 className="size-4" />
                <ChevronDown className="size-4" />
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
        </div>
      </div>
      <DataGrid table={table} className="border-border/70 rounded-xl" />
      <DataGridPagination table={table} className="px-1" />
    </div>
  );
}
