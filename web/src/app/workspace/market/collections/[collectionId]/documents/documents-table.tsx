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

import { Document, SharedCollection } from '@/api';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { cn, objectKeys, parsePageParams } from '@/lib/utils';
import _ from 'lodash';
import { ChevronDown, Columns3 } from 'lucide-react';

import {
  FileIndexTypes,
  getDocumentStatusColor,
} from '@/app/workspace/collections/tools';
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
    objectKeys(FileIndexTypes).map((key) => {
      const accessorKey = key.toLowerCase() + '_index_status';

      indexCols.push({
        accessorKey,
        header: FileIndexTypes[key].title,
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
        header: 'Name',
        cell: ({ row }) => {
          const extension =
            row.original.name?.split('.').pop()?.toLowerCase() ||
            ('unknow' as keyof typeof defaultStyles);
          const iconProps = _.get(defaultStyles, extension);
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
                      href={`/workspace/market/collections/${collection.id}/documents/${row.original.id}`}
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
        header: 'Last Updated',
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
  }, [collection]);

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
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-row items-center gap-2">
          <Input
            placeholder="Search"
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
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Columns3 />
                <span className="hidden lg:inline">Columns</span>
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
        </div>
      </div>
      <DataGrid table={table} />
      <DataGridPagination table={table} />
    </div>
  );
}
