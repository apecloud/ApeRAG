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

import { Document, DocumentVectorIndexStatusEnum } from '@/api';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import { ChevronDown, Columns3, MonitorUp } from 'lucide-react';

export function FilesTable({ data }: { data: Document[] }) {
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 20,
  });
  const [searchValue, setSearchValue] = React.useState<string>('');

  const getStatusColor = React.useCallback(
    (status?: DocumentVectorIndexStatusEnum) => {
      const data = {
        ACTIVE: 'bg-green-500',
        CREATING: 'bg-emerald-500',
        DELETING: 'bg-pink-500',
        DELETION_IN_PROGRESS: 'bg-cyan-500',
        FAILED: 'bg-red-500',
        PENDING: 'bg-amber-500',
        SKIPPED: 'bg-gray-500',
      };
      return status ? data[status] : 'bg-gray-500';
    },
    [],
  );

  const columns: ColumnDef<Document>[] = React.useMemo(() => {
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
          const icon = <FileIcon extension={extension} {...iconProps} />;
          return (
            <div className="flex flex-row items-center gap-2">
              <div className="h-8 w-6">{icon}</div>
              <div>
                <div className="max-w-60 truncate">{row.original.name}</div>
                <div className="text-muted-foreground">
                  {(Number(row.original.size || 0) / 1000).toFixed(2)} KB
                </div>
              </div>
            </div>
          );
        },
      },
      {
        accessorKey: 'vector_index_status',
        header: 'Vector',
        cell: ({ row }) => {
          const status = row.original.vector_index_status;
          const color = getStatusColor(status);
          return (
            <div className="flex flex-row items-center gap-2">
              <div className={cn('size-2 rounded-4xl', color)}></div>
              {_.capitalize(status)}
            </div>
          );
        },
      },
      {
        accessorKey: 'fulltext_index_status',
        header: 'Fulltext',
        cell: ({ row }) => {
          const status = row.original.fulltext_index_status;
          const color = getStatusColor(status);
          return (
            <div className="flex flex-row items-center gap-2">
              <div className={cn('size-2 rounded-4xl', color)}></div>
              {_.capitalize(status)}
            </div>
          );
        },
      },
      {
        accessorKey: 'graph_index_status',
        header: 'Graph',
        cell: ({ row }) => {
          const status = row.original.graph_index_status;
          const color = getStatusColor(status);
          return (
            <div className="flex flex-row items-center gap-2">
              <div className={cn('size-2 rounded-4xl', color)}></div>
              {_.capitalize(status)}
            </div>
          );
        },
      },
      {
        accessorKey: 'summary_index_status',
        header: 'Summary',
        cell: ({ row }) => {
          const status = row.original.summary_index_status;
          const color = getStatusColor(status);
          return (
            <div className="flex flex-row items-center gap-2">
              <div className={cn('size-2 rounded-4xl', color)}></div>
              {_.capitalize(status)}
            </div>
          );
        },
      },
      {
        accessorKey: 'vision_index_status',
        header: 'Vision',
        cell: ({ row }) => {
          const status = row.original.vision_index_status;
          const color = getStatusColor(status);
          return (
            <div className="flex flex-row items-center gap-2">
              <div className={cn('size-2 rounded-4xl', color)}></div>
              {_.capitalize(status)}
            </div>
          );
        },
      },

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
      {
        id: 'action',
      },
    ];
    return cols;
  }, [getStatusColor]);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination,
      globalFilter: searchValue,
    },
    getRowId: (row) => String(row.id),
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    onPaginationChange: setPagination,
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
          />
        </div>
        <div className="flex items-center gap-2">
          <Button>
            <MonitorUp />
            <span className="hidden lg:inline">UpLoad</span>
          </Button>
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
