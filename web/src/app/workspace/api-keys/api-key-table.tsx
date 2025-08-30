'use client';

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

import { CopyToClipboard } from '@/components/copy-to-clipboard';

import { z } from 'zod';

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

import { ApiKey } from '@/api';
import { FormatDate } from '@/components/format-date';
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  Plus,
  SquarePen,
  Trash,
} from 'lucide-react';

import { ApiKeyActions } from './api-key-actions';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { Input } from '@/components/ui/input';
import { useTranslations } from 'next-intl';
export const schema = z.object({
  id: z.number(),
  header: z.string(),
  type: z.string(),
  status: z.string(),
  target: z.string(),
  limit: z.string(),
  reviewer: z.string(),
});

export function ApiKeyTable({ data }: { data: ApiKey[] }) {
  const [rowSelection, setRowSelection] = React.useState({});
  const page_api_keys = useTranslations('page_api_keys');
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

  const columns: ColumnDef<ApiKey>[] = [
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
      accessorKey: 'key',
      header: page_api_keys('api_keys'),
      cell: ({ row }) => {
        return (
          <div className="flex flex-row items-center gap-2">
            <span>{row.original.key}</span>
            {row.original.key && (
              <CopyToClipboard variant="ghost" text={row.original.key} />
            )}
          </div>
        );
      },
    },
    {
      accessorKey: 'description',
      header: page_api_keys('description'),
    },
    {
      accessorKey: 'created_at',
      header: page_api_keys('creation_time'),
      cell: ({ row }) => {
        if (row.original.created_at) {
          return <FormatDate datetime={new Date(row.original.created_at)} />;
        }
      },
    },
    {
      accessorKey: 'last_used_at',
      header: page_api_keys('last_used_time'),
      cell: ({ row }) => {
        if (row.original.last_used_at) {
          return <FormatDate datetime={new Date(row.original.last_used_at)} />;
        }
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
          <DropdownMenuContent align="end" className="w-32">
            <ApiKeyActions action="edit" apiKey={row.original}>
              <DropdownMenuItem>
                <SquarePen /> {page_api_keys('edit_api_keys')}
              </DropdownMenuItem>
            </ApiKeyActions>
            <DropdownMenuSeparator />
            <ApiKeyActions action="delete" apiKey={row.original}>
              <DropdownMenuItem variant="destructive">
                <Trash /> {page_api_keys('delete_api_key')}
              </DropdownMenuItem>
            </ApiKeyActions>
          </DropdownMenuContent>
        </DropdownMenu>
      ),
    },
  ];

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
        <div>
          <Input
            placeholder={page_api_keys('search_api_keys')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="flex items-center gap-2">
          <ApiKeyActions action="add">
            <Button>
              <Plus />
              <span className="hidden lg:inline">
                {page_api_keys('add_api_keys')}
              </span>
            </Button>
          </ApiKeyActions>
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
        </div>
      </div>
      <DataGrid table={table} />
      <DataGridPagination table={table} />
    </div>
  );
}
