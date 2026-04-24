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

import { z } from 'zod';

import { Badge } from '@/components/ui/badge';
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

import { FormatDate } from '@/components/format-date';
import type { ApiKey } from '@/features/api-key/types';
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  KeyRound,
  Plus,
  Search,
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
          <div className="flex items-center gap-3">
            <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
              <KeyRound className="size-4" />
            </div>
            <div className="min-w-0">
              <span className="block truncate font-mono text-sm">
                {row.original.key}
              </span>
              <Badge
                variant="outline"
                className="bg-muted mt-1 rounded-full font-mono text-[10px]"
              >
                {page_api_keys('masked_badge')}
              </Badge>
            </div>
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
    <div className="flex flex-col gap-5">
      <div className="border-border/70 bg-card grid gap-4 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="min-w-0">
          <div className="relative max-w-xl">
            <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
            <Input
              className="bg-background/70 h-10 rounded-lg pl-9"
              placeholder={page_api_keys('search_api_keys')}
              value={searchValue}
              onChange={(e) => setSearchValue(e.currentTarget.value)}
            />
          </div>
          <p className="text-muted-foreground mt-2 text-xs">
            {page_api_keys('masked_notice')}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
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
                <span className="hidden sm:inline">
                  {page_api_keys('columns')}
                </span>
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
      <div className="border-border/70 bg-card rounded-xl border p-3 shadow-sm">
        <DataGrid table={table} className="border-border/70 rounded-lg" />
      </div>
      <DataGridPagination table={table} />
    </div>
  );
}
