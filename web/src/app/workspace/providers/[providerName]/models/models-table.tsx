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

import { LlmProvider, LlmProviderModel } from '@/api';
import { FormatDate } from '@/components/format-date';
import {
  ArrowLeft,
  ChevronDown,
  Columns3,
  EllipsisVertical,
  Plus,
  SquarePen,
  Trash,
} from 'lucide-react';

import { TableList, TableListPagination } from '@/components/table-list';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import Link from 'next/link';
import { ModelActions } from './model-actions';
export const schema = z.object({
  id: z.number(),
  header: z.string(),
  type: z.string(),
  status: z.string(),
  target: z.string(),
  limit: z.string(),
  reviewer: z.string(),
});

export function ModelsTable({
  provider,
  data,
  pathnamePrefix,
}: {
  provider: LlmProvider;
  data: LlmProviderModel[];
  pathnamePrefix: string;
}) {
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({
      created: false,
    });
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const [sorting, setSorting] = React.useState<SortingState>([
    {
      id: 'created',
      desc: true,
    },
    {
      id: 'model',
      desc: false,
    },
  ]);
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 20,
  });
  const [searchValue, setSearchValue] = React.useState<string>('');

  const columns: ColumnDef<LlmProviderModel>[] = [
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
      accessorKey: 'model',
      header: 'Model',
      cell: ({ row }) => {
        return (
          <Tooltip>
            <TooltipTrigger>
              <div className="flex flex-col gap-2">
                <div className="text-left">{row.original.model}</div>
                <div className="flex gap-1">
                  {row.original.tags
                    ?.filter((tag) => tag !== '__autogen__')
                    .map((tag, index) => {
                      return (
                        <Badge variant="secondary" key={index}>
                          {tag}
                        </Badge>
                      );
                    })}
                </div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="left">
              <div>
                <div>Context Window: {row.original.context_window}</div>
                <div>Max Input Token: {row.original.max_input_tokens}</div>
                <div>Max Output Token: {row.original.max_output_tokens}</div>
              </div>
            </TooltipContent>
          </Tooltip>
        );
      },
    },

    {
      accessorKey: 'context_window',
      header: 'Context Window',
    },
    {
      accessorKey: 'tokens',
      header: 'Tokens',
      cell: ({ row }) => {
        return (
          <Tooltip>
            <TooltipTrigger>
              <div className="flex h-3 items-center space-x-2 text-sm">
                <div>{row.original.max_input_tokens}</div>
                {row.original.max_input_tokens &&
                  row.original.max_output_tokens && (
                    <Separator orientation="vertical" />
                  )}
                <div>{row.original.max_output_tokens}</div>
              </div>
            </TooltipTrigger>
            <TooltipContent side="left">
              <div>
                <div>Max Input: {row.original.max_input_tokens}</div>
                <div>Max Output: {row.original.max_output_tokens}</div>
              </div>
            </TooltipContent>
          </Tooltip>
        );
      },
    },
    {
      accessorKey: 'api',
      header: 'API Type',
    },
    {
      accessorKey: 'created',
      header: 'Creation time',
      cell: ({ row }) => {
        if (row.original.created) {
          return <FormatDate datetime={new Date(row.original.created)} />;
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
            <ModelActions
              action="edit"
              provider={provider}
              model={row.original}
            >
              <DropdownMenuItem>
                <SquarePen /> Edit
              </DropdownMenuItem>
            </ModelActions>
            <DropdownMenuSeparator />
            <ModelActions
              action="delete"
              provider={provider}
              model={row.original}
            >
              <DropdownMenuItem variant="destructive">
                <Trash /> Delete
              </DropdownMenuItem>
            </ModelActions>
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
    getRowId: (row) => String(row.model),
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
          <Button asChild variant="outline">
            <Link href={`${pathnamePrefix}/providers`}>
              <ArrowLeft />
            </Link>
          </Button>
          <Input
            placeholder="Search"
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
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

          <ModelActions action="add" provider={provider}>
            <Button>
              <Plus />
              <span className="hidden lg:inline">Add Model</span>
            </Button>
          </ModelActions>
        </div>
      </div>
      <TableList idKey="model" table={table} />
      <TableListPagination table={table} />
    </div>
  );
}
