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

import { FormatDate } from '@/components/format-date';
import {
  ArrowLeft,
  ArrowUpDown,
  BetweenVerticalStart,
  ChevronDown,
  Columns3,
  EllipsisVertical,
  MessageSquareCode,
  Plus,
  SquarePen,
  Trash,
} from 'lucide-react';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type {
  Provider,
  ProviderModel,
  ProviderModelApi,
} from '@/features/providers/types';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { ModelActions } from './model-actions';
import { ModelTagSwitch } from './model-tag-switch';
export const schema = z.object({
  id: z.number(),
  header: z.string(),
  type: z.string(),
  status: z.string(),
  target: z.string(),
  limit: z.string(),
  reviewer: z.string(),
});

export function ModelTable({
  provider,
  data,
  pathnamePrefix,
}: {
  provider: Provider;
  data: ProviderModel[];
  pathnamePrefix: string;
}) {
  const page_models = useTranslations('page_models');
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({
      created: false,
    });
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([
    {
      id: 'api',
      value: 'completion',
    },
  ]);
  const currentApiFilter = React.useMemo(
    () =>
      columnFilters.find((item) => item.id === 'api')?.value as
        | ProviderModelApi
        | undefined,
    [columnFilters],
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
  const columns: ColumnDef<ProviderModel>[] = React.useMemo(() => {
    const cols: ColumnDef<ProviderModel>[] = [
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
        header: page_models('model.name'),
        cell: ({ row }) => {
          return (
            <div className="flex items-center gap-3">
              <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
                <MessageSquareCode className="size-4" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-left font-medium">
                  {row.original.model}
                </div>
                <div className="mt-1 flex flex-wrap gap-1">
                  {row.original.tags
                    ?.filter((tag) => tag !== '__autogen__')
                    .map((tag) => {
                      return (
                        <Badge
                          key={tag}
                          variant="secondary"
                          className="rounded-full font-mono text-[10px]"
                        >
                          {tag}
                        </Badge>
                      );
                    })}
                  </div>
              </div>
            </div>
          );
        },
      },
      {
        accessorKey: 'params',
        header: page_models('model.llm_params'),
        cell: ({ row }) => {
          return (
            <div className="flex flex-wrap items-center gap-2">
              <div className="bg-muted rounded-lg px-3 py-2">
                <div className="text-muted-foreground font-mono text-[10px] uppercase">
                  {page_models('model.param_context')}
                </div>
                <div className="mt-1 max-w-24 truncate font-mono text-xs">
                  {row.original.context_window || '-'}
                </div>
              </div>
              <div className="bg-muted rounded-lg px-3 py-2">
                <div className="text-muted-foreground font-mono text-[10px] uppercase">
                  {page_models('model.param_input')}
                </div>
                <div className="mt-1 max-w-24 truncate font-mono text-xs">
                  {row.original.max_input_tokens || '-'}
                </div>
              </div>
              <div className="bg-muted rounded-lg px-3 py-2">
                <div className="text-muted-foreground font-mono text-[10px] uppercase">
                  {page_models('model.param_output')}
                </div>
                <div className="mt-1 max-w-24 truncate font-mono text-xs">
                  {row.original.max_output_tokens || '-'}
                </div>
              </div>
            </div>
          );
        },
      },
      {
        accessorKey: 'agent',
        header: page_models('model.agent'),
        cell: ({ row }) => {
          return (
            <ModelTagSwitch
              model={row.original}
              provider={provider}
              tag="enable_for_agent"
            />
          );
        },
      },
      {
        accessorKey: 'collection',
        header: page_models('model.collection'),
        cell: ({ row }) => {
          return (
            <ModelTagSwitch
              model={row.original}
              provider={provider}
              tag="enable_for_collection"
            />
          );
        },
      },
      {
        accessorKey: 'api',
        header: page_models('model.api_type'),
        cell: ({ row }) => {
          let icon;
          switch (row.original.api) {
            case 'completion':
              icon = <MessageSquareCode />;
              break;
            case 'embedding':
              icon = <BetweenVerticalStart />;
              break;
            case 'rerank':
              icon = <ArrowUpDown />;
              break;
          }
          return (
            <Badge
              variant="outline"
              className="rounded-full bg-muted font-mono uppercase"
            >
              {icon} {row.original.api}
            </Badge>
          );
        },
      },
      {
        accessorKey: 'created',
        header: page_models('model.creation_time'),
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
                  <SquarePen /> {page_models('model.edit')}
                </DropdownMenuItem>
              </ModelActions>
              <DropdownMenuSeparator />
              <ModelActions
                action="delete"
                provider={provider}
                model={row.original}
              >
                <DropdownMenuItem variant="destructive">
                  <Trash /> {page_models('model.delete')}
                </DropdownMenuItem>
              </ModelActions>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [page_models, provider]);

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
    <div className="flex flex-col gap-5">
      <div className="border-border/70 bg-card grid gap-4 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <Button asChild variant="outline">
            <Link href={`${pathnamePrefix}/providers`}>
              <ArrowLeft />
              <span className="hidden sm:inline">
                {page_models('metadata.provider_title')}
              </span>
            </Link>
          </Button>
          <Select
            onValueChange={(value) => {
              setColumnFilters([
                {
                  id: 'api',
                  value,
                },
              ]);
            }}
            value={currentApiFilter}
          >
            <SelectTrigger className="w-full max-w-40">
              <SelectValue placeholder={page_models('model.api_type')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="completion">Completion</SelectItem>
              <SelectItem value="embedding">Embedding</SelectItem>
              <SelectItem value="rerank">Rerank</SelectItem>
            </SelectContent>
          </Select>
          <div className="relative min-w-0 flex-1 md:min-w-80">
            <Input
              className="bg-background/70 h-10 rounded-lg pl-9"
              placeholder={page_models('model.search_placeholder')}
              value={searchValue}
              onChange={(e) => setSearchValue(e.currentTarget.value)}
            />
            <MessageSquareCode className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <ModelActions action="add" provider={provider}>
            <Button>
              <Plus />
              <span className="hidden lg:inline">
                {page_models('model.add_model')}
              </span>
            </Button>
          </ModelActions>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Columns3 />
                <span className="hidden sm:inline">
                  {page_models('provider.columns')}
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
        <DataGrid
          table={table}
          idKey="model"
          className="rounded-lg border-border/70"
        />
      </div>
      <DataGridPagination table={table} />
    </div>
  );
}
