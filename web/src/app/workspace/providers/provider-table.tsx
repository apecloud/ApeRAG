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

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import type { Provider, ProviderModel } from '@/features/providers/types';
import { cn } from '@/lib/utils';
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  FolderCog,
  Globe,
  Plus,
  Search,
  SquarePen,
  Trash,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { ModelsDefaultConfiguration } from './models-default-configuration';
import { ProviderActions } from './provider-actions';
import { ProviderToggle } from './provider-toggle';

export const ProviderTable = ({
  data,
  models,
  urlPrefix,
}: {
  data: Provider[];
  models: ProviderModel[];
  urlPrefix: string;
}) => {
  const { user } = useAppContext();
  const page_models = useTranslations('page_models');
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
      id: 'name',
      desc: false,
    },
  ]);
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 20,
  });
  const [searchValue, setSearchValue] = React.useState<string>('');
  const modelCounts = React.useMemo(() => {
    const counts = new Map<string, number>();
    models.forEach((model) => {
      if (model.provider_name) {
        counts.set(model.provider_name, (counts.get(model.provider_name) || 0) + 1);
      }
    });
    return counts;
  }, [models]);

  const columns: ColumnDef<Provider>[] = React.useMemo(() => {
    const cols: ColumnDef<Provider>[] = [
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
        accessorKey: 'label',
        header: page_models('provider.name'),
        cell: ({ row }) => {
          return (
            <Link
              className="group flex items-center gap-3"
              href={`${urlPrefix}/providers/${row.original.name}/models`}
            >
              <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
                <FolderCog className="size-4" />
              </div>
              <div className="min-w-0">
                <div className="group-hover:text-primary truncate font-medium transition-colors">
                  {row.original.label || row.original.name}
                </div>
                <div className="text-muted-foreground font-mono text-[11px]">
                  {row.original.name}
                </div>
              </div>
            </Link>
          );
        },
      },
      {
        accessorKey: 'base_url',
        header: page_models('provider.base_url'),
        cell: ({ row }) => (
          <div className="text-muted-foreground max-w-[260px] truncate font-mono text-xs">
            {row.original.base_url || '-'}
          </div>
        ),
      },
      {
        accessorKey: 'name',
        header: page_models('provider.models_count'),
        cell: ({ row }) => {
          return (
            <Badge variant="secondary" className="font-mono tabular-nums">
              {modelCounts.get(row.original.name) || 0}
            </Badge>
          );
        },
      },
      {
        accessorKey: 'user_id',
        header: page_models('provider.scope'),
        cell: ({ row }) => {
          const text =
            row.original.user_id === 'public'
              ? page_models('provider.public')
              : page_models('provider.private');
          const isPublic = row.original.user_id === 'public';
          return (
            <Badge
              variant="outline"
              className={cn(
                'rounded-full',
                isPublic
                  ? 'border-primary/20 bg-accent-soft text-accent-ink'
                  : 'bg-muted text-muted-foreground',
              )}
            >
              {text}
            </Badge>
          );
        },
      },
      {
        accessorKey: 'enabled',
        header: page_models('provider.enabled'),
        cell: ({ row }) => {
          return <ProviderToggle provider={row.original} />;
        },
      },
      {
        accessorKey: 'created',
        header: page_models('provider.creation_time'),
        cell: ({ row }) => {
          return row.original.created ? (
            <FormatDate datetime={new Date(row.original.created)} />
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
            <DropdownMenuContent align="end" className="w-32">
              <DropdownMenuItem asChild>
                <Link
                  href={`${urlPrefix}/providers/${row.original.name}/models`}
                >
                  <FolderCog /> {page_models('metadata.model_title')}
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <ProviderActions action="edit" provider={row.original}>
                <DropdownMenuItem>
                  <SquarePen /> {page_models('provider.edit')}
                </DropdownMenuItem>
              </ProviderActions>
              {row.original.user_id !== 'public' && user?.role === 'admin' && (
                <ProviderActions action="publish" provider={row.original}>
                  <DropdownMenuItem>
                    <Globe /> {page_models('provider.publish')}
                  </DropdownMenuItem>
                </ProviderActions>
              )}
              <ProviderActions action="delete" provider={row.original}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> {page_models('provider.delete')}
                </DropdownMenuItem>
              </ProviderActions>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [modelCounts, page_models, urlPrefix, user?.role]);

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
    getRowId: (row) => String(row.name),
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
        <div className="relative max-w-xl">
          <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          <Input
            className="bg-background/70 h-10 rounded-lg pl-9"
            placeholder={page_models('provider.search_placeholder')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          {user?.role === 'admin' && <ModelsDefaultConfiguration />}

          <ProviderActions action="add">
            <Button>
              <Plus />
              <span className="hidden lg:inline">
                {page_models('provider.add_provider')}
              </span>
            </Button>
          </ProviderActions>

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
        <DataGrid table={table} idKey="name" className="rounded-lg border-border/70" />
      </div>
      <DataGridPagination table={table} />
    </div>
  );
};
