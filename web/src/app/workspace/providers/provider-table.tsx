'use client';

import { LlmProvider, LlmProviderModel } from '@/api';

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
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  Plus,
  SquarePen,
  Trash,
} from 'lucide-react';
import Link from 'next/link';
import { ModelsDefaultConfiguration } from './models-default-configuration';
import { ProviderActions } from './provider-actions';
import { ProviderToggle } from './provider-toggle';

export const ProviderTable = ({
  data,
  models,
  urlPrefix,
}: {
  data: LlmProvider[];
  models: LlmProviderModel[];
  urlPrefix: string;
}) => {
  const { user } = useAppContext();
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

  const columns: ColumnDef<LlmProvider>[] = React.useMemo(() => {
    const cols: ColumnDef<LlmProvider>[] = [
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
        header: 'Name',
        cell: ({ row }) => {
          return (
            <Link
              className="hover:text-primary"
              href={`${urlPrefix}/providers/${row.original.name}/models`}
            >
              {row.original.label || row.original.name}
            </Link>
          );
        },
      },
      {
        accessorKey: 'base_url',
        header: 'Base Url',
      },
      {
        accessorKey: 'name',
        header: 'Models',
        cell: ({ row }) => {
          const providerModels = models.filter(
            (m) => m.provider_name === row.original.name,
          );
          return <div>{providerModels.length}</div>;
        },
      },
      {
        accessorKey: 'user_id',
        header: 'Scope',
        cell: ({ row }) => {
          const text = row.original.user_id === 'public' ? 'Public' : 'Private';
          const variant =
            row.original.user_id === 'public' ? 'default' : 'destructive';
          return <Badge variant={variant}>{text}</Badge>;
        },
      },
      {
        accessorKey: 'enabled',
        header: 'Enabled',
        cell: ({ row }) => {
          return <ProviderToggle provider={row.original} />;
        },
      },
      {
        accessorKey: 'created',
        header: 'Creation time',
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
              <ProviderActions action="edit" provider={row.original}>
                <DropdownMenuItem>
                  <SquarePen /> Edit
                </DropdownMenuItem>
              </ProviderActions>
              <DropdownMenuSeparator />
              <ProviderActions action="delete" provider={row.original}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> Delete
                </DropdownMenuItem>
              </ProviderActions>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [models, urlPrefix]);

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

          {user?.role === 'admin' && <ModelsDefaultConfiguration />}

          <ProviderActions action="add">
            <Button>
              <Plus />
              <span className="hidden lg:inline">Add Provider</span>
            </Button>
          </ProviderActions>
        </div>
      </div>
      <DataGrid table={table} />
      <DataGridPagination table={table} />
    </div>
  );
};
