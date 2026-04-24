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

import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Input } from '@/components/ui/input';
import type { SearchResult } from '@/features/retrieval/types';
import _ from 'lodash';
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  FlaskConical,
  Search,
  Trash,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import { filterVisibleSearchItems } from '../../feature-visibility';
import { SearchDelete } from './search-delete';
import { SearchResultDrawer } from './search-result-drawer';
import { SearchTest } from './search-test';

const SearchParamBadge = ({
  label,
  value,
}: {
  label: string;
  value?: number | string | null;
}) => {
  if (value === null || value === undefined || value === '') return null;
  return (
    <Badge variant="secondary" className="rounded-sm font-mono text-[10px]">
      {label} {value}
    </Badge>
  );
};

export const SearchTable = ({ data }: { data: SearchResult[] }) => {
  const { collection } = useCollectionContext();
  const page_search = useTranslations('page_search');
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
  ]);
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 20,
  });
  const [searchValue, setSearchValue] = React.useState<string>('');

  const columns: ColumnDef<SearchResult>[] = React.useMemo(() => {
    const indexCols: ColumnDef<SearchResult>[] = [];

    if (collection.config?.enable_vector) {
      indexCols.push({
        accessorKey: 'vector_search',
        header: page_search('vector_search'),
        cell: ({ row }) => {
          return (
            <div className="flex flex-wrap gap-1.5">
              <SearchParamBadge
                label="top"
                value={row.original.vector_search?.topk}
              />
              <SearchParamBadge
                label="sim"
                value={row.original.vector_search?.similarity}
              />
            </div>
          );
        },
      });
    }

    if (collection.config?.enable_fulltext) {
      indexCols.push({
        accessorKey: 'fulltext_search',
        header: page_search('fulltext_search'),
        cell: ({ row }) => {
          return (
            <div className="flex flex-wrap gap-1.5">
              <SearchParamBadge
                label="top"
                value={row.original.fulltext_search?.topk}
              />
              {row.original.fulltext_search?.keywords?.map((keyword) => (
                <Badge
                  key={keyword}
                  variant="outline"
                  className="rounded-sm text-[10px]"
                >
                  {keyword}
                </Badge>
              ))}
            </div>
          );
        },
      });
    }

    if (collection.config?.enable_knowledge_graph) {
      indexCols.push({
        accessorKey: 'graph_search',
        header: page_search('graph_search'),
        cell: ({ row }) => {
          return (
            <div className="flex flex-wrap gap-1.5">
              <SearchParamBadge
                label="top"
                value={row.original.graph_search?.topk}
              />
            </div>
          );
        },
      });
    }

    const cols: ColumnDef<SearchResult>[] = [
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
        accessorKey: 'query',
        header: page_search('questions'),
        cell: ({ row }) => {
          const visibleItems = filterVisibleSearchItems(
            row.original.items ?? undefined,
          );
          return (
            <div>
              <SearchResultDrawer result={row.original}>
                <div
                  data-result={!_.isEmpty(visibleItems)}
                  className="data-[result=true]:hover:text-primary max-w-md truncate text-sm font-medium transition-colors data-[result=true]:cursor-pointer"
                >
                  {row.original.query}
                </div>
              </SearchResultDrawer>
              <div className="text-muted-foreground mt-1 flex flex-row items-center gap-4 text-xs">
                {visibleItems.length} results
              </div>
            </div>
          );
        },
      },
      ...indexCols,
      {
        accessorKey: 'created',
        header: page_search('creation_time'),
        cell: ({ row }) => {
          return row.original.created ? (
            <span className="text-muted-foreground text-xs">
              <FormatDate datetime={new Date(row.original.created)} />
            </span>
          ) : undefined;
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
              <SearchDelete searchResult={row.original}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> {page_search('delete')}
                </DropdownMenuItem>
              </SearchDelete>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [
    collection.config?.enable_fulltext,
    collection.config?.enable_knowledge_graph,
    collection.config?.enable_vector,
    page_search,
  ]);

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
      <div className="border-border/70 bg-card grid gap-3 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="relative max-w-xl">
          <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          <Input
            className="bg-background/70 h-10 rounded-lg pl-9"
            placeholder={page_search('search')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <SearchTest>
            <Button>
              <FlaskConical />{' '}
              <span className="hidden sm:inline">{page_search('test')}</span>
            </Button>
          </SearchTest>
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
      <DataGrid table={table} className="border-border/70 bg-card shadow-sm" />
      <DataGridPagination table={table} />
    </div>
  );
};
