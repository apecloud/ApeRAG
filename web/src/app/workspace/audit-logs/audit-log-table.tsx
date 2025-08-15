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
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';

import { AuditApiListAuditLogsRequest, AuditLog } from '@/api';
import { TableList, TableListPagination } from '@/components/table-list';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

import { DateTimePicker24h } from '@/components/date-time-picker-24h';
import _ from 'lodash';
import { ChevronDown, Columns3, Search } from 'lucide-react';
import { useFormatter } from 'next-intl';
import { usePathname, useRouter } from 'next/navigation';

export function AuditLogTable({
  data,
  searchParams: initSearchParams,
}: {
  data: AuditLog[];
  searchParams: AuditApiListAuditLogsRequest;
}) {
  const [rowSelection, setRowSelection] = React.useState({});
  const [query, setQuery] =
    React.useState<AuditApiListAuditLogsRequest>(initSearchParams);
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const format = useFormatter();
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 20,
  });
  const router = useRouter();
  const pathname = usePathname();

  const columns: ColumnDef<AuditLog>[] = React.useMemo(() => {
    const cols: ColumnDef<AuditLog>[] = [
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
        accessorKey: 'api_name',
        header: 'API',
        cell: ({ row }) => {
          return (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="inline-flex flex-col gap-1">
                  <div>{row.original.api_name}</div>
                  <div className="text-muted-foreground">
                    {row.original.path}
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent side="left">
                <p>Resource ID: {row.original.resource_id}</p>
                <p>Resource Type: {row.original.resource_type}</p>
              </TooltipContent>
            </Tooltip>
          );
        },
      },
      {
        accessorKey: 'status_code',
        header: 'Status',
      },
      {
        accessorKey: 'end_time',
        header: 'Duration',
        cell: ({ row }) => {
          if (row.original.start_time && row.original.end_time) {
            return `${row.original.end_time - row.original.start_time}ms`;
          }
        },
      },
      {
        accessorKey: 'start_time',
        header: 'Start Time',
        cell: ({ row }) =>
          row.original.start_time
            ? format.dateTime(row.original.start_time, 'medium')
            : '--',
      },
    ];
    return cols;
  }, [format]);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination,
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

  const handleSearch = React.useCallback(() => {
    const sp = new URLSearchParams();
    _.forEach(query, (value, key) => {
      sp.set(key, String(value));
    });
    router.push(`${pathname}?${sp.toString()}`);
  }, [pathname, query, router]);

  React.useEffect(() => {
    setQuery(initSearchParams);
  }, [initSearchParams]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-row items-center gap-2">
          <Input
            placeholder="Search api name"
            value={query.apiName}
            onChange={(e) => {
              setQuery({
                ...query,
                apiName: e.currentTarget.value,
              });
            }}
          />
          <div className="flex flex-row items-center gap-0.5">
            <DateTimePicker24h
              className="w-48"
              date={query.startDate ? new Date(query.startDate) : undefined}
              onChange={(d) => {
                setQuery({
                  ...query,
                  startDate: d ? new Date(d).toISOString() : undefined,
                });
              }}
            />
            <span>-</span>
            <DateTimePicker24h
              className="w-48"
              date={query.endDate ? new Date(query.endDate) : undefined}
              onChange={(d) => {
                setQuery({
                  ...query,
                  endDate: d ? new Date(d).toISOString() : undefined,
                });
              }}
            />
          </div>

          <Button onClick={handleSearch}>
            <Search />
            <span className="hidden lg:inline">Search</span>
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Columns3 />
                <span className="hidden lg:inline">Columns</span>
                <span className="lg:hidden">Columns</span>
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
      <TableList table={table} />
      <TableListPagination table={table} />
    </div>
  );
}
