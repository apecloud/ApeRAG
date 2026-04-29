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

import type {
  ListAuditLogsParams as AuditApiListAuditLogsRequest,
  AuditLog,
} from '@/features/audit/types';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { DateTimePicker24h } from '@/components/date-time-picker-24h';
import { cn, objectKeys, parsePageParams } from '@/lib/utils';
import { ChevronDown, Columns3, Search } from 'lucide-react';
import { useFormatter, useTranslations } from 'next-intl';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { AuditLogDetail } from './audit-log-detail';

export function AuditLogTable({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  urlPrefix,
  data,
  pageCount,
}: {
  urlPrefix: string;
  data: AuditLog[];
  pageCount: number;
}) {
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const [apiNameValue, setApiNameValue] = React.useState<string>('');
  const page_audit_logs = useTranslations('page_audit_logs');

  const format = useFormatter();
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const query = React.useMemo(() => {
    return {
      ...parsePageParams({
        page: searchParams.get('page'),
        pageSize: searchParams.get('pageSize'),
      }),
      startDate: searchParams.get('startDate'),
      endDate: searchParams.get('endDate'),
      apiName: searchParams.get('apiName'),
      userId: searchParams.get('userId'),
    };
  }, [searchParams]);

  React.useEffect(() => {
    setApiNameValue(query.apiName || '');
  }, [query]);

  const handleSearch = React.useCallback(
    (params: AuditApiListAuditLogsRequest) => {
      const urlSearchParams = new URLSearchParams();
      const data = { ...query, ...params };
      objectKeys(data).forEach((key) => {
        const value = data[key];
        if (value !== null && value !== undefined) {
          urlSearchParams.set(key, String(value));
        }
      });
      router.push(`${pathname}?${urlSearchParams.toString()}`);
    },
    [query, router, pathname],
  );

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
            <>
              <AuditLogDetail auditLog={row.original}>
                <span className="hover:text-primary cursor-pointer font-medium">
                  {row.original.api_name}
                </span>
              </AuditLogDetail>
              <div className="text-muted-foreground sm:w-sm md:w-md lg:w-lg truncate pt-0.5 font-mono text-xs">
                {row.original.path}
              </div>
            </>
          );
        },
      },
      {
        accessorKey: 'status_code',
        header: page_audit_logs('status'),
        cell: ({ row }) => {
          let color;
          switch (row.original.status_code) {
            case 200:
              color = 'text-accent-ink';
              break;
            case 500:
              color = 'text-destructive';
              break;
            default:
          }
          return (
            <div className={cn('font-mono tabular-nums', color)}>
              {row.original.status_code}
            </div>
          );
        },
      },
      {
        accessorKey: 'duration_ms',
        header: page_audit_logs('duration'),
        cell: ({ row }) => {
          return (
            <span className="font-mono tabular-nums">
              {row.original.duration_ms}ms
            </span>
          );
        },
      },
      {
        accessorKey: 'start_time',
        header: page_audit_logs('start_time'),
        cell: ({ row }) =>
          row.original.start_time
            ? format.dateTime(row.original.start_time, 'medium')
            : '--',
      },
    ];
    return cols;
  }, [format, page_audit_logs]);

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
    <div className="flex flex-col gap-5">
      <div className="border-border/70 bg-card grid gap-4 rounded-xl border p-4 shadow-sm lg:grid-cols-[1fr_auto] lg:items-center">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
          <div className="relative min-w-0 flex-1 xl:min-w-80">
            <Search className="text-muted-foreground pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2" />
            <Input
              className="bg-background/70 h-10 rounded-lg pl-9"
              placeholder={page_audit_logs('search_placeholder')}
              value={apiNameValue}
              onChange={(e) => setApiNameValue(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSearch({
                    apiName: e.currentTarget.value,
                  });
                }
              }}
            />
          </div>
          <div className="grid gap-2 sm:grid-cols-[1fr_auto_1fr] sm:items-center xl:flex xl:flex-row">
            <DateTimePicker24h
              className="w-full sm:w-48"
              date={query.startDate ? new Date(query.startDate) : undefined}
              onChange={(d) => {
                handleSearch({
                  startDate: d ? new Date(d).toISOString() : undefined,
                });
              }}
            />
            <span className="text-muted-foreground hidden text-center sm:block">
              -
            </span>
            <DateTimePicker24h
              className="w-full sm:w-48"
              date={query.endDate ? new Date(query.endDate) : undefined}
              onChange={(d) => {
                handleSearch({
                  endDate: d ? new Date(d).toISOString() : undefined,
                });
              }}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Columns3 />
                <span className="hidden sm:inline">
                  {page_audit_logs('columns')}
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
      <div className="border-border/70 bg-card rounded-xl border p-2 shadow-sm sm:p-3">
        <DataGrid
          table={table}
          className="border-border/70 overflow-x-auto rounded-lg [&_table]:min-w-[760px]"
        />
      </div>
      <DataGridPagination table={table} />
    </div>
  );
}
