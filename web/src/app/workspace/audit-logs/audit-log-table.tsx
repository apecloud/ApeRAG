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

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { DateTimePicker24h } from '@/components/date-time-picker-24h';
import { cn, objectKeys, parsePageParams } from '@/lib/utils';
import { ChevronDown, Columns3 } from 'lucide-react';
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
                <span className="hover:text-primary cursor-pointer">
                  {row.original.api_name}
                </span>
              </AuditLogDetail>
              <div className="text-muted-foreground truncate pt-0.5 sm:w-sm md:w-md lg:w-lg">
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
              color = 'text-green-500';
              break;
            case 500:
              color = 'text-red-500';
              break;
            default:
          }
          return <div className={cn(color)}>{row.original.status_code}</div>;
        },
      },
      {
        accessorKey: 'duration_ms',
        header: page_audit_logs('duration'),
        cell: ({ row }) => {
          return row.original.duration_ms + 'ms';
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
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-row items-center gap-2">
          <Input
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
          <div className="flex flex-row items-center gap-0.5">
            <DateTimePicker24h
              className="w-48"
              date={query.startDate ? new Date(query.startDate) : undefined}
              onChange={(d) => {
                handleSearch({
                  startDate: d ? new Date(d).toISOString() : undefined,
                });
              }}
            />
            <span>-</span>
            <DateTimePicker24h
              className="w-48"
              date={query.endDate ? new Date(query.endDate) : undefined}
              onChange={(d) => {
                handleSearch({
                  endDate: d ? new Date(d).toISOString() : undefined,
                });
              }}
            />
          </div>
        </div>
        <div className="flex items-center gap-2">
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
