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

import { User } from '@/api';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { useAppContext } from '@/components/providers/app-provider';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import {
  BatteryMedium,
  Check,
  ChevronDown,
  Columns3,
  EllipsisVertical,
  Key,
  ScrollText,
  Trash,
} from 'lucide-react';
import { useFormatter } from 'next-intl';
import Link from 'next/link';
import { FaGithub, FaGoogle } from 'react-icons/fa6';
import { UserQuotaAction } from './user-quota-action';

export function UsersDataTable({ data }: { data: User[] }) {
  const { user } = useAppContext();
  const [rowSelection, setRowSelection] = React.useState({});
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
  const [searchValue, setSearchValue] = React.useState<string>('');

  const columns: ColumnDef<User>[] = React.useMemo(() => {
    const cols: ColumnDef<User>[] = [
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
        accessorKey: 'username',
        header: 'User',
        cell: ({ row }) => {
          return (
            <div className="text-left">
              <div>
                {row.original.username}{' '}
                {user?.id === row.original.id && (
                  <Badge variant="destructive">Me</Badge>
                )}
              </div>
              <div className="text-muted-foreground">{row.original.email}</div>
            </div>
          );
        },
      },
      {
        accessorKey: 'id',
        header: 'ID',
      },
      {
        accessorKey: 'role',
        header: 'Role',
        cell: ({ row }) => {
          return (
            <Badge
              className="w-18"
              variant={row.original.role === 'admin' ? 'default' : 'secondary'}
            >
              {row.original.role}
            </Badge>
          );
        },
      },
      {
        accessorKey: 'is_active',
        header: 'Status',
        cell: ({ row }) => {
          return (
            <Check
              className={cn(
                'size-4',
                row.original.is_active ? 'text-green-500' : 'text-red-500',
              )}
            />
          );
        },
      },
      {
        accessorKey: 'registration_source',
        header: 'Source',
        cell: ({ row }) => {
          let icon;
          switch (row.original.registration_source) {
            case 'google':
              icon = <FaGoogle className="size-4" />;
              break;
            case 'github':
              icon = <FaGithub className="size-4" />;
              break;
            default:
              icon = <Key className="size-4" />;
          }
          return (
            <Tooltip>
              <TooltipTrigger>{icon}</TooltipTrigger>
              <TooltipContent>
                {row.original.registration_source}
              </TooltipContent>
            </Tooltip>
          );
        },
      },
      {
        accessorKey: 'date_joined',
        header: 'Creation time',
        cell: ({ row }) =>
          row.original.date_joined
            ? format.dateTime(new Date(row.original.date_joined), 'medium')
            : '--',
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
              <UserQuotaAction user={row.original}>
                <DropdownMenuItem>
                  <BatteryMedium /> Quotas
                </DropdownMenuItem>
              </UserQuotaAction>
              <DropdownMenuItem asChild>
                <Link href={`/admin/audit-logs?userId=${row.original.id}`}>
                  <ScrollText /> Audit logs
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" disabled>
                <Trash /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [format, user?.id]);

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
      <DataGrid table={table} />
      <DataGridPagination table={table} />
    </div>
  );
}
