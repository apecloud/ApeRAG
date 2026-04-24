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

import type { User } from '@/features/identity/types';

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
  CircleMinus,
  Columns3,
  EllipsisVertical,
  Key,
  ScrollText,
  Search,
  Trash,
  UserRound,
} from 'lucide-react';
import { useFormatter, useTranslations } from 'next-intl';
import Link from 'next/link';
import { FaGithub, FaGoogle } from 'react-icons/fa6';
import { UserQuotaAction } from './user-quota-action';

export function UsersDataTable({ data }: { data: User[] }) {
  const admin_users = useTranslations('admin_users');
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
        header: admin_users('user_name'),
        cell: ({ row }) => {
          return (
            <div className="flex items-center gap-3 text-left">
              <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
                <UserRound className="size-4" />
              </div>
              <div className="min-w-0">
                <div className="truncate font-medium">
                  {row.original.username}
                  {user?.id === row.original.id && (
                    <Badge
                      variant="outline"
                      className="bg-accent-soft text-accent-ink ml-2 rounded-full"
                    >
                      {admin_users('self_badge')}
                    </Badge>
                  )}
                </div>
                <div className="text-muted-foreground truncate">
                  {row.original.email}
                </div>
              </div>
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
        header: admin_users('user_role'),
        cell: ({ row }) => {
          return (
            <Badge
              className={cn(
                'w-18 rounded-full',
                row.original.role === 'admin'
                  ? 'border-primary/20 bg-accent-soft text-accent-ink'
                  : 'bg-muted text-muted-foreground',
              )}
              variant="outline"
            >
              {row.original.role}
            </Badge>
          );
        },
      },
      {
        accessorKey: 'is_active',
        header: admin_users('user_status'),
        cell: ({ row }) => {
          const isActive = row.original.is_active;
          return (
            <Badge
              variant="outline"
              className={cn(
                'gap-1 rounded-full',
                isActive
                  ? 'border-primary/20 bg-accent-soft text-accent-ink'
                  : 'bg-muted text-muted-foreground',
              )}
            >
              {isActive ? (
                <Check className="size-3" />
              ) : (
                <CircleMinus className="size-3" />
              )}
              {admin_users(isActive ? 'status_active' : 'status_inactive')}
            </Badge>
          );
        },
      },
      {
        accessorKey: 'registration_source',
        header: admin_users('user_source'),
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
        header: admin_users('user_creation_time'),
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
                  <BatteryMedium />
                  {admin_users('user_quotas')}
                </DropdownMenuItem>
              </UserQuotaAction>
              <DropdownMenuItem asChild>
                <Link href={`/admin/audit-logs?userId=${row.original.id}`}>
                  <ScrollText /> {admin_users('user_logs')}
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" disabled>
                <Trash /> {admin_users('user_delete')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [admin_users, format, user?.id]);

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
        <div className="relative max-w-xl">
          <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2" />
          <Input
            className="bg-background/70 h-10 rounded-lg pl-9"
            placeholder={admin_users('search')}
            value={searchValue}
            onChange={(e) => setSearchValue(e.currentTarget.value)}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Columns3 />
                <span className="hidden sm:inline">
                  {admin_users('columns')}
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
