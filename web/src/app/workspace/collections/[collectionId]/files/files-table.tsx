'use client';

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
import { Input } from '@/components/ui/input';
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
import { defaultStyles, FileIcon } from 'react-file-icon';

import { Collection, Document } from '@/api';

import { DataGrid, DataGridPagination } from '@/components/data-grid';
import { FormatDate } from '@/components/format-date';
import { cn, objectKeys } from '@/lib/utils';
import _ from 'lodash';
import {
  ChevronDown,
  Columns3,
  EllipsisVertical,
  FolderSync,
  MonitorUp,
  Trash,
} from 'lucide-react';

import { FileIndexType, getDocumentStatusColor } from '@/lib/document';
import { FileDelete } from './file-delete';
import { FileIndexStatus } from './file-index-status';
import { FileReBuildIndex } from './file-rebuild-index';

export function FilesTable({
  collection,
  data,
}: {
  collection: Collection;
  data: Document[];
}) {
  const [rowSelection, setRowSelection] = React.useState({});
  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({});
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    [],
  );
  const [sorting, setSorting] = React.useState<SortingState>([]);
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 20,
  });
  const [searchValue, setSearchValue] = React.useState<string>('');

  const columns: ColumnDef<Document>[] = React.useMemo(() => {
    const indexCols: ColumnDef<Document>[] = objectKeys(FileIndexType).map(
      (key) => {
        const accessorKey = key.toLowerCase() + '_index_status';
        return {
          accessorKey,
          header: FileIndexType[key].title,
          cell: ({ row }) => (
            <FileIndexStatus
              document={row.original}
              accessorKey={accessorKey}
            />
          ),
        };
      },
    );

    const cols: ColumnDef<Document>[] = [
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
        accessorKey: 'name',
        header: 'Name',
        cell: ({ row }) => {
          const extension =
            row.original.name?.split('.').pop()?.toLowerCase() ||
            ('unknow' as keyof typeof defaultStyles);
          const iconProps = _.get(defaultStyles, extension);
          const icon = (
            <FileIcon
              color="var(--muted-foreground)"
              extension={extension}
              {...iconProps}
            />
          );
          return (
            <div className="flex flex-row items-center gap-2">
              <div className="h-8 w-6">{icon}</div>
              <div>
                <div
                  className={cn(
                    'max-w-60 truncate',
                    getDocumentStatusColor(row.original.status),
                  )}
                >
                  {row.original.name}
                </div>
                <div className="text-muted-foreground text-sm">
                  {(Number(row.original.size || 0) / 1000).toFixed(2)} KB
                </div>
              </div>
            </div>
          );
        },
      },
      ...indexCols,
      {
        accessorKey: 'updated',
        header: 'Last Updated',
        cell: ({ row }) => {
          return row.original.updated ? (
            <FormatDate datetime={new Date(row.original.updated)} />
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
            <DropdownMenuContent align="end" className="w-42">
              <FileReBuildIndex collection={collection} file={row.original}>
                <DropdownMenuItem>
                  <FolderSync /> Rebuild File Index
                </DropdownMenuItem>
              </FileReBuildIndex>
              <DropdownMenuSeparator />
              <FileDelete collection={collection} file={row.original}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> Delete
                </DropdownMenuItem>
              </FileDelete>
            </DropdownMenuContent>
          </DropdownMenu>
        ),
      },
    ];
    return cols;
  }, [collection]);

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
          <Button>
            <MonitorUp />
            <span className="hidden lg:inline">Upload</span>
          </Button>
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
        </div>
      </div>
      <DataGrid table={table} />
      <DataGridPagination table={table} />
    </div>
  );
}
