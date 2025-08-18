import React, { useState } from 'react';
import { Table, Input, Select, Space, Typography, TableProps } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useIntl } from 'umi';

interface PaginatedTableProps<T = any> extends Omit<TableProps<T>, 'pagination'> {
  // Pagination props
  total?: number;
  current?: number;
  pageSize?: number;
  showSizeChanger?: boolean;
  pageSizeOptions?: string[];
  onPageChange?: (page: number, pageSize: number) => void;
  
  // Search props
  searchPlaceholder?: string;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  showSearch?: boolean;
  
  // Sort props
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  sortOptions?: Array<{ label: string; value: string }>;
  onSortChange?: (sortBy: string, sortOrder: 'asc' | 'desc') => void;
  showSort?: boolean;
  
  // Custom header content
  headerContent?: React.ReactNode;
}

export const PaginatedTable = <T extends Record<string, any>>({
  // Pagination props
  total = 0,
  current = 1,
  pageSize = 10,
  showSizeChanger = true,
  pageSizeOptions = ['10', '20', '50', '100'],
  onPageChange,
  
  // Search props
  searchPlaceholder,
  searchValue,
  onSearchChange,
  showSearch = true,
  
  // Sort props
  sortBy,
  sortOrder = 'desc',
  sortOptions = [],
  onSortChange,
  showSort = false,
  
  // Custom header content
  headerContent,
  
  // Table props
  ...tableProps
}: PaginatedTableProps<T>) => {
  const { formatMessage } = useIntl();
  
  const handleSearch = (value: string) => {
    // Trim the search value to handle empty spaces
    const trimmedValue = value.trim();
    onSearchChange?.(trimmedValue || undefined);
  };
  
  const handleSortByChange = (value: string) => {
    onSortChange?.(value, sortOrder);
  };
  
  const handleSortOrderChange = (value: 'asc' | 'desc') => {
    onSortChange?.(sortBy || '', value);
  };
  
  const handlePageChange = (page: number, size: number) => {
    onPageChange?.(page, size);
  };
  
  return (
    <>
      {/* Header controls */}
      {(showSearch || showSort || headerContent) && (
        <Space
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginBottom: 16,
          }}
        >
          <Space>
            {/* Search input */}
            {showSearch && (
              <Input
                placeholder={searchPlaceholder || formatMessage({ id: 'action.search' })}
                prefix={
                  <Typography.Text disabled>
                    <SearchOutlined />
                  </Typography.Text>
                }
                value={searchValue}
                onChange={(e) => handleSearch(e.target.value)}
                allowClear
                style={{ width: 250 }}
              />
            )}
            
            {/* Sort controls */}
            {showSort && sortOptions.length > 0 && (
              <Space>
                <Typography.Text type="secondary">
                  {formatMessage({ id: 'common.sort.by' })}:
                </Typography.Text>
                <Select
                  value={sortBy}
                  onChange={handleSortByChange}
                  options={sortOptions}
                  style={{ width: 120 }}
                  placeholder={formatMessage({ id: 'common.sort.field' })}
                />
                <Select
                  value={sortOrder}
                  onChange={handleSortOrderChange}
                  options={[
                    { label: formatMessage({ id: 'common.sort.desc' }), value: 'desc' },
                    { label: formatMessage({ id: 'common.sort.asc' }), value: 'asc' },
                  ]}
                  style={{ width: 80 }}
                />
              </Space>
            )}
          </Space>
          
          {/* Custom header content */}
          {headerContent && (
            <Space>
              {headerContent}
            </Space>
          )}
        </Space>
      )}
      
      {/* Table with pagination */}
      <Table
        {...tableProps}
        pagination={{
          current,
          pageSize,
          total,
          showSizeChanger,
          pageSizeOptions,
          showQuickJumper: true,
          showTotal: (total, range) =>
            formatMessage(
              { id: 'common.pagination.total' },
              {
                start: range[0],
                end: range[1],
                total,
              }
            ),
          onChange: handlePageChange,
        }}
      />
    </>
  );
};

export default PaginatedTable;
