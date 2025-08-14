import { AuditApi } from '@/api/apis/audit-api';
import type { AuditLog } from '@/api/models';
import { CopyOutlined, EyeOutlined, SearchOutlined } from '@ant-design/icons';
import {
  Button,
  Card,
  DatePicker,
  Divider,
  Drawer,
  Form,
  Input,
  message,
  Space,
  Table,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import React, { useEffect, useState } from 'react';
import { useIntl } from 'umi';

const { RangePicker } = DatePicker;
const { Text, Title } = Typography;

const AuditLogsPage: React.FC = () => {
  const intl = useIntl();
  const { token } = theme.useToken();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AuditLog[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<AuditLog | null>(null);
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 20,
  });

  // Format duration
  // const formatDuration = (ms?: number): string => {
  //   if (!ms) return '-';
  //   if (ms < 1000) return `${ms.toFixed(0)}ms`;
  //   return `${(ms / 1000).toFixed(2)}s`;
  // };

  // Get status display
  const getStatusDisplay = (
    statusCode?: number,
  ): { text: string; color: string } => {
    if (!statusCode)
      return {
        text: intl.formatMessage({
          id: 'common.status.unknown',
          defaultMessage: 'Unknown',
        }),
        color: 'default',
      };
    if (statusCode >= 200 && statusCode < 300)
      return {
        text: intl.formatMessage({
          id: 'common.status.success',
          defaultMessage: 'Success',
        }),
        color: 'success',
      };
    return {
      text: intl.formatMessage({
        id: 'common.status.failed',
        defaultMessage: 'Failed',
      }),
      color: 'error',
    };
  };

  // Fetch audit logs
  const fetchData = async (params?: any) => {
    setLoading(true);
    try {
      const api = new AuditApi();
      const response = await api.listAuditLogs(params || {});
      setData(response.data.items || []);
    } catch (error) {
      console.error('Failed to fetch audit logs:', error);
      message.error(
        intl.formatMessage({
          id: 'audit.logs.fetchError',
          defaultMessage: 'Failed to fetch audit logs',
        }),
      );
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    // 默认展示最近一天的数据
    const endTime = dayjs();
    const startTime = endTime.subtract(1, 'day');
    fetchData({
      limit: 100,
      startDate: startTime.toISOString(),
      endDate: endTime.toISOString(),
    });

    // 设置表单默认值
    form.setFieldsValue({
      dateRange: [startTime, endTime],
    });
  }, []);

  // Handle search
  const handleSearch = () => {
    const values = form.getFieldsValue();
    const params: any = { ...values, limit: 100 };

    // Handle date range
    if (values.dateRange) {
      params.startDate = values.dateRange[0]?.toISOString();
      params.endDate = values.dateRange[1]?.toISOString();
      delete params.dateRange;
    }

    fetchData(params);
  };

  // Handle pagination change
  const handlePaginationChange = (page: number, pageSize?: number) => {
    setPagination({
      current: page,
      pageSize: pageSize || pagination.pageSize,
    });
  };

  // Handle view details
  const handleViewDetails = async (record: AuditLog) => {
    try {
      const api = new AuditApi();
      const log = await api.getAuditLog({ auditId: record.id! });
      console.log('Audit log details:', log.data);
      console.log('Request data:', log.data.request_data);
      console.log('Response data:', log.data.response_data);
      setSelectedRecord(log.data);
      setDetailDrawerVisible(true);
    } catch (error) {
      console.error('Failed to get audit details:', error);
      message.error(
        intl.formatMessage({
          id: 'audit.logs.detailError',
          defaultMessage: 'Failed to get audit details',
        }),
      );
    }
  };

  // Handle copy to clipboard
  const handleCopy = async (text: string, type: 'request' | 'response') => {
    try {
      await navigator.clipboard.writeText(text);
      message.success(
        intl.formatMessage(
          {
            id: 'audit.logs.copySuccess',
            defaultMessage: '{type} data copied to clipboard',
          },
          { type: type === 'request' ? 'Request' : 'Response' },
        ),
      );
    } catch (error) {
      message.error(
        intl.formatMessage({
          id: 'audit.logs.copyError',
          defaultMessage: 'Failed to copy to clipboard',
        }),
      );
    }
  };

  // Format JSON data for display
  const formatJsonData = (data: any): string => {
    if (!data) return '';
    try {
      if (typeof data === 'string') {
        return JSON.stringify(JSON.parse(data), null, 2);
      }
      return JSON.stringify(data, null, 2);
    } catch {
      return typeof data === 'string' ? data : JSON.stringify(data);
    }
  };

  // Table columns
  const columns: ColumnsType<AuditLog> = [
    {
      title: intl.formatMessage({
        id: 'audit.logs.username',
        defaultMessage: 'Username',
      }),
      dataIndex: 'username',
      key: 'username',
      width: 180,
      render: (text?: string) => <Text strong>{text || '-'}</Text>,
    },
    {
      title: intl.formatMessage({
        id: 'audit.logs.apiName',
        defaultMessage: 'API Name',
      }),
      dataIndex: 'api_name',
      key: 'api_name',
      width: 200,
      render: (text?: string) => (
        <Tooltip title={text}>
          <Text code style={{ fontSize: '12px' }}>
            {text && text.length > 30
              ? `${text.substring(0, 30)}...`
              : text || '-'}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: intl.formatMessage({
        id: 'audit.logs.resourceType',
        defaultMessage: 'Resource Type',
      }),
      dataIndex: 'resource_type',
      key: 'resource_type',
      width: 120,
      render: (type?: string) => {
        return type ? (
          <Tag color="blue" style={{ minWidth: 80, textAlign: 'center' }}>
            {type}
          </Tag>
        ) : (
          '-'
        );
      },
    },
    {
      title: intl.formatMessage({
        id: 'audit.logs.resourceId',
        defaultMessage: 'Resource ID',
      }),
      dataIndex: 'resource_id',
      key: 'resource_id',
      width: 240,
      render: (id?: string) =>
        id ? (
          <Text code style={{ fontSize: '11px' }}>
            {id}
          </Text>
        ) : (
          '-'
        ),
    },
    {
      title: intl.formatMessage({
        id: 'audit.logs.status',
        defaultMessage: 'Status',
      }),
      dataIndex: 'status_code',
      key: 'status_code',
      width: 100,
      align: 'center' as const,
      render: (code?: number) => {
        const status = getStatusDisplay(code);
        return (
          <Tag
            color={status.color}
            style={{ minWidth: 50, textAlign: 'center' }}
          >
            {status.text}
          </Tag>
        );
      },
    },
    {
      title: intl.formatMessage({
        id: 'audit.logs.startTime',
        defaultMessage: 'Start Time',
      }),
      dataIndex: 'start_time',
      key: 'start_time',
      width: 180,
      render: (time?: number) =>
        time ? (
          <Text style={{ fontSize: '12px' }}>
            {dayjs(time).format('YYYY-MM-DD HH:mm:ss.SSS')}
          </Text>
        ) : (
          '-'
        ),
    },
    {
      title: intl.formatMessage({
        id: 'audit.logs.endTime',
        defaultMessage: 'End Time',
      }),
      dataIndex: 'end_time',
      key: 'end_time',
      width: 180,
      render: (time?: number) =>
        time ? (
          <Text style={{ fontSize: '12px' }}>
            {dayjs(time).format('YYYY-MM-DD HH:mm:ss.SSS')}
          </Text>
        ) : (
          '-'
        ),
    },
    {
      title: intl.formatMessage({
        id: 'common.actions',
        defaultMessage: 'Actions',
      }),
      key: 'actions',
      width: 80,
      align: 'center' as const,
      render: (_, record) => (
        <Button
          type="link"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetails(record)}
        >
          {intl.formatMessage({
            id: 'common.detail',
            defaultMessage: 'Detail',
          })}
        </Button>
      ),
    },
  ];

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px' }}>
      <Card>
        <div style={{ marginBottom: '24px' }}>
          <Title level={4} style={{ margin: 0 }}>
            {intl.formatMessage({
              id: 'audit.logs.title',
              defaultMessage: 'Audit Logs',
            })}
          </Title>
          <Text type="secondary">
            {intl.formatMessage({
              id: 'audit.logs.description',
              defaultMessage:
                'View detailed audit records of system operations',
            })}
          </Text>
        </div>

        <Form
          form={form}
          layout="inline"
          onFinish={handleSearch}
          style={{ marginBottom: '24px' }}
        >
          <Space wrap style={{ width: '100%' }}>
            <Form.Item
              name="apiName"
              label={intl.formatMessage({
                id: 'audit.logs.apiName',
                defaultMessage: 'API Name',
              })}
              style={{ marginBottom: 0 }}
            >
              <Input
                placeholder={intl.formatMessage({
                  id: 'audit.logs.apiNamePlaceholder',
                  defaultMessage: 'Enter API name',
                })}
                allowClear
                style={{ width: 200 }}
              />
            </Form.Item>

            <Form.Item
              name="dateRange"
              label={intl.formatMessage({
                id: 'audit.logs.timeRange',
                defaultMessage: 'Time Range',
              })}
              style={{ marginBottom: 0 }}
            >
              <RangePicker
                showTime
                format="YYYY-MM-DD HH:mm:ss"
                placeholder={[
                  intl.formatMessage({
                    id: 'audit.logs.startTime',
                    defaultMessage: 'Start Time',
                  }),
                  intl.formatMessage({
                    id: 'audit.logs.endTime',
                    defaultMessage: 'End Time',
                  }),
                ]}
                style={{ width: 420 }}
              />
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SearchOutlined />}
                loading={loading}
              >
                {intl.formatMessage({
                  id: 'common.search',
                  defaultMessage: 'Search',
                })}
              </Button>
            </Form.Item>
          </Space>
        </Form>

        <Divider />

        <Table
          columns={columns}
          dataSource={data}
          loading={loading}
          rowKey="id"
          pagination={{
            current: pagination.current,
            pageSize: pagination.pageSize,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              intl.formatMessage(
                {
                  id: 'common.pagination.total',
                  defaultMessage: 'Showing {start}-{end} of {total} records',
                },
                { start: range[0] || 0, end: range[1] || 0, total },
              ),
            pageSizeOptions: ['20', '50', '100'],
            onChange: handlePaginationChange,
            onShowSizeChange: handlePaginationChange,
            total: data.length,
          }}
          scroll={{ x: 1240 }}
          size="small"
          bordered
        />
      </Card>

      <Drawer
        title={intl.formatMessage({
          id: 'audit.logs.detail.title',
          defaultMessage: 'Audit Log Details',
        })}
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
        width={800}
        destroyOnClose
      >
        {selectedRecord && (
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 16,
                }}
              >
                <Title level={5} style={{ margin: 0 }}>
                  {intl.formatMessage({
                    id: 'audit.logs.detail.requestData',
                    defaultMessage: 'Request Data',
                  })}
                </Title>
                <Button
                  type="link"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={() =>
                    handleCopy(selectedRecord.request_data || '', 'request')
                  }
                  disabled={!selectedRecord.request_data}
                >
                  {intl.formatMessage({
                    id: 'common.copy',
                    defaultMessage: 'Copy',
                  })}
                </Button>
              </div>
              <div
                style={{
                  maxHeight: '400px',
                  overflow: 'auto',
                  border: `1px solid ${token.colorBorder}`,
                  borderRadius: '6px',
                  backgroundColor: token.colorBgContainer,
                  padding: '16px',
                  minHeight: '120px',
                }}
              >
                <pre
                  style={{
                    fontSize: '12px',
                    lineHeight: '1.5',
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontFamily:
                      'Monaco, Menlo, "Ubuntu Mono", Consolas, monospace',
                    color: token.colorText,
                  }}
                >
                  {formatJsonData(selectedRecord.request_data)}
                </pre>
              </div>
            </div>

            <div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 16,
                }}
              >
                <Title level={5} style={{ margin: 0 }}>
                  {intl.formatMessage({
                    id: 'audit.logs.detail.responseData',
                    defaultMessage: 'Response Data',
                  })}
                </Title>
                <Button
                  type="link"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={() =>
                    handleCopy(selectedRecord.response_data || '', 'response')
                  }
                  disabled={!selectedRecord.response_data}
                >
                  {intl.formatMessage({
                    id: 'common.copy',
                    defaultMessage: 'Copy',
                  })}
                </Button>
              </div>
              <div
                style={{
                  maxHeight: '400px',
                  overflow: 'auto',
                  border: `1px solid ${token.colorBorder}`,
                  borderRadius: '6px',
                  backgroundColor: token.colorBgContainer,
                  padding: '16px',
                  minHeight: '120px',
                }}
              >
                <pre
                  style={{
                    fontSize: '12px',
                    lineHeight: '1.5',
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontFamily:
                      'Monaco, Menlo, "Ubuntu Mono", Consolas, monospace',
                    color: token.colorText,
                  }}
                >
                  {formatJsonData(selectedRecord.response_data)}
                </pre>
              </div>
            </div>
          </Space>
        )}
      </Drawer>
    </div>
  );
};

export default AuditLogsPage;
