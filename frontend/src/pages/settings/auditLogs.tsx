import React, { useState, useEffect } from 'react';
import {
  Table,
  Card,
  Form,
  Input,
  Button,
  Select,
  DatePicker,
  Space,
  Tag,
  Modal,
  message,
  Typography,
  Descriptions,
  InputNumber,
  Tooltip,
  Alert,
} from 'antd';
import { SearchOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons';
import { useIntl } from '@umijs/max';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import { AuditApi } from '@/api';
import type { AuditLog } from '@/api/models';

const { RangePicker } = DatePicker;
const { Text } = Typography;
const { Option } = Select;

const AuditLogsPage: React.FC = () => {
  const intl = useIntl();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AuditLog[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<AuditLog | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);

  const resourceTypes = [
    'collection', 'document', 'bot', 'chat', 'message', 
    'api_key', 'llm_provider', 'llm_provider_model', 
    'model_service_provider', 'user', 'config'
  ];

  const httpMethodOptions = [
    { value: 'POST', label: 'POST' },
    { value: 'PUT', label: 'PUT' },
    { value: 'DELETE', label: 'DELETE' },
  ];

  // Format duration
  const formatDuration = (ms?: number): string => {
    if (!ms) return '-';
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  // Format JSON data
  const formatJsonData = (data?: string): string => {
    if (!data) return '';
    try {
      return JSON.stringify(JSON.parse(data), null, 2);
    } catch {
      return data;
    }
  };

  // Get status color
  const getStatusColor = (statusCode?: number): string => {
    if (!statusCode) return 'default';
    if (statusCode >= 200 && statusCode < 300) return 'green';
    if (statusCode >= 400 && statusCode < 500) return 'orange';
    if (statusCode >= 500) return 'red';
    return 'default';
  };

  // Get HTTP method color
  const getHttpMethodColor = (method?: string): string => {
    switch (method) {
      case 'POST': return 'blue';
      case 'PUT': return 'orange';
      case 'DELETE': return 'red';
      default: return 'default';
    }
  };

  // Fetch audit logs
  const fetchData = async (params?: any) => {
    setLoading(true);
    try {
      const api = new AuditApi();
      const response = await api.listAuditLogs(params);
      setData(response.data.items || []);
    } catch (error) {
      console.error('Failed to fetch audit logs:', error);
      message.error(intl.formatMessage({ id: 'audit.fetchError' }));
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchData({ limit: 1000 });
  }, []);

  // Handle search
  const handleSearch = () => {
    const values = form.getFieldsValue();
    const params: any = { ...values, limit: 1000 };
    
    // Handle date range
    if (values.dateRange) {
      params.startDate = values.dateRange[0]?.toISOString();
      params.endDate = values.dateRange[1]?.toISOString();
      delete params.dateRange;
    }
    
    fetchData(params);
  };

  // Handle reset
  const handleReset = () => {
    form.resetFields();
    fetchData({ limit: 1000 });
  };

  // Handle view details
  const handleViewDetails = async (record: AuditLog) => {
    try {
      const api = new AuditApi();
      const log = await api.getAuditLog({ auditId: record.id! });
      setSelectedRecord(log.data);
      setDetailModalVisible(true);
    } catch (error) {
      message.error(intl.formatMessage({ id: 'audit.fetchDetailError' }));
    }
  };

  // Table columns
  const columns: ColumnsType<AuditLog> = [
    {
      title: intl.formatMessage({ id: 'audit.username' }),
      dataIndex: 'username',
      key: 'username',
      width: 120,
      render: (text?: string) => text || '-',
    },
    {
      title: intl.formatMessage({ id: 'audit.apiName' }),
      dataIndex: 'api_name',
      key: 'api_name',
      width: 150,
      render: (text?: string) => (
        <Tooltip title={text}>
          <Text code style={{ fontSize: '12px' }}>
            {text && text.length > 20 ? `${text.substring(0, 20)}...` : text || '-'}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: intl.formatMessage({ id: 'audit.httpMethod' }),
      dataIndex: 'http_method',
      key: 'http_method',
      width: 100,
      render: (method?: string) => (
        <Tag color={getHttpMethodColor(method)}>{method || '-'}</Tag>
      ),
    },
    {
      title: intl.formatMessage({ id: 'audit.resourceType' }),
      dataIndex: 'resource_type',
      key: 'resource_type',
      width: 120,
      render: (type?: string) => {
        return type ? <Tag color="blue">{type}</Tag> : '-';
      },
    },
    {
      title: intl.formatMessage({ id: 'audit.resourceId' }),
      dataIndex: 'resource_id',
      key: 'resource_id',
      width: 120,
      render: (id?: string) => (
        id ? (
          <Tooltip title={id}>
            <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
              {id.length > 15 ? `${id.substring(0, 15)}...` : id}
            </span>
          </Tooltip>
        ) : '-'
      ),
    },
    {
      title: intl.formatMessage({ id: 'audit.statusCode' }),
      dataIndex: 'status_code',
      key: 'status_code',
      width: 100,
      render: (code?: number) => {
        if (!code) return '-';
        return <Tag color={getStatusColor(code)}>{code}</Tag>;
      },
    },
    {
      title: intl.formatMessage({ id: 'audit.duration' }),
      dataIndex: 'duration_ms',
      key: 'duration_ms',
      width: 100,
      render: (ms?: number) => formatDuration(ms),
    },
    {
      title: intl.formatMessage({ id: 'audit.ipAddress' }),
      dataIndex: 'ip_address',
      key: 'ip_address',
      width: 120,
      render: (ip?: string) => ip || '-',
    },
    {
      title: intl.formatMessage({ id: 'audit.created' }),
      dataIndex: 'created',
      key: 'created',
      width: 150,
      render: (text: string) => dayjs(text).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: intl.formatMessage({ id: 'audit.actions' }),
      key: 'actions',
      width: 80,
      render: (_, record: AuditLog) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetails(record)}
          size="small"
        />
      ),
    },
  ];

  return (
    <div>
      <Card>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Alert
            message={intl.formatMessage({ id: 'audit.description' })}
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
          
          <Form
            form={form}
            layout="inline"
            onFinish={handleSearch}
            style={{ marginBottom: 16 }}
          >
            <Form.Item name="username" label={intl.formatMessage({ id: 'audit.username' })}>
              <Input placeholder={intl.formatMessage({ id: 'audit.enterUsername' })} />
            </Form.Item>

            <Form.Item name="apiName" label={intl.formatMessage({ id: 'audit.apiName' })}>
              <Input placeholder={intl.formatMessage({ id: 'audit.enterApiName' })} />
            </Form.Item>

            <Form.Item name="resourceType" label={intl.formatMessage({ id: 'audit.resourceType' })}>
              <Select
                placeholder={intl.formatMessage({ id: 'audit.selectResourceType' })}
                allowClear
                style={{ width: 150 }}
              >
                {resourceTypes.map(type => (
                  <Option key={type} value={type}>{type}</Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item name="httpMethod" label={intl.formatMessage({ id: 'audit.httpMethod' })}>
              <Select
                placeholder={intl.formatMessage({ id: 'audit.selectHttpMethod' })}
                allowClear
                style={{ width: 120 }}
              >
                {httpMethodOptions.map(option => (
                  <Option key={option.value} value={option.value}>
                    {option.label}
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item name="statusCode" label={intl.formatMessage({ id: 'audit.statusCode' })}>
              <InputNumber 
                placeholder="200" 
                style={{ width: 100 }}
                min={100}
                max={599}
              />
            </Form.Item>

            <Form.Item name="dateRange" label={intl.formatMessage({ id: 'audit.dateRange' })}>
              <RangePicker
                showTime
                format="YYYY-MM-DD HH:mm:ss"
                placeholder={[
                  intl.formatMessage({ id: 'audit.startDate' }),
                  intl.formatMessage({ id: 'audit.endDate' })
                ]}
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" icon={<SearchOutlined />}>
                  {intl.formatMessage({ id: 'audit.search' })}
                </Button>
                <Button onClick={handleReset}>
                  {intl.formatMessage({ id: 'audit.reset' })}
                </Button>
                <Button icon={<ReloadOutlined />} onClick={() => fetchData({ limit: 1000 })}>
                  {intl.formatMessage({ id: 'audit.refresh' })}
                </Button>
              </Space>
            </Form.Item>
          </Form>

          <Table
            columns={columns}
            dataSource={data}
            loading={loading}
            rowKey="id"
            pagination={{
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total, range) =>
                intl.formatMessage(
                  { id: 'audit.pagination' },
                  { start: range[0], end: range[1], total }
                ),
            }}
            scroll={{ x: 1200 }}
            size="small"
          />
        </Space>
      </Card>

      <Modal
        title={intl.formatMessage({ id: 'audit.detailTitle' })}
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedRecord && (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.id' })}>
              {selectedRecord.id}
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.username' })}>
              {selectedRecord.username || '-'}
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.userId' })}>
              {selectedRecord.user_id || '-'}
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.apiName' })}>
              <Text code>{selectedRecord.api_name || '-'}</Text>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.httpMethod' })}>
              <Tag color={getHttpMethodColor(selectedRecord.http_method)}>
                {selectedRecord.http_method}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.path' })}>
              <Text code style={{ fontSize: '12px' }}>{selectedRecord.path}</Text>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.resourceType' })}>
              {selectedRecord.resource_type ? (
                <Tag color="blue">{selectedRecord.resource_type}</Tag>
              ) : '-'}
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.resourceId' })}>
              <Text code style={{ fontSize: '12px' }}>
                {selectedRecord.resource_id || '-'}
              </Text>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.statusCode' })}>
              <Tag color={getStatusColor(selectedRecord.status_code)}>
                {selectedRecord.status_code}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.startTime' })}>
              {selectedRecord.start_time ? 
                dayjs(selectedRecord.start_time).format('YYYY-MM-DD HH:mm:ss') : '-'
              }
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.endTime' })}>
              {selectedRecord.end_time ? 
                dayjs(selectedRecord.end_time).format('YYYY-MM-DD HH:mm:ss') : '-'
              }
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.duration' })}>
              {formatDuration(selectedRecord.duration_ms)}
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.ipAddress' })}>
              {selectedRecord.ip_address || '-'}
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.userAgent' })}>
              <Text ellipsis={{ tooltip: true }} style={{ maxWidth: 500 }}>
                {selectedRecord.user_agent || '-'}
              </Text>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.requestId' })}>
              <Text code style={{ fontSize: '12px' }}>
                {selectedRecord.request_id || '-'}
              </Text>
            </Descriptions.Item>
            <Descriptions.Item label={intl.formatMessage({ id: 'audit.created' })}>
              {selectedRecord.created ? 
                dayjs(selectedRecord.created).format('YYYY-MM-DD HH:mm:ss') : '-'
              }
            </Descriptions.Item>
            {selectedRecord.error_message && (
              <Descriptions.Item label={intl.formatMessage({ id: 'audit.errorMessage' })}>
                <Text type="danger">{selectedRecord.error_message}</Text>
              </Descriptions.Item>
            )}
            {selectedRecord.request_data && (
              <Descriptions.Item label={intl.formatMessage({ id: 'audit.requestData' })}>
                <pre style={{ 
                  fontSize: '12px', 
                  maxHeight: '200px', 
                  overflow: 'auto',
                  backgroundColor: '#f5f5f5',
                  padding: '8px',
                  borderRadius: '4px'
                }}>
                  {formatJsonData(selectedRecord.request_data)}
                </pre>
              </Descriptions.Item>
            )}
            {selectedRecord.response_data && (
              <Descriptions.Item label={intl.formatMessage({ id: 'audit.responseData' })}>
                <pre style={{ 
                  fontSize: '12px', 
                  maxHeight: '200px', 
                  overflow: 'auto',
                  backgroundColor: '#f5f5f5',
                  padding: '8px',
                  borderRadius: '4px'
                }}>
                  {formatJsonData(selectedRecord.response_data)}
                </pre>
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default AuditLogsPage; 