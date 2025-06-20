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
  Row,
  Col,
  Divider,
  Tooltip,
} from 'antd';
import { SearchOutlined, ReloadOutlined, EyeOutlined, ClearOutlined } from '@ant-design/icons';
import { useIntl } from 'umi';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';
import { AuditApi } from '@/api/apis/audit-api';
import type { AuditLog } from '@/api/models';

const { RangePicker } = DatePicker;
const { Text, Title } = Typography;
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

  // Get status display
  const getStatusDisplay = (statusCode?: number): { text: string; color: string } => {
    if (!statusCode) return { text: '未知', color: 'default' };
    if (statusCode >= 200 && statusCode < 300) return { text: '成功', color: 'success' };
    return { text: '失败', color: 'error' };
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
      message.error('获取审计日志失败');
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
      endDate: endTime.toISOString()
    });
    
    // 设置表单默认值
    form.setFieldsValue({
      dateRange: [startTime, endTime]
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

  // Handle reset
  const handleReset = () => {
    form.resetFields();
    fetchData({ limit: 100 });
  };

  // Handle view details
  const handleViewDetails = async (record: AuditLog) => {
    try {
      const api = new AuditApi();
      const log = await api.getAuditLog({ auditId: record.id! });
      setSelectedRecord(log.data);
      setDetailModalVisible(true);
    } catch (error) {
      message.error('获取详细信息失败');
    }
  };

  // Table columns
  const columns: ColumnsType<AuditLog> = [
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username',
      width: 120,
      render: (text?: string) => (
        <Text strong>{text || '-'}</Text>
      ),
    },
    {
      title: 'API 名称',
      dataIndex: 'api_name',
      key: 'api_name',
      width: 200,
      render: (text?: string) => (
        <Tooltip title={text}>
          <Text code style={{ fontSize: '12px' }}>
            {text && text.length > 30 ? `${text.substring(0, 30)}...` : text || '-'}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: '资源类型',
      dataIndex: 'resource_type',
      key: 'resource_type',
      width: 120,
      render: (type?: string) => {
        return type ? (
          <Tag color="blue" style={{ minWidth: 80, textAlign: 'center' }}>
            {type}
          </Tag>
        ) : '-';
      },
    },
    {
      title: '资源 ID',
      dataIndex: 'resource_id',
      key: 'resource_id',
      width: 140,
      render: (id?: string) => (
        id ? (
          <Tooltip title={id}>
            <Text code style={{ fontSize: '11px' }}>
              {id.length > 18 ? `${id.substring(0, 18)}...` : id}
            </Text>
          </Tooltip>
        ) : '-'
      ),
    },
    {
      title: '状态',
      dataIndex: 'status_code',
      key: 'status_code',
      width: 100,
      align: 'center',
      render: (code?: number) => {
        const status = getStatusDisplay(code);
        return (
          <Tag color={status.color} style={{ minWidth: 50, textAlign: 'center' }}>
            {status.text}
          </Tag>
        );
      },
    },
    {
      title: '开始时间',
      dataIndex: 'start_time',
      key: 'start_time',
      width: 160,
      render: (time?: number) => (
        time ? (
          <Text style={{ fontSize: '12px' }}>
            {dayjs(time).format('MM-DD HH:mm:ss')}
          </Text>
        ) : '-'
      ),
    },
    {
      title: '结束时间',
      dataIndex: 'end_time',
      key: 'end_time',
      width: 160,
      render: (time?: number) => (
        time ? (
          <Text style={{ fontSize: '12px' }}>
            {dayjs(time).format('MM-DD HH:mm:ss')}
          </Text>
        ) : '-'
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      align: 'center',
      render: (_, record) => (
        <Button
          type="link"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetails(record)}
        >
          详情
        </Button>
      ),
    },
  ];

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px' }}>
      <Card>
        <div style={{ marginBottom: '24px' }}>
          <Title level={4} style={{ margin: 0 }}>
            审计日志
          </Title>
          <Text type="secondary">
            查看系统操作的详细审计记录
          </Text>
        </div>

        <Form
          form={form}
          layout="inline"
          onFinish={handleSearch}
          style={{ marginBottom: '24px' }}
        >
          <Space wrap style={{ width: '100%' }}>
            <Form.Item name="apiName" label="API 名称" style={{ marginBottom: 0 }}>
              <Input 
                placeholder="输入API名称" 
                allowClear
                style={{ width: 200 }}
              />
            </Form.Item>

            <Form.Item name="resourceType" label="资源类型" style={{ marginBottom: 0 }}>
              <Select
                placeholder="选择资源类型"
                allowClear
                style={{ width: 150 }}
              >
                {resourceTypes.map(type => (
                  <Option key={type} value={type}>
                    {type}
                  </Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item name="dateRange" label="时间范围" style={{ marginBottom: 0 }}>
              <RangePicker
                showTime
                format="YYYY-MM-DD HH:mm:ss"
                placeholder={['开始时间', '结束时间']}
                style={{ width: 350 }}
              />
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Space>
                <Button 
                  type="primary" 
                  htmlType="submit" 
                  icon={<SearchOutlined />}
                  loading={loading}
                >
                  搜索
                </Button>
                <Button 
                  onClick={handleReset}
                  icon={<ClearOutlined />}
                >
                  重置
                </Button>
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={() => {
                    const endTime = dayjs();
                    const startTime = endTime.subtract(1, 'day');
                    fetchData({ 
                      limit: 100,
                      startDate: startTime.toISOString(),
                      endDate: endTime.toISOString()
                    });
                  }}
                  loading={loading}
                >
                  刷新
                </Button>
              </Space>
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
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `显示 ${range[0] || 0}-${range[1] || 0} 条，共 ${total} 条记录`,
            pageSizeOptions: ['20', '50', '100'],
            defaultPageSize: 20,
          }}
          scroll={{ x: 1200 }}
          size="small"
          bordered
        />
      </Card>

      <Modal
        title="审计日志详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={900}
        style={{ top: 20 }}
      >
        {selectedRecord && (
          <Descriptions column={2} bordered size="small">
            <Descriptions.Item label="日志 ID" span={2}>
              <Text code>{selectedRecord.id}</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="用户名">
              <Text strong>{selectedRecord.username || '-'}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="用户 ID">
              <Text code>{selectedRecord.user_id || '-'}</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="API 名称" span={2}>
              <Text code>{selectedRecord.api_name || '-'}</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="状态">
              {(() => {
                const status = getStatusDisplay(selectedRecord.status_code || undefined);
                return (
                  <Tag color={status.color}>
                    {status.text}
                  </Tag>
                );
              })()}
            </Descriptions.Item>
            <Descriptions.Item label="状态码">
              <Text code>{selectedRecord.status_code || '-'}</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="请求路径" span={2}>
              <Text code style={{ fontSize: '12px', wordBreak: 'break-all' }}>
                {selectedRecord.path}
              </Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="资源类型">
              {selectedRecord.resource_type ? (
                <Tag color="blue">{selectedRecord.resource_type}</Tag>
              ) : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="资源 ID">
              <Text code style={{ fontSize: '12px' }}>
                {selectedRecord.resource_id || '-'}
              </Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="开始时间">
              {selectedRecord.start_time ? 
                dayjs(selectedRecord.start_time).format('YYYY-MM-DD HH:mm:ss') : '-'
              }
            </Descriptions.Item>
            <Descriptions.Item label="结束时间">
              {selectedRecord.end_time ? 
                dayjs(selectedRecord.end_time).format('YYYY-MM-DD HH:mm:ss') : '-'
              }
            </Descriptions.Item>
            
            <Descriptions.Item label="持续时间">
              <Text strong>
                {formatDuration(selectedRecord.duration_ms || undefined)}
              </Text>
            </Descriptions.Item>
            <Descriptions.Item label="IP 地址">
              <Text code>{selectedRecord.ip_address || '-'}</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="User Agent" span={2}>
              <Text 
                ellipsis={{ tooltip: true }} 
                style={{ maxWidth: '100%', fontSize: '12px' }}
              >
                {selectedRecord.user_agent || '-'}
              </Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="请求 ID" span={2}>
              <Text code style={{ fontSize: '12px' }}>
                {selectedRecord.request_id || '-'}
              </Text>
            </Descriptions.Item>
            
            <Descriptions.Item label="创建时间" span={2}>
              {selectedRecord.created ? 
                dayjs(selectedRecord.created).format('YYYY-MM-DD HH:mm:ss') : '-'
              }
            </Descriptions.Item>
            
            {selectedRecord.error_message && (
              <Descriptions.Item label="错误信息" span={2}>
                <Text type="danger" style={{ fontSize: '12px' }}>
                  {selectedRecord.error_message}
                </Text>
              </Descriptions.Item>
            )}
            
            {selectedRecord.request_data && (
              <Descriptions.Item label="请求数据" span={2}>
                <pre style={{ 
                  fontSize: '11px', 
                  maxHeight: '200px', 
                  overflow: 'auto',
                  backgroundColor: '#f5f5f5',
                  padding: '8px',
                  borderRadius: '4px',
                  margin: 0,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all'
                }}>
                  {formatJsonData(selectedRecord.request_data)}
                </pre>
              </Descriptions.Item>
            )}
            
            {selectedRecord.response_data && (
              <Descriptions.Item label="响应数据" span={2}>
                <pre style={{ 
                  fontSize: '11px', 
                  maxHeight: '200px', 
                  overflow: 'auto',
                  backgroundColor: '#f5f5f5',
                  padding: '8px',
                  borderRadius: '4px',
                  margin: 0,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all'
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