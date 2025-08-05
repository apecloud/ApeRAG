import { QuotaInfo, UserQuotaInfo, UserQuotaList, SystemDefaultQuotas, SystemDefaultQuotasResponse } from '@/api';
import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { quotasApi } from '@/services';
import { EditOutlined, ReloadOutlined, SettingOutlined, SearchOutlined, ClearOutlined } from '@ant-design/icons';
import { 
  Button, 
  Card, 
  Col, 
  Form, 
  Input,
  InputNumber, 
  Modal, 
  Progress, 
  Row, 
  Select, 
  Table, 
  TableProps, 
  Typography, 
  message,
  Tabs,
  Space 
} from 'antd';
import { useCallback, useEffect, useState } from 'react';
import { FormattedMessage, useIntl, useModel } from 'umi';

export default () => {
  const { formatMessage } = useIntl();
  const userModel = useModel('user');
  const [currentUserQuota, setCurrentUserQuota] = useState<UserQuotaInfo>();
  const [searchedUserQuota, setSearchedUserQuota] = useState<UserQuotaInfo>();
  const [systemDefaultQuotas, setSystemDefaultQuotas] = useState<SystemDefaultQuotas>();
  const [loading, setLoading] = useState<boolean>(false);
  const [searchLoading, setSearchLoading] = useState<boolean>(false);
  const [systemQuotasLoading, setSystemQuotasLoading] = useState<boolean>(false);
  const [editModalVisible, setEditModalVisible] = useState<boolean>(false);
  const [systemQuotasModalVisible, setSystemQuotasModalVisible] = useState<boolean>(false);
  const [editingUser, setEditingUser] = useState<UserQuotaInfo>();
  const [searchValue, setSearchValue] = useState<string>('');
  const [form] = Form.useForm();
  const [systemQuotasForm] = Form.useForm();

  const isAdmin = (userModel as any)?.user?.role === 'admin';

  const getQuotaTypeName = (quotaType: string) => {
    const typeMap: Record<string, string> = {
      'max_collection_count': formatMessage({ id: 'quota.max_collection_count' }),
      'max_document_count': formatMessage({ id: 'quota.max_document_count' }),
      'max_document_count_per_collection': formatMessage({ id: 'quota.max_document_count_per_collection' }),
      'max_bot_count': formatMessage({ id: 'quota.max_bot_count' }),
    };
    return typeMap[quotaType] || quotaType;
  };

  const getCurrentUserQuotas = useCallback(async () => {
    setLoading(true);
    try {
      const res = await quotasApi.quotasGet();
      const userQuota = res.data as UserQuotaInfo;
      setCurrentUserQuota(userQuota);
    } catch (error) {
      message.error(formatMessage({ id: 'quota.fetch_error' }));
    } finally {
      setLoading(false);
    }
  }, [formatMessage]);

  const searchUserQuotas = useCallback(async (searchTerm: string) => {
    if (!searchTerm.trim()) {
      setSearchedUserQuota(undefined);
      return;
    }

    setSearchLoading(true);
    // Clear previous search result immediately when starting new search
    setSearchedUserQuota(undefined);
    
    try {
      // Use backend search functionality
      const res = await quotasApi.quotasGet({ allUsers: true, search: searchTerm });
      const quotaList = res.data as UserQuotaList;
      
      if (quotaList.items.length === 1) {
        // Found exactly one user
        setSearchedUserQuota(quotaList.items[0]);
      } else if (quotaList.items.length === 0) {
        // No users found
        message.warning(formatMessage({ id: 'quota.user_not_found' }));
        setSearchedUserQuota(undefined);
      } else {
        // Multiple users found (shouldn't happen with exact search, but just in case)
        setSearchedUserQuota(quotaList.items[0]);
      }
    } catch (error) {
      message.error(formatMessage({ id: 'quota.search_error' }));
      // Clear search result on error as well
      setSearchedUserQuota(undefined);
    } finally {
      setSearchLoading(false);
    }
  }, [formatMessage]);

  const handleSearch = (value: string) => {
    setSearchValue(value);
    if (isAdmin) {
      searchUserQuotas(value);
    }
  };

  const clearSearch = () => {
    setSearchValue('');
    setSearchedUserQuota(undefined);
  };

  const handleEditQuota = (user: UserQuotaInfo) => {
    setEditingUser(user);
    setEditModalVisible(true);
    form.resetFields();
  };

  const handleUpdateQuota = async (values: { quota_type: string; new_limit: number }) => {
    if (!editingUser) return;

    try {
      await quotasApi.quotasUserIdPut({
        userId: editingUser.user_id,
        quotaUpdateRequest: {
          quota_type: values.quota_type as any,
          new_limit: values.new_limit
        }
      });
      message.success(formatMessage({ id: 'quota.update_success' }));
      setEditModalVisible(false);
      
      // Refresh the data
      if (searchedUserQuota && searchedUserQuota.user_id === editingUser.user_id) {
        // If we're viewing a searched user, refresh their data
        searchUserQuotas(searchValue);
      } else {
        // Otherwise refresh current user data
        getCurrentUserQuotas();
      }
    } catch (error) {
      message.error(formatMessage({ id: 'quota.update_error' }));
    }
  };

  const handleRecalculateUsage = async (userId: string) => {
    try {
      await quotasApi.quotasUserIdRecalculatePost({
        userId: userId
      });
      message.success(formatMessage({ id: 'quota.recalculate_success' }));
      
      // Refresh the data
      if (searchedUserQuota && searchedUserQuota.user_id === userId) {
        // If we're viewing a searched user, refresh their data
        searchUserQuotas(searchValue);
      } else {
        // Otherwise refresh current user data
        getCurrentUserQuotas();
      }
    } catch (error) {
      message.error(formatMessage({ id: 'quota.recalculate_error' }));
    }
  };

  const getSystemDefaultQuotas = useCallback(async () => {
    if (!isAdmin) return;
    
    setSystemQuotasLoading(true);
    try {
      const res = await quotasApi.systemDefaultQuotasGet();
      const response = res.data as SystemDefaultQuotasResponse;
      setSystemDefaultQuotas(response.quotas);
    } catch (error) {
      message.error(formatMessage({ id: 'quota.system_fetch_error' }));
    } finally {
      setSystemQuotasLoading(false);
    }
  }, [isAdmin, formatMessage]);

  const handleEditSystemQuotas = () => {
    if (systemDefaultQuotas) {
      systemQuotasForm.setFieldsValue(systemDefaultQuotas);
      setSystemQuotasModalVisible(true);
    }
  };

  const handleUpdateSystemQuotas = async (values: SystemDefaultQuotas) => {
    try {
      await quotasApi.systemDefaultQuotasPut({
        systemDefaultQuotasUpdateRequest: {
          quotas: values
        }
      });
      message.success(formatMessage({ id: 'quota.system_update_success' }));
      setSystemQuotasModalVisible(false);
      getSystemDefaultQuotas();
    } catch (error) {
      message.error(formatMessage({ id: 'quota.system_update_error' }));
    }
  };


  const userQuotaColumns: TableProps<QuotaInfo>['columns'] = [
    {
      title: formatMessage({ id: 'quota.name' }),
      dataIndex: 'quota_type',
      render: (quotaType: string) => getQuotaTypeName(quotaType),
    },
    {
      title: formatMessage({ id: 'quota.max_limit' }),
      dataIndex: 'quota_limit',
      align: 'right',
    },
    {
      title: formatMessage({ id: 'quota.current_usage' }),
      dataIndex: 'current_usage',
      align: 'right',
    },
    {
      title: formatMessage({ id: 'quota.usage_rate' }),
      render: (_, record) => {
        const percentage = record.quota_limit > 0 ? (record.current_usage / record.quota_limit) * 100 : 0;
        const status = percentage >= 100 ? 'exception' : percentage >= 80 ? 'active' : 'normal';
        return (
          <Progress
            percent={Math.min(percentage, 100)}
            status={status}
            size="small"
            format={() => `${Math.round(percentage)}%`}
          />
        );
      },
    },
  ];

  useEffect(() => {
    getCurrentUserQuotas();
    if (isAdmin) {
      getSystemDefaultQuotas();
    }
  }, [getCurrentUserQuotas, getSystemDefaultQuotas, isAdmin]);

  const renderSystemDefaultQuotasTab = () => (
    <Card 
      title={formatMessage({ id: 'quota.system_default_quotas' })}
      extra={
        <Button
          type="primary"
          icon={<SettingOutlined />}
          onClick={handleEditSystemQuotas}
          loading={systemQuotasLoading}
        >
          <FormattedMessage id="action.edit" />
        </Button>
      }
    >
      {systemDefaultQuotas ? (
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Typography.Title level={3} style={{ margin: 0 }}>
                  {systemDefaultQuotas.max_collection_count}
                </Typography.Title>
                <Typography.Text type="secondary">
                  {getQuotaTypeName('max_collection_count')}
                </Typography.Text>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Typography.Title level={3} style={{ margin: 0 }}>
                  {systemDefaultQuotas.max_document_count}
                </Typography.Title>
                <Typography.Text type="secondary">
                  {getQuotaTypeName('max_document_count')}
                </Typography.Text>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Typography.Title level={3} style={{ margin: 0 }}>
                  {systemDefaultQuotas.max_document_count_per_collection}
                </Typography.Title>
                <Typography.Text type="secondary">
                  {getQuotaTypeName('max_document_count_per_collection')}
                </Typography.Text>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Typography.Title level={3} style={{ margin: 0 }}>
                  {systemDefaultQuotas.max_bot_count}
                </Typography.Title>
                <Typography.Text type="secondary">
                  {getQuotaTypeName('max_bot_count')}
                </Typography.Text>
              </div>
            </Card>
          </Col>
        </Row>
      ) : (
        <Typography.Text type="secondary">
          <FormattedMessage id="quota.system_loading" />
        </Typography.Text>
      )}
    </Card>
  );

  const renderUserQuotasTab = () => {
    // Determine which user data to display
    // If admin is searching and has search value but no search result, don't show any user
    const shouldShowUser = isAdmin ? 
      (searchValue ? !!searchedUserQuota : !!currentUserQuota) : 
      !!currentUserQuota;
    
    const displayUser = searchedUserQuota || currentUserQuota;

    return (
      <div>
        {/* Search bar for admin users */}
        {isAdmin && (
          <Card style={{ marginBottom: 16 }}>
            <Space.Compact style={{ width: '100%' }}>
              <Input
                placeholder={formatMessage({ id: 'quota.search_placeholder' })}
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                onPressEnter={() => handleSearch(searchValue)}
                style={{ flex: 1 }}
              />
              <Button
                type="primary"
                icon={<SearchOutlined />}
                onClick={() => handleSearch(searchValue)}
                loading={searchLoading}
              >
                <FormattedMessage id="action.search" />
              </Button>
              <Button
                icon={<ClearOutlined />}
                onClick={clearSearch}
                disabled={!searchValue}
              >
                <FormattedMessage id="quota.clear" />
              </Button>
            </Space.Compact>
            {searchValue && (
              <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                <FormattedMessage 
                  id="quota.search_tip" 
                  values={{ searchTerm: searchValue }}
                />
              </div>
            )}
          </Card>
        )}


        {/* User quota display */}
        {shouldShowUser && displayUser && (
          <div>
            {/* User Information Card */}
            <Card 
              title={formatMessage({ id: 'quota.user_info' })}
              style={{ marginBottom: 16 }}
            >
              <Row gutter={[24, 16]}>
                <Col span={6}>
                  <div>
                    <Typography.Text type="secondary">{formatMessage({ id: 'quota.username' })}</Typography.Text>
                    <br />
                    <Typography.Text strong>{displayUser.username}</Typography.Text>
                  </div>
                </Col>
                <Col span={6}>
                  <div>
                    <Typography.Text type="secondary">{formatMessage({ id: 'quota.user_id' })}</Typography.Text>
                    <br />
                    <Typography.Text strong>{displayUser.user_id}</Typography.Text>
                  </div>
                </Col>
                <Col span={6}>
                  <div>
                    <Typography.Text type="secondary">{formatMessage({ id: 'quota.email' })}</Typography.Text>
                    <br />
                    <Typography.Text strong>{displayUser.email || formatMessage({ id: 'quota.not_set' })}</Typography.Text>
                  </div>
                </Col>
                <Col span={6}>
                  <div>
                    <Typography.Text type="secondary">{formatMessage({ id: 'quota.role' })}</Typography.Text>
                    <br />
                    <Typography.Text strong>{displayUser.role}</Typography.Text>
                  </div>
                </Col>
              </Row>
            </Card>

            {/* Quota Information Card */}
            <Card 
              title={formatMessage({ id: 'quota.quota_info' })}
              extra={
                isAdmin && displayUser && (
                  <Space>
                    <Button
                      type="link"
                      size="small"
                      icon={<EditOutlined />}
                      onClick={() => handleEditQuota(displayUser)}
                    >
                      <FormattedMessage id="action.edit" />
                    </Button>
                    <Button
                      type="link"
                      size="small"
                      icon={<ReloadOutlined />}
                      onClick={() => handleRecalculateUsage(displayUser.user_id)}
                    >
                      <FormattedMessage id="quota.recalculate" />
                    </Button>
                  </Space>
                )
              }
            >
              <Table
                rowKey="quota_type"
                bordered
                columns={userQuotaColumns}
                dataSource={displayUser.quotas}
                loading={loading || searchLoading}
                pagination={false}
                size="middle"
              />
            </Card>
          </div>
        )}

        {/* No data state */}
        {!shouldShowUser && !loading && !searchLoading && (
          <Card>
            <div style={{ textAlign: 'center', padding: 40 }}>
              <Typography.Text type="secondary">
                {isAdmin && searchValue 
                  ? formatMessage({ id: 'quota.no_search_results' })
                  : formatMessage({ id: 'quota.no_data' })
                }
              </Typography.Text>
            </div>
          </Card>
        )}
      </div>
    );
  };

  return (
    <PageContainer>
      <PageHeader
        title={formatMessage({ id: 'quota.management' })}
        description={formatMessage({ id: 'quota.management_tips' })}
      >
        <RefreshButton 
          loading={loading} 
          onClick={() => {
            if (searchedUserQuota) {
              searchUserQuotas(searchValue);
            } else {
              getCurrentUserQuotas();
            }
          }} 
        />
      </PageHeader>

      {isAdmin ? (
        <Tabs
          defaultActiveKey="user_quotas"
          items={[
            {
              key: 'user_quotas',
              label: formatMessage({ id: 'quota.user_quotas' }),
              children: renderUserQuotasTab(),
            },
            {
              key: 'system_defaults',
              label: formatMessage({ id: 'quota.system_defaults' }),
              children: renderSystemDefaultQuotasTab(),
            },
          ]}
        />
      ) : (
        renderUserQuotasTab()
      )}

      <Modal
        title={formatMessage({ id: 'quota.edit_quota' })}
        open={editModalVisible}
        onCancel={() => setEditModalVisible(false)}
        onOk={() => form.submit()}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpdateQuota}
        >
          <Form.Item
            name="quota_type"
            label={formatMessage({ id: 'quota.type' })}
            rules={[{ required: true }]}
          >
            <Select placeholder={formatMessage({ id: 'quota.select_type' })}>
              <Select.Option value="max_collection_count">
                {getQuotaTypeName('max_collection_count')}
              </Select.Option>
              <Select.Option value="max_document_count">
                {getQuotaTypeName('max_document_count')}
              </Select.Option>
              <Select.Option value="max_document_count_per_collection">
                {getQuotaTypeName('max_document_count_per_collection')}
              </Select.Option>
              <Select.Option value="max_bot_count">
                {getQuotaTypeName('max_bot_count')}
              </Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="new_limit"
            label={formatMessage({ id: 'quota.new_limit' })}
            rules={[
              { required: true },
              { type: 'number', min: 0 }
            ]}
          >
            <InputNumber
              min={0}
              style={{ width: '100%' }}
              placeholder={formatMessage({ id: 'quota.enter_new_limit' })}
            />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={formatMessage({ id: 'quota.edit_system_defaults' })}
        open={systemQuotasModalVisible}
        onCancel={() => setSystemQuotasModalVisible(false)}
        onOk={() => systemQuotasForm.submit()}
        destroyOnClose
        width={600}
      >
        <Form
          form={systemQuotasForm}
          layout="vertical"
          onFinish={handleUpdateSystemQuotas}
        >
          <Form.Item
            name="max_collection_count"
            label={getQuotaTypeName('max_collection_count')}
            rules={[
              { required: true },
              { type: 'number', min: 1 }
            ]}
          >
            <InputNumber
              min={1}
              style={{ width: '100%' }}
              placeholder={formatMessage({ id: 'quota.enter_default_limit' })}
            />
          </Form.Item>
          <Form.Item
            name="max_document_count"
            label={getQuotaTypeName('max_document_count')}
            rules={[
              { required: true },
              { type: 'number', min: 1 }
            ]}
          >
            <InputNumber
              min={1}
              style={{ width: '100%' }}
              placeholder={formatMessage({ id: 'quota.enter_default_limit' })}
            />
          </Form.Item>
          <Form.Item
            name="max_document_count_per_collection"
            label={getQuotaTypeName('max_document_count_per_collection')}
            rules={[
              { required: true },
              { type: 'number', min: 1 }
            ]}
          >
            <InputNumber
              min={1}
              style={{ width: '100%' }}
              placeholder={formatMessage({ id: 'quota.enter_default_limit' })}
            />
          </Form.Item>
          <Form.Item
            name="max_bot_count"
            label={getQuotaTypeName('max_bot_count')}
            rules={[
              { required: true },
              { type: 'number', min: 1 }
            ]}
          >
            <InputNumber
              min={1}
              style={{ width: '100%' }}
              placeholder={formatMessage({ id: 'quota.enter_default_limit' })}
            />
          </Form.Item>
        </Form>
      </Modal>
    </PageContainer>
  );
};
