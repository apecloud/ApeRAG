import { QuotaInfo, UserQuotaInfo, UserQuotaList, SystemDefaultQuotas, SystemDefaultQuotasResponse, QuotaUpdateRequest } from '@/api';
import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { quotasApi } from '@/services';
import { EditOutlined, ReloadOutlined, SettingOutlined, SearchOutlined, ClearOutlined, SaveOutlined, CloseOutlined } from '@ant-design/icons';
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
  Table, 
  TableProps, 
  Typography, 
  message,
  Tabs,
  Space,
  Alert 
} from 'antd';
import { useCallback, useEffect, useState } from 'react';
import { FormattedMessage, useIntl, useModel } from 'umi';

interface EditableQuotaInfo extends QuotaInfo {
  editable?: boolean;
  originalLimit?: number;
}

export default () => {
  const { formatMessage } = useIntl();
  const userModel = useModel('user');
  const [currentUserQuota, setCurrentUserQuota] = useState<UserQuotaInfo>();
  const [searchedUserQuota, setSearchedUserQuota] = useState<UserQuotaInfo>();
  const [searchResults, setSearchResults] = useState<UserQuotaInfo[]>();
  const [systemDefaultQuotas, setSystemDefaultQuotas] = useState<SystemDefaultQuotas>();
  const [loading, setLoading] = useState<boolean>(false);
  const [searchLoading, setSearchLoading] = useState<boolean>(false);
  const [systemQuotasLoading, setSystemQuotasLoading] = useState<boolean>(false);
  const [systemQuotasModalVisible, setSystemQuotasModalVisible] = useState<boolean>(false);
  const [searchValue, setSearchValue] = useState<string>('');
  const [systemQuotasForm] = Form.useForm();
  
  // 表格编辑模式状态
  const [isTableEditMode, setIsTableEditMode] = useState<boolean>(false);
  const [editableQuotas, setEditableQuotas] = useState<EditableQuotaInfo[]>([]);
  const [updateLoading, setUpdateLoading] = useState<boolean>(false);

  const isAdmin = (userModel as any)?.user?.role === 'admin';

  const getQuotaTypeName = (quotaType: string) => {
    const typeMap: Record<string, string> = {
      'max_collection_count': formatMessage({ id: 'quota.collection_count' }),
      'max_document_count': formatMessage({ id: 'quota.document_count_overall' }),
      'max_document_count_per_collection': formatMessage({ id: 'quota.document_count_per_collection' }),
      'max_bot_count': formatMessage({ id: 'quota.bot_count' }),
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
      setSearchResults(undefined);
      return;
    }

    setSearchLoading(true);
    setSearchedUserQuota(undefined);
    setSearchResults(undefined);
    
    try {
      const res = await quotasApi.quotasGet({ search: searchTerm });
      const data = res.data;
      
      if ((data as UserQuotaList).items) {
        const userList = data as UserQuotaList;
        setSearchResults(userList.items);
      } else {
        const userQuota = data as UserQuotaInfo;
        setSearchedUserQuota(userQuota);
      }
    } catch (error: any) {
      if (error?.response?.status === 404) {
        message.warning(formatMessage({ id: 'quota.user_not_found' }));
      } else {
        message.error(formatMessage({ id: 'quota.search_error' }));
      }
      setSearchedUserQuota(undefined);
      setSearchResults(undefined);
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
    setSearchResults(undefined);
    setIsTableEditMode(false);
    setEditableQuotas([]);
  };

  const handleSelectUser = async (userId: string) => {
    try {
      const res = await quotasApi.quotasGet({ userId });
      const userQuota = res.data as UserQuotaInfo;
      setSearchedUserQuota(userQuota);
      setSearchResults(undefined);
      setIsTableEditMode(false);
      setEditableQuotas([]);
    } catch (error) {
      message.error(formatMessage({ id: 'quota.fetch_error' }));
    }
  };

  const handleRecalculateUsage = async (userId: string) => {
    try {
      await quotasApi.quotasUserIdRecalculatePost({
        userId: userId
      });
      message.success(formatMessage({ id: 'quota.recalculate.success' }));
      
      if (searchedUserQuota && searchedUserQuota.user_id === userId) {
        searchUserQuotas(searchValue);
      } else {
        getCurrentUserQuotas();
      }
    } catch (error) {
      message.error(formatMessage({ id: 'quota.recalculate_error' }));
    }
  };

  // 进入表格编辑模式
  const enterTableEditMode = (user: UserQuotaInfo) => {
    const editableData = user.quotas.map(quota => ({
      ...quota,
      editable: true,
      originalLimit: quota.quota_limit
    }));
    setEditableQuotas(editableData);
    setIsTableEditMode(true);
  };

  // 退出表格编辑模式
  const exitTableEditMode = () => {
    setIsTableEditMode(false);
    setEditableQuotas([]);
  };

  // 保存更改
  const handleSave = async () => {
    const displayUser = searchedUserQuota || currentUserQuota;
    if (!displayUser) return;

    setUpdateLoading(true);
    try {
      // 构建更新请求
      const quotaUpdates: any = {};
      editableQuotas.forEach(quota => {
        if (quota.quota_limit !== quota.originalLimit) {
          quotaUpdates[quota.quota_type] = quota.quota_limit;
        }
      });

      // 如果没有更改，直接退出编辑模式
      if (Object.keys(quotaUpdates).length === 0) {
        exitTableEditMode();
        return;
      }

      const request: QuotaUpdateRequest = quotaUpdates;

      await quotasApi.quotasUserIdPut({
        userId: displayUser.user_id,
        quotaUpdateRequest: request
      });

      message.success(formatMessage({ id: 'quota.update.success' }));
      exitTableEditMode();
      
      // 刷新数据
      if (searchedUserQuota) {
        searchUserQuotas(searchValue);
      } else {
        getCurrentUserQuotas();
      }
    } catch (error) {
      message.error(formatMessage({ id: 'quota.update_error' }));
    } finally {
      setUpdateLoading(false);
    }
  };

  // 更新可编辑配额的值
  const updateEditableQuota = (quotaType: string, newLimit: number) => {
    setEditableQuotas(prev => 
      prev.map(quota => 
        quota.quota_type === quotaType 
          ? { ...quota, quota_limit: newLimit }
          : quota
      )
    );
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

  const userQuotaColumns: TableProps<EditableQuotaInfo>['columns'] = [
    {
      title: formatMessage({ id: 'quota.name' }),
      dataIndex: 'quota_type',
      width: 200,
      render: (quotaType: string) => (
        <div style={{ height: '32px', display: 'flex', alignItems: 'center' }}>
          {getQuotaTypeName(quotaType)}
        </div>
      ),
    },
    {
      title: formatMessage({ id: 'quota.usage_rate' }),
      width: 150,
      render: (_, record) => {
        const percentage = record.quota_limit > 0 ? (record.current_usage / record.quota_limit) * 100 : 0;
        const status = percentage >= 100 ? 'exception' : percentage >= 80 ? 'active' : 'normal';
        return (
          <div style={{ height: '32px', display: 'flex', alignItems: 'center' }}>
            <Progress
              percent={Math.min(percentage, 100)}
              status={status}
              size="small"
              format={() => `${Math.round(percentage)}%`}
            />
          </div>
        );
      },
    },
    {
      title: formatMessage({ id: 'quota.current_usage' }),
      dataIndex: 'current_usage',
      width: 120,
      align: 'right',
      render: (value: number) => (
        <div style={{ height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
          {value}
        </div>
      ),
    },
    {
      title: formatMessage({ id: 'quota.max_limit' }),
      dataIndex: 'quota_limit',
      width: 120,
      align: 'right',
      render: (value: number, record: EditableQuotaInfo) => (
        <div style={{ height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
          {isTableEditMode && record.editable ? (
            <InputNumber
              value={value}
              min={0}
              size="small"
              style={{ width: '100px' }}
              onChange={(newValue) => {
                if (newValue !== null) {
                  updateEditableQuota(record.quota_type, newValue);
                }
              }}
            />
          ) : (
            value
          )}
        </div>
      ),
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
    const shouldShowUser = isAdmin ? 
      (searchValue ? !!searchedUserQuota : !!currentUserQuota) : 
      !!currentUserQuota;
    
    const displayUser = searchedUserQuota || currentUserQuota;

    return (
      <div>
        {/* 管理员搜索栏 */}
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

        {/* 多个搜索结果选择 */}
        {searchResults && searchResults.length > 0 && (
          <Card 
            title={formatMessage({ id: 'quota.search_results' })}
            style={{ marginBottom: 16 }}
          >
            <Typography.Paragraph type="secondary">
              <FormattedMessage 
                id="quota.multiple_results_found" 
                values={{ count: searchResults.length }}
              />
            </Typography.Paragraph>
            <Table
              rowKey="user_id"
              columns={[
                {
                  title: formatMessage({ id: 'quota.username' }),
                  dataIndex: 'username',
                },
                {
                  title: formatMessage({ id: 'quota.user_id' }),
                  dataIndex: 'user_id',
                },
                {
                  title: formatMessage({ id: 'quota.email' }),
                  dataIndex: 'email',
                  render: (email) => email || formatMessage({ id: 'quota.not_set' }),
                },
                {
                  title: formatMessage({ id: 'quota.role' }),
                  dataIndex: 'role',
                },
                {
                  title: formatMessage({ id: 'action.name' }),
                  render: (_, record) => (
                    <Button
                      type="link"
                      onClick={() => handleSelectUser(record.user_id)}
                    >
                      <FormattedMessage id="quota.select_user" />
                    </Button>
                  ),
                },
              ]}
              dataSource={searchResults}
              pagination={false}
              size="small"
            />
          </Card>
        )}

        {/* 用户配额显示 */}
        {shouldShowUser && displayUser && (
          <div>
            {/* 用户信息卡片 */}
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


            {/* 配额信息卡片 */}
            <Card 
              title={formatMessage({ id: 'quota.quota_info' })}
              extra={
                isAdmin && displayUser && (
                  <Space>
                    {isTableEditMode ? (
                      <>
                        <Button
                          type="primary"
                          icon={<SaveOutlined />}
                          onClick={handleSave}
                          loading={updateLoading}
                        >
                          <FormattedMessage id="quota.save_changes" />
                        </Button>
                        <Button
                          icon={<CloseOutlined />}
                          onClick={exitTableEditMode}
                        >
                          <FormattedMessage id="quota.cancel_edit" />
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button
                          icon={<EditOutlined />}
                          onClick={() => enterTableEditMode(displayUser)}
                        >
                          <FormattedMessage id="action.edit" />
                        </Button>
                        <Button
                          icon={<ReloadOutlined />}
                          onClick={() => handleRecalculateUsage(displayUser.user_id)}
                        >
                          <FormattedMessage id="quota.recalculate" />
                        </Button>
                      </>
                    )}
                  </Space>
                )
              }
            >
              <div className="quota-table-container">
                <style>
                  {`
                    .quota-table-container .ant-table-tbody > tr {
                      height: 54px !important;
                    }
                    .quota-table-container .ant-table-tbody > tr > td {
                      height: 54px !important;
                      vertical-align: middle !important;
                      padding: 8px 16px !important;
                      line-height: 38px !important;
                    }
                    .quota-table-container .ant-input-number {
                      height: 32px !important;
                      line-height: 30px !important;
                    }
                    .quota-table-container .ant-input-number-input {
                      height: 30px !important;
                      line-height: 30px !important;
                    }
                    .quota-table-container .ant-progress {
                      margin: 0 !important;
                    }
                  `}
                </style>
                <Table
                  rowKey="quota_type"
                  bordered
                  columns={userQuotaColumns}
                  dataSource={isTableEditMode ? editableQuotas : displayUser.quotas}
                  loading={loading || searchLoading}
                  pagination={false}
                  size="middle"
                />
              </div>
            </Card>
          </div>
        )}

        {/* 无数据状态 */}
        {!shouldShowUser && !searchResults && !loading && !searchLoading && (
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
