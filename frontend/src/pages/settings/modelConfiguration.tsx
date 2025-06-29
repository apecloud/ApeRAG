import { LlmConfigurationResponse, LlmProvider, LlmProviderModel } from '@/api';
import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { api } from '@/services';
import {
  ArrowLeftOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  SearchOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import {
  Button,
  Col,
  Divider,
  Form,
  Input,
  InputNumber,
  message,
  Modal,
  Row,
  Select,
  Space,
  Table,
  TableProps,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import { useCallback, useEffect, useState } from 'react';
import { useIntl, useModel } from 'umi';

const { Title, Text } = Typography;

// 弹窗内容视图类型
type ModalViewType = 'list' | 'add' | 'edit';

// API Key配置相关类型 - 已移除，API Key现在直接在LLM Provider中管理

export default () => {
  const { loading, setLoading } = useModel('global');
  const { formatMessage } = useIntl();

  const [configuration, setConfiguration] = useState<LlmConfigurationResponse>({
    providers: [],
    models: [],
  });

  // Provider form state
  const [providerForm] = Form.useForm();
  const [providerModalVisible, setProviderModalVisible] = useState(false);
  const [editingProvider, setEditingProvider] = useState<LlmProvider | null>(
    null,
  );

  // API Key配置相关状态 - 已移除，API Key现在直接在LLM Provider中管理

  // Model management modal state
  const [modelManagementVisible, setModelManagementVisible] = useState(false);
  const [currentProvider, setCurrentProvider] = useState<LlmProvider | null>(
    null,
  );
  const [modalView, setModalView] = useState<ModalViewType>('list');
  const [modelForm] = Form.useForm();
  const [editingModel, setEditingModel] = useState<LlmProviderModel | null>(
    null,
  );
  // Model search state
  const [modelSearchText, setModelSearchText] = useState<string>('');

  const [modal, contextHolder] = Modal.useModal();

  // Fetch LLM configuration
  const fetchConfiguration = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.llmConfigurationGet();
      setConfiguration(res.data);
    } catch (error) {
      message.error(formatMessage({ id: 'model.configuration.fetch.failed' }));
    } finally {
      setLoading(false);
    }
  }, [setLoading, formatMessage]);

  // API Key配置相关方法 - 已移除，API Key现在直接在LLM Provider中管理

  // Provider operations
  const handleCreateProvider = useCallback(() => {
    setEditingProvider(null);
    providerForm.resetFields();
    providerForm.setFieldsValue({
      completion_dialect: 'openai',
      embedding_dialect: 'openai',
      rerank_dialect: 'jina_ai',
      api_key: '', // Explicitly clear API key field to prevent sharing between providers
    });
    setProviderModalVisible(true);
  }, [providerForm]);

  const handleEditProvider = useCallback(
    async (provider: LlmProvider) => {
      setEditingProvider(provider);

      // 加载provider数据
      const providerData: any = { ...provider };

      // 如果API key存在且是掩码形式（包含***），则清空字段让用户重新输入
      // 这样用户知道需要重新输入API密钥
      if (providerData.api_key && providerData.api_key.includes('***')) {
        providerData.api_key = ''; // Clear masked API key for re-entry
      }

      providerForm.setFieldsValue(providerData);
      setProviderModalVisible(true);
    },
    [providerForm],
  );

  const handleSaveProvider = useCallback(async () => {
    try {
      const values = await providerForm.validateFields();
      setLoading(true);

      if (editingProvider) {
        await api.llmProvidersProviderNamePut({
          providerName: editingProvider.name,
          llmProviderUpdateWithApiKey: values,
        });
        message.success(formatMessage({ id: 'model.provider.update.success' }));
      } else {
        await api.llmProvidersPost({
          llmProviderCreateWithApiKey: values,
        });
        message.success(formatMessage({ id: 'model.provider.create.success' }));
      }

      providerForm.resetFields(); // Clear form state after successful save
      setProviderModalVisible(false);
      await fetchConfiguration();
    } catch (error) {
      message.error(formatMessage({ id: 'model.provider.save.failed' }));
    } finally {
      setLoading(false);
    }
  }, [
    editingProvider,
    providerForm,
    setLoading,
    fetchConfiguration,
    formatMessage,
  ]);

  const handleDeleteProvider = useCallback(
    async (provider: LlmProvider) => {
      const confirmed = await modal.confirm({
        title: formatMessage({ id: 'action.confirm' }),
        content: formatMessage(
          { id: 'model.provider.delete.confirm' },
          { name: provider.label },
        ),
        okButtonProps: { danger: true },
      });

      if (confirmed) {
        setLoading(true);
        try {
          await api.llmProvidersProviderNameDelete({
            providerName: provider.name,
          });
          message.success(
            formatMessage({ id: 'model.provider.delete.success' }),
          );
          await fetchConfiguration();
        } catch (error) {
          message.error(formatMessage({ id: 'model.provider.delete.failed' }));
        } finally {
          setLoading(false);
        }
      }
    },
    [modal, formatMessage, setLoading, fetchConfiguration],
  );

  // Model management operations
  const handleManageModels = useCallback((provider: LlmProvider) => {
    setCurrentProvider(provider);
    setModalView('list');
    setModelManagementVisible(true);
  }, []);

  const handleAddModel = useCallback(() => {
    setEditingModel(null);
    modelForm.resetFields();
    if (currentProvider) {
      modelForm.setFieldValue('provider_name', currentProvider.name);
      // 默认设置为completion的custom_llm_provider
      modelForm.setFieldValue(
        'custom_llm_provider',
        currentProvider.completion_dialect || 'openai',
      );
    }
    setModalView('add');
  }, [modelForm, currentProvider]);

  const handleEditModel = useCallback(
    (model: LlmProviderModel) => {
      setEditingModel(model);
      modelForm.setFieldsValue(model);
      setModalView('edit');
    },
    [modelForm],
  );

  const handleSaveModel = useCallback(async () => {
    try {
      const values = await modelForm.validateFields();
      setLoading(true);

      // 确保包含provider_name
      const modelData = {
        ...values,
        provider_name: currentProvider?.name,
      };

      if (editingModel) {
        await api.llmProvidersProviderNameModelsApiModelPut({
          providerName: editingModel.provider_name,
          api: editingModel.api as any,
          model: editingModel.model,
          llmProviderModelUpdate: modelData,
        });
        message.success(formatMessage({ id: 'model.update.success' }));
      } else {
        await api.llmProvidersProviderNameModelsPost({
          providerName: currentProvider?.name || '',
          llmProviderModelCreate: modelData,
        });
        message.success(formatMessage({ id: 'model.create.success' }));
      }

      setModalView('list');
      await fetchConfiguration();
    } catch (error) {
      message.error(formatMessage({ id: 'model.save.failed' }));
    } finally {
      setLoading(false);
    }
  }, [
    editingModel,
    modelForm,
    setLoading,
    fetchConfiguration,
    formatMessage,
    currentProvider,
  ]);

  const handleDeleteModel = useCallback(
    async (model: LlmProviderModel) => {
      const confirmed = await modal.confirm({
        title: formatMessage({ id: 'action.confirm' }),
        content: formatMessage(
          { id: 'model.delete.confirm' },
          { model: model.model },
        ),
        okButtonProps: { danger: true },
      });

      if (confirmed) {
        setLoading(true);
        try {
          await api.llmProvidersProviderNameModelsApiModelDelete({
            providerName: model.provider_name,
            api: model.api as any,
            model: model.model,
          });
          message.success(formatMessage({ id: 'model.delete.success' }));
          await fetchConfiguration();
        } catch (error) {
          message.error(formatMessage({ id: 'model.delete.failed' }));
        } finally {
          setLoading(false);
        }
      }
    },
    [modal, formatMessage, setLoading, fetchConfiguration],
  );

  const handleBackToList = useCallback(() => {
    setModalView('list');
    setEditingModel(null);
    // 重置表单但保留provider_name
    const providerName = modelForm.getFieldValue('provider_name');
    modelForm.resetFields();
    if (providerName) {
      modelForm.setFieldValue('provider_name', providerName);
    }
  }, [modelForm]);

  const handleCloseModelManagement = useCallback(() => {
    setModelManagementVisible(false);
    setModalView('list');
    setCurrentProvider(null);
    setEditingModel(null);
    setModelSearchText(''); // Reset search state
    modelForm.resetFields();
  }, [modelForm]);

  // Handle API type change to auto-fill custom_llm_provider
  const handleApiTypeChange = useCallback(
    (api: string) => {
      if (currentProvider) {
        let dialect = '';
        switch (api) {
          case 'completion':
            dialect = currentProvider.completion_dialect || 'openai';
            break;
          case 'embedding':
            dialect = currentProvider.embedding_dialect || 'openai';
            break;
          case 'rerank':
            dialect = currentProvider.rerank_dialect || 'jina_ai';
            break;
        }
        modelForm.setFieldValue('custom_llm_provider', dialect);
      }
    },
    [modelForm, currentProvider],
  );

  // Get models for current provider with search filter
  const getCurrentProviderModels = useCallback(() => {
    if (!currentProvider) return [];
    let models = configuration.models.filter(
      (model) => model.provider_name === currentProvider.name,
    );

    // Apply search filter if search text exists
    if (modelSearchText.trim()) {
      models = models.filter((model) =>
        model.model.toLowerCase().includes(modelSearchText.toLowerCase()),
      );
    }

    return models;
  }, [configuration.models, currentProvider, modelSearchText]);

  // Get model count for provider
  const getProviderModelCount = useCallback(
    (providerName: string) => {
      return configuration.models.filter(
        (model) => model.provider_name === providerName,
      ).length;
    },
    [configuration.models],
  );

  // Provider table columns
  const providerColumns: TableProps<LlmProvider>['columns'] = [
    {
      title: formatMessage({ id: 'model.provider.name' }),
      dataIndex: 'label',
      key: 'label',
      width: 180,
      render: (label: string) => <Text strong>{label}</Text>,
    },
    {
      title: formatMessage({ id: 'model.provider.url' }),
      dataIndex: 'base_url',
      key: 'base_url',
      width: 280,
      ellipsis: {
        showTitle: false,
      },
      render: (url: string) => (
        <Tooltip title={url}>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {url}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: formatMessage({ id: 'model.provider.api_key_short' }),
      key: 'api_key_status',
      width: 180,
      ellipsis: {
        showTitle: false,
      },
      render: (_, record) => {
        const apiKey =
          record.api_key && record.api_key.trim() !== ''
            ? record.api_key
            : null;
        return apiKey ? (
          <Tooltip title={apiKey}>
            <Text
              style={{
                fontSize: '12px',
                fontFamily: 'monospace',
                maxWidth: '140px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {apiKey}
            </Text>
          </Tooltip>
        ) : (
          <Text type="secondary" style={{ fontSize: '12px' }}>
            -
          </Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'model.provider.model_count' }),
      key: 'model_count',
      align: 'center',
      width: 100,
      render: (_, record) => {
        const count = getProviderModelCount(record.name);
        return (
          <Text
            style={{
              fontSize: '13px',
              fontWeight: 500,
              color: count > 0 ? '#1890ff' : '#8c8c8c',
            }}
          >
            {count}
          </Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'action.name' }),
      key: 'actions',
      width: 160,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title={formatMessage({ id: 'model.provider.manage' })}>
            <Button
              type="text"
              icon={<SettingOutlined />}
              onClick={() => handleManageModels(record)}
            />
          </Tooltip>
          <Tooltip title={formatMessage({ id: 'model.provider.edit' })}>
            <Button
              type="text"
              icon={<EditOutlined />}
              onClick={() => handleEditProvider(record)}
            />
          </Tooltip>
          <Tooltip title={formatMessage({ id: 'model.provider.delete' })}>
            <Button
              type="text"
              icon={<DeleteOutlined />}
              danger
              onClick={() => handleDeleteProvider(record)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // Model table columns
  const modelColumns: TableProps<LlmProviderModel>['columns'] = [
    {
      title: formatMessage({ id: 'model.field.model' }),
      dataIndex: 'model',
      key: 'model',
      render: (text) => <Text style={{ maxWidth: 200 }} ellipsis={{ tooltip: text }}>{text}</Text>,
    },
    {
      title: formatMessage({ id: 'model.field.api' }),
      dataIndex: 'api',
      key: 'api',
    },
    {
      title: formatMessage({ id: 'model.field.contextWindow' }),
      dataIndex: 'context_window',
      key: 'context_window',
      render: (tokens) => tokens || '-',
    },
    {
      title: formatMessage({ id: 'model.field.maxInput' }),
      dataIndex: 'max_input_tokens',
      key: 'max_input_tokens',
      render: (tokens) => tokens || '-',
    },
    {
      title: formatMessage({ id: 'model.field.maxOutput' }),
      dataIndex: 'max_output_tokens',
      key: 'max_output_tokens',
      render: (tokens) => tokens || '-',
    },
    {
      title: formatMessage({ id: 'model.field.tags' }),
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <Space wrap>
          {tags?.map((tag) => (
            <Tag key={tag}>{tag}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: formatMessage({ id: 'model.field.action' }),
      key: 'action',
      render: (_, record) => (
        <Space>
          <Tooltip title={formatMessage({ id: 'action.edit' })}>
            <Button
              icon={<EditOutlined />}
              onClick={() => handleEditModel(record)}
            />
          </Tooltip>
          <Tooltip title={formatMessage({ id: 'action.delete' })}>
            <Button
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteModel(record)}
              danger
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  useEffect(() => {
    fetchConfiguration();
  }, [fetchConfiguration]);

  return (
    <PageContainer>
      {contextHolder}
      <PageHeader
        title={formatMessage({ id: 'model.configuration' })}
        description={formatMessage({ id: 'model.configuration.description' })}
      >
        <Button type="primary" onClick={handleCreateProvider}>
          {formatMessage({ id: 'model.provider.add' })}
        </Button>
        <RefreshButton onClick={() => fetchConfiguration()} loading={loading} />
      </PageHeader>
      <Table
        columns={providerColumns}
        dataSource={configuration.providers}
        rowKey="name"
        loading={loading}
        pagination={false}
        scroll={{ x: 1000 }}
        bordered
      />
      <Modal
        title={
          <Space>
            <SettingOutlined />
            <span>
              {editingProvider
                ? formatMessage({ id: 'model.provider.edit' })
                : formatMessage({ id: 'model.provider.add' })}
            </span>
          </Space>
        }
        open={providerModalVisible}
        onCancel={() => {
          providerForm.resetFields(); // Clear form state when closing modal
          setProviderModalVisible(false);
        }}
        onOk={handleSaveProvider}
        width={650}
        confirmLoading={loading}
        maskClosable={false}
        style={{ top: 100 }}
      >
        <Divider style={{ margin: '16px 0 24px 0' }} />
        <Form
          form={providerForm}
          layout="vertical"
          requiredMark={false}
          colon={false}
        >
          <Form.Item
            name="label"
            label={formatMessage({ id: 'model.provider.label' })}
            rules={[
              {
                required: true,
                message: formatMessage({
                  id: 'model.provider.label.required',
                }),
              },
            ]}
          >
            <Input
              placeholder={formatMessage({
                id: 'model.provider.label.placeholder',
              })}
              style={{ borderRadius: '6px' }}
            />
          </Form.Item>

          <Form.Item
            name="base_url"
            label={formatMessage({ id: 'model.provider.base_url' })}
            rules={[
              {
                required: true,
                message: formatMessage({
                  id: 'model.provider.base_url.required',
                }),
              },
            ]}
          >
            <Input
              placeholder={formatMessage({
                id: 'model.provider.base_url.placeholder',
              })}
              style={{ borderRadius: '6px' }}
            />
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="completion_dialect"
                label={formatMessage({
                  id: 'model.provider.completion_dialect',
                })}
                rules={[
                  {
                    required: true,
                    message: formatMessage({
                      id: 'model.provider.completion_dialect.required',
                    }),
                  },
                ]}
              >
                <Input
                  placeholder={formatMessage({
                    id: 'model.provider.completion_dialect.placeholder',
                  })}
                  style={{ borderRadius: '6px' }}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="embedding_dialect"
                label={formatMessage({
                  id: 'model.provider.embedding_dialect',
                })}
                rules={[
                  {
                    required: true,
                    message: formatMessage({
                      id: 'model.provider.embedding_dialect.required',
                    }),
                  },
                ]}
              >
                <Input
                  placeholder={formatMessage({
                    id: 'model.provider.embedding_dialect.placeholder',
                  })}
                  style={{ borderRadius: '6px' }}
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="rerank_dialect"
                label={formatMessage({ id: 'model.provider.rerank_dialect' })}
                rules={[
                  {
                    required: true,
                    message: formatMessage({
                      id: 'model.provider.rerank_dialect.required',
                    }),
                  },
                ]}
              >
                <Input
                  placeholder={formatMessage({
                    id: 'model.provider.rerank_dialect.placeholder',
                  })}
                  style={{ borderRadius: '6px' }}
                />
              </Form.Item>
            </Col>
          </Row>

          <Divider orientation="left">
            {formatMessage({ id: 'model.provider.api_key.settings' })}
          </Divider>

          <Form.Item
            name="api_key"
            label={formatMessage({ id: 'model.provider.api_key' })}
            help={
              editingProvider
                ? editingProvider.api_key
                  ? formatMessage({ id: 'model.provider.api_key.edit.help' })
                  : formatMessage({ id: 'model.provider.api_key.help' })
                : formatMessage({ id: 'model.provider.api_key.help' })
            }
          >
            <Input
              placeholder={
                editingProvider && editingProvider.api_key
                  ? formatMessage({
                      id: 'model.provider.api_key.edit.placeholder',
                    })
                  : formatMessage({ id: 'model.provider.api_key.placeholder' })
              }
              autoComplete="off"
              spellCheck={false}
              style={{ borderRadius: '6px', fontFamily: 'monospace' }}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Model Management Modal */}
      <Modal
        title={
          currentProvider
            ? formatMessage(
                { id: 'model.management.title' },
                { provider: currentProvider.label },
              )
            : formatMessage({ id: 'model.management.title' }, { provider: '' })
        }
        open={modelManagementVisible}
        onCancel={handleCloseModelManagement}
        footer={null}
        width={1000}
        destroyOnClose
        maskClosable={false}
        keyboard={true} // 支持ESC退出
        style={{ top: 60 }}
      >
        <Divider style={{ margin: '16px 0 24px 0' }} />

        {modalView === 'list' && (
          <div>
            <div
              style={{
                marginBottom: 20,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '0 4px',
                gap: 16,
              }}
            >
              <Title level={5} style={{ margin: 0, fontWeight: 600 }}>
                {formatMessage({ id: 'model.list.title' })}
              </Title>
              <div style={{ flex: 1, maxWidth: 300 }}>
                <Input
                  placeholder={formatMessage({
                    id: 'model.search.placeholder',
                  })}
                  prefix={<SearchOutlined style={{ color: '#bfbfbf' }} />}
                  value={modelSearchText}
                  onChange={(e) => setModelSearchText(e.target.value)}
                  allowClear
                  style={{ borderRadius: '6px' }}
                />
              </div>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleAddModel}
                style={{ fontWeight: 500 }}
              >
                {formatMessage({ id: 'model.add' })}
              </Button>
            </div>
            <Table
              columns={modelColumns}
              dataSource={getCurrentProviderModels()}
              rowKey={(record) => `${record.api}-${record.model}`}
              loading={loading}
              // pagination={false}
              size="small"
              scroll={{ x: 700 }}
              style={{
                fontSize: '13px',
                border: '1px solid #f0f0f0',
                borderRadius: '6px',
              }}
            />
          </div>
        )}

        {(modalView === 'add' || modalView === 'edit') && (
          <div>
            <div
              style={{
                marginBottom: 24,
                display: 'flex',
                alignItems: 'center',
                padding: '12px 16px',
                background: '#fafafa',
                borderRadius: '6px',
                border: '1px solid #f0f0f0',
              }}
            >
              <Button
                type="text"
                icon={<ArrowLeftOutlined />}
                onClick={handleBackToList}
                style={{ marginRight: 12, fontWeight: 500 }}
              >
                {formatMessage({ id: 'model.back_to_list' })}
              </Button>
              <Title level={5} style={{ margin: 0, fontWeight: 600 }}>
                {modalView === 'add'
                  ? formatMessage({ id: 'model.add.title' })
                  : formatMessage(
                      { id: 'model.edit.title' },
                      { model: editingModel?.model },
                    )}
              </Title>
            </div>

            <Form
              form={modelForm}
              layout="vertical"
              onFinish={handleSaveModel}
              requiredMark={false}
              colon={false}
              style={{ padding: '0 4px' }}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="api"
                    label={formatMessage({ id: 'model.api_type' })}
                    rules={[
                      {
                        required: true,
                        message: formatMessage({
                          id: 'model.api_type.required',
                        }),
                      },
                    ]}
                  >
                    <Select
                      placeholder={formatMessage({
                        id: 'model.api_type.required',
                      })}
                      disabled={!!editingModel}
                      onChange={handleApiTypeChange}
                      style={{ borderRadius: '6px' }}
                    >
                      <Select.Option value="completion">
                        {formatMessage({ id: 'model.api_type.completion' })}
                      </Select.Option>
                      <Select.Option value="embedding">
                        {formatMessage({ id: 'model.api_type.embedding' })}
                      </Select.Option>
                      <Select.Option value="rerank">
                        {formatMessage({ id: 'model.api_type.rerank' })}
                      </Select.Option>
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="model"
                    label={formatMessage({ id: 'model.model_name' })}
                    rules={[
                      {
                        required: true,
                        message: formatMessage({
                          id: 'model.model_name.required',
                        }),
                      },
                    ]}
                  >
                    <Input
                      placeholder={formatMessage({
                        id: 'model.model_name.placeholder',
                      })}
                      disabled={!!editingModel}
                      style={{ borderRadius: '6px' }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="custom_llm_provider"
                    label={formatMessage({ id: 'model.field.custom_llm_provider' })}
                    rules={[
                      {
                        required: true,
                        message: formatMessage({
                          id: 'model.field.custom_llm_provider.required',
                        }),
                      },
                    ]}
                  >
                    <Input
                      placeholder={formatMessage({
                        id: 'model.field.custom_llm_provider.placeholder',
                      })}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label={formatMessage({ id: 'model.field.contextWindow' })}
                    name="context_window"
                    tooltip={formatMessage({ id: 'model.field.contextWindow.tooltip' })}
                  >
                    <InputNumber style={{ width: '100%' }} placeholder="e.g. 128000" />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label={formatMessage({ id: 'model.field.maxInput' })}
                    name="max_input_tokens"
                    tooltip={formatMessage({ id: 'model.field.maxInput.tooltip' })}
                  >
                    <InputNumber style={{ width: '100%' }} placeholder="e.g. 128000" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label={formatMessage({ id: 'model.field.maxOutput' })}
                    name="max_output_tokens"
                    tooltip={formatMessage({ id: 'model.field.maxOutput.tooltip' })}
                  >
                    <InputNumber style={{ width: '100%' }} placeholder="e.g. 4096" />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item
                label={formatMessage({ id: 'model.field.tags' })}
                name="tags"
                tooltip={formatMessage({ id: 'model.field.tags.tooltip' })}
              >
                <Select
                  mode="tags"
                  placeholder={formatMessage({
                    id: 'model.tags.placeholder',
                  })}
                  tokenSeparators={[',']}
                  style={{ width: '100%', borderRadius: '6px' }}
                />
              </Form.Item>

              <div
                style={{
                  marginTop: 32,
                  padding: '16px 0 8px 0',
                  borderTop: '1px solid #f0f0f0',
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: 12,
                }}
              >
                <Button
                  onClick={handleBackToList}
                  style={{ borderRadius: '6px' }}
                >
                  {formatMessage({ id: 'action.cancel' })}
                </Button>
                <Button
                  type="primary"
                  htmlType="submit"
                  loading={loading}
                  style={{ borderRadius: '6px', fontWeight: 500 }}
                >
                  {formatMessage({ id: 'action.save' })}
                </Button>
              </div>
            </Form>
          </div>
        )}
      </Modal>
    </PageContainer>
  );
};
