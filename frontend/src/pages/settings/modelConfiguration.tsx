import {
  LlmConfigurationResponse,
  LlmProvider,
  LlmProviderModel,
  LlmProviderModelCreate,
  LlmProviderModelUpdate,
} from '@/api';
import {
  LlmProvidersProviderNameModelsApiModelDeleteApiEnum,
  LlmProvidersProviderNameModelsApiModelPutApiEnum,
} from '@/api/apis/default-api';
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
  Switch,
  Table,
  TableProps,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl, useModel } from 'umi';

const { Title, Text } = Typography;

// 弹窗内容视图类型
type ModalViewType = 'list' | 'add' | 'edit';

// API Key配置相关类型 - 已移除，API Key现在直接在LLM Provider中管理

export default () => {
  const { loading, setLoading } = useModel('global');
  const { user } = useModel('user');
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

  // Provider status state
  const [providerStatus, setProviderStatus] = useState<'enable' | 'disable'>(
    'enable',
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

  // Check if provider is enabled (has API key)
  const isProviderEnabled = useCallback((provider: LlmProvider) => {
    return provider.api_key && provider.api_key.trim() !== '';
  }, []);

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
    setProviderStatus('enable');
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

      // Set provider status based on API key presence
      const enabled = isProviderEnabled(provider);
      setProviderStatus(enabled ? 'enable' : 'disable');

      // 加载provider数据
      const providerData: any = { ...provider };

      // 无论原有key是否mask，编辑时都清空输入框
      providerData.api_key = '';

      providerForm.setFieldsValue(providerData);
      setProviderModalVisible(true);
    },
    [providerForm, isProviderEnabled],
  );

  const handleSaveProvider = useCallback(async () => {
    try {
      const values = await providerForm.validateFields();
      setLoading(true);

      // Add status to the request
      const requestData = {
        ...values,
        status: providerStatus,
      };

      if (editingProvider) {
        await api.llmProvidersProviderNamePut({
          providerName: editingProvider.name,
          llmProviderUpdateWithApiKey: requestData,
        });
        message.success(formatMessage({ id: 'model.provider.update.success' }));
      } else {
        await api.llmProvidersPost({
          llmProviderCreateWithApiKey: requestData,
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
    providerStatus,
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

  // Handle provider status toggle
  const handleProviderStatusChange = useCallback(
    (enabled: boolean) => {
      const newStatus = enabled ? 'enable' : 'disable';
      setProviderStatus(newStatus);

      // If disabling, clear the API key field
      if (newStatus === 'disable') {
        providerForm.setFieldValue('api_key', '');
      }
    },
    [providerForm],
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
      modelForm.setFieldsValue({
        ...model,
      });
      setModalView('edit');
    },
    [modelForm],
  );

  const handleSaveModel = useCallback(
    async (values: any) => {
      try {
        setLoading(true);

        if (editingModel) {
          // Update existing model - 保持原有的tags和token限制字段不变
          const finalData = {
            ...values,
            tags: editingModel.tags, // 保持原有标签
            max_input_tokens: editingModel.max_input_tokens, // 保持原有最大输入
            max_output_tokens: editingModel.max_output_tokens, // 保持原有最大输出
          };

          await api.llmProvidersProviderNameModelsApiModelPut({
            providerName: editingModel.provider_name,
            api: editingModel.api as LlmProvidersProviderNameModelsApiModelPutApiEnum,
            model: editingModel.model,
            llmProviderModelUpdate: finalData as LlmProviderModelUpdate,
          });
          message.success(formatMessage({ id: 'model.update.success' }));
        } else {
          // Create new model - 新建模型时设置默认值
          const modelCreateData: LlmProviderModelCreate = {
            ...values,
            provider_name: currentProvider!.name,
            tags: [], // 新建模型初始无标签
            max_input_tokens: null, // 可选字段设为null
            max_output_tokens: null, // 可选字段设为null
          };
          await api.llmProvidersProviderNameModelsPost({
            providerName: currentProvider!.name,
            llmProviderModelCreate: modelCreateData,
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
    },
    [
      editingModel,
      currentProvider,
      setLoading,
      fetchConfiguration,
      formatMessage,
    ],
  );

  const handleDeleteModel = useCallback(
    async (model: LlmProviderModel) => {
      const confirmed = await modal.confirm({
        title: formatMessage({ id: 'action.confirm' }),
        content: formatMessage(
          { id: 'model.delete.confirm' },
          { name: model.model },
        ),
        okButtonProps: { danger: true },
      });

      if (confirmed) {
        setLoading(true);
        try {
          await api.llmProvidersProviderNameModelsApiModelDelete({
            providerName: model.provider_name,
            api: model.api as LlmProvidersProviderNameModelsApiModelDeleteApiEnum,
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

  const handleUseCaseChange = useCallback(
    async (model: LlmProviderModel, tag: string, checked: boolean) => {
      try {
        setLoading(true);

        // 更新tags数组
        const currentTags = model.tags || [];
        let newTags: string[];

        if (checked) {
          // 添加标签
          newTags = currentTags.includes(tag)
            ? currentTags
            : [...currentTags, tag];
        } else {
          // 移除标签
          newTags = currentTags.filter((t) => t !== tag);
        }

        // 调用API更新模型
        await api.llmProvidersProviderNameModelsApiModelPut({
          providerName: model.provider_name,
          api: model.api as LlmProvidersProviderNameModelsApiModelPutApiEnum,
          model: model.model,
          llmProviderModelUpdate: {
            ...model,
            tags: newTags,
          } as LlmProviderModelUpdate,
        });

        message.success(formatMessage({ id: 'model.useCase.update.success' }));
        await fetchConfiguration();
      } catch (error) {
        message.error(formatMessage({ id: 'model.useCase.update.failed' }));
      } finally {
        setLoading(false);
      }
    },
    [setLoading, fetchConfiguration, formatMessage],
  );

  const handleBackToList = useCallback(() => {
    setModalView('list');
    setEditingModel(null);
    modelForm.resetFields();
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
  const providerColumns: TableProps<LlmProvider>['columns'] = useMemo(() => {
    const baseColumns: TableProps<LlmProvider>['columns'] = [
      {
        title: formatMessage({ id: 'model.provider.name' }),
        dataIndex: 'label',
        key: 'label',
        render: (text: string) => <span>{text}</span>,
      },
    ];

    // Add provider type column only for admin users
    if (user?.role === 'admin') {
      baseColumns.push({
        title: formatMessage({ id: 'model.provider.type' }),
        dataIndex: 'user_id',
        key: 'user_id',
        render: (user_id: string) => (
          <Tag color={user_id === 'public' ? 'blue' : 'green'}>
            {user_id === 'public'
              ? formatMessage({ id: 'model.provider.type.public' })
              : formatMessage({ id: 'model.provider.type.user' })}
          </Tag>
        ),
      });
    }

    // Add status and other columns
    baseColumns.push(
      {
        title: formatMessage({ id: 'common.status' }),
        key: 'status',
        dataIndex: '', // Add empty dataIndex to satisfy type requirement
        render: (_, record: LlmProvider) => {
          const enabled = isProviderEnabled(record);
          return (
            <Tag color={enabled ? 'green' : 'default'}>
              {enabled
                ? formatMessage({ id: 'common.enabled' })
                : formatMessage({ id: 'common.disabled' })}
            </Tag>
          );
        },
      },
      {
        title: 'URL',
        dataIndex: 'base_url',
        key: 'base_url',
        render: (url: string) => (
          <Tooltip title={url}>
            <span
              style={{
                maxWidth: '280px',
                display: 'inline-block',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {url}
            </span>
          </Tooltip>
        ),
      },
      {
        title: formatMessage({ id: 'model.provider.model_count' }),
        key: 'model_count',
        dataIndex: '', // Add empty dataIndex to satisfy type requirement
        render: (_: any, record: LlmProvider) => (
          <span>{getProviderModelCount(record.name)}</span>
        ),
      },
      {
        title: formatMessage({ id: 'common.actions' }),
        dataIndex: 'action',
        key: 'action',
        render: (_, record: LlmProvider) => (
          <Space size="middle">
            <Tooltip title={formatMessage({ id: 'model.provider.manage' })}>
              <Button
                icon={<SettingOutlined />}
                size="small"
                onClick={() => handleManageModels(record)}
              />
            </Tooltip>
            <Tooltip title={formatMessage({ id: 'action.edit' })}>
              <Button
                icon={<EditOutlined />}
                size="small"
                onClick={() => handleEditProvider(record)}
              />
            </Tooltip>
            <Tooltip title={formatMessage({ id: 'action.delete' })}>
              <Button
                icon={<DeleteOutlined />}
                size="small"
                danger
                onClick={() => handleDeleteProvider(record)}
              />
            </Tooltip>
          </Space>
        ),
      },
    );

    return baseColumns;
  }, [
    formatMessage,
    user?.role,
    isProviderEnabled,
    getProviderModelCount,
    handleManageModels,
    handleEditProvider,
    handleDeleteProvider,
  ]);

  // Model table columns
  const modelColumns: TableProps<LlmProviderModel>['columns'] = [
    {
      title: formatMessage({ id: 'model.field.model' }),
      dataIndex: 'model',
      key: 'model',
      render: (text) => (
        <Text style={{ maxWidth: 200 }} ellipsis={{ tooltip: text }}>
          {text}
        </Text>
      ),
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
      title: formatMessage({ id: 'model.field.useCases' }),
      key: 'useCases',
      width: 160,
      render: (_, record: LlmProviderModel) => (
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Text style={{ fontSize: '12px' }}>
              {formatMessage({ id: 'model.usecase.collection' })}
            </Text>
            <Switch
              size="small"
              checked={record.tags?.includes('enable_for_collection') || false}
              onChange={(checked) =>
                handleUseCaseChange(record, 'enable_for_collection', checked)
              }
            />
          </div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Text style={{ fontSize: '12px' }}>
              {formatMessage({ id: 'model.usecase.agent' })}
            </Text>
            <Switch
              size="small"
              checked={record.tags?.includes('enable_for_agent') || false}
              onChange={(checked) =>
                handleUseCaseChange(record, 'enable_for_agent', checked)
              }
            />
          </div>
        </Space>
      ),
    },
    {
      title: formatMessage({ id: 'model.field.tags' }),
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <Space wrap>
          {tags
            ?.filter((tag) => tag !== '__autogen__')
            .map((tag) => <Tag key={tag}>{tag}</Tag>)}
        </Space>
      ),
    },
    {
      title: formatMessage({ id: 'model.field.action' }),
      key: 'action',
      render: (_, record: LlmProviderModel) => (
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
            label={formatMessage({ id: 'model.provider.status' })}
            style={{ marginBottom: '16px' }}
          >
            <Space align="center">
              <Switch
                checked={providerStatus === 'enable'}
                onChange={handleProviderStatusChange}
                checkedChildren={formatMessage({ id: 'model.provider.enable' })}
                unCheckedChildren={formatMessage({
                  id: 'model.provider.disable',
                })}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {providerStatus === 'enable'
                  ? formatMessage({ id: 'model.provider.enable.help' })
                  : formatMessage({ id: 'model.provider.disable.help' })}
              </Text>
            </Space>
          </Form.Item>

          {providerStatus === 'enable' && (
            <Form.Item
              name="api_key"
              label={formatMessage({ id: 'model.provider.api_key' })}
              rules={
                !editingProvider
                  ? [
                      {
                        required: true,
                        message: formatMessage({
                          id: 'model.provider.api_key.required',
                        }),
                      },
                    ]
                  : !editingProvider.api_key
                    ? [
                        {
                          required: true,
                          message: formatMessage({
                            id: 'model.provider.api_key.required',
                          }),
                        },
                      ]
                    : []
              }
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
                    : formatMessage({
                        id: 'model.provider.api_key.placeholder',
                      })
                }
                autoComplete="off"
                spellCheck={false}
                style={{ borderRadius: '6px', fontFamily: 'monospace' }}
              />
            </Form.Item>
          )}
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
              pagination={{ pageSize: 100 }}
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
                marginBottom: 20,
                display: 'flex',
                alignItems: 'center',
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
                    label={formatMessage({
                      id: 'model.field.custom_llm_provider',
                    })}
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
                    tooltip={formatMessage({
                      id: 'model.field.contextWindow.tooltip',
                    })}
                  >
                    <InputNumber
                      style={{ width: '100%' }}
                      placeholder="e.g. 128000"
                    />
                  </Form.Item>
                </Col>
              </Row>

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
