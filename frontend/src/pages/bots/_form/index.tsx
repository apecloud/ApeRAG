import { CollectionStatusEnum, AvailableModel, SupportedModelServiceProvider } from '@/api';
import iconCommon from '@/assets/bots/common.svg';
import iconKnowledge from '@/assets/bots/knowledge.svg';
import { CheckCard, RefreshButton } from '@/components';
import {
  MODEL_PROVIDER_ICON,
  UI_COLLECTION_STATUS,
} from '@/constants';
import { ApeBot } from '@/types';
import { api } from '@/services';
import {
  Avatar,
  Badge,
  Button,
  Col,
  Divider,
  Form,
  FormInstance,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Switch,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import { useEffect, useMemo, useState } from 'react';
import { FormattedMessage, Link, useIntl, useModel } from 'umi';

type Props = {
  action: 'add' | 'edit';
  values: ApeBot;
  form: FormInstance<ApeBot>;
  onSubmit: (data: ApeBot) => void;
};

export default ({ form, onSubmit, values, action }: Props) => {
  const { formatMessage } = useIntl();
  const { collections, collectionsLoading, getCollections } =
    useModel('collection');
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>();
  const [supportedModelServiceProviders, setSupportedModelServiceProviders] =
      useState<SupportedModelServiceProvider[]>();
  const { models, promptTemplates, getPromptTemplates } = useModel('models');
  const { loading, setLoading } = useModel('global');

  const { token } = theme.useToken();

  const onFinish = async () => {
    const data = await form.validateFields();
    onSubmit(data);
  };

  const modelsOptions = useMemo(
    () =>
      _.map(
        _.groupBy(availableModels, 'model_service_provider'),
        (aebs, providerName) => {
          const provider = supportedModelServiceProviders?.find(
            (smp) => smp.name === providerName,
          );
          return {
            label: (
              <Space>
                <Avatar
                  size={24}
                  shape="square"
                  src={MODEL_PROVIDER_ICON[providerName]}
                />
                <span>{provider?.label || providerName}</span>
              </Space>
            ),
            options: aebs.map((aeb) => {
              return {
                label: aeb.model_name,
                value: `${providerName}:${aeb.model_name}`,
              };
            }),
          };
        },
      ),
    [availableModels, supportedModelServiceProviders],
  );

  const botType = Form.useWatch(['type'], form);
  const botChractor = Form.useWatch(['config', 'chractor'], form);
  const botModel = Form.useWatch(['config', 'model'], form);
  const botMemory = Form.useWatch(['config', 'memory'], form);
  const botUseRelatedQuestion = Form.useWatch(
    ['config', 'use_related_question'],
    form,
  );

  const getModels = async () => {
    setLoading(true);
    const [
      availableModelsRes,
      supportedModelServiceProvidersRes,
    ] = await Promise.all([
      api.availableModelsGet(),
      api.supportedModelServiceProvidersGet(),
    ]);
    setLoading(false);
    setAvailableModels(availableModelsRes.data.items);
    setSupportedModelServiceProviders(
      supportedModelServiceProvidersRes.data.items,
    );
  };

  const currentModel = useMemo( () => {
    return models?.find((m) => `${m.model_service_provider}:${m.value}` === botModel)
  }, [botModel, models]);

  const currentPromptTemplate = useMemo(() => {
    return promptTemplates?.find((m) => m.prompt === botChractor);
  }, [botChractor, promptTemplates]);

  useEffect(() => {
    const config = values.config;
    let memory = Boolean(config?.memory);

    const llm = config?.llm;
    let template = llm?.prompt_template;

    let contextWindow = _.isNumber(llm?.context_window)
      ? llm?.context_window
      : 3500;
    let temperature = _.isNumber(llm?.temperature) ? llm?.temperature : 0.3;
    let similarityTopk = _.isNumber(llm?.similarity_topk)
      ? llm?.similarity_topk
      : 3;
    let similarityScoreThreshold = _.isNumber(llm?.similarity_score_threshold)
      ? llm?.similarity_score_threshold
      : 0.5;

    if (botModel !== config?.model) {
      memory = false;
      contextWindow = _.isNumber(currentModel?.context_window)
        ? currentModel?.context_window
        : 3500;
      temperature = _.isNumber(currentModel?.temperature)
        ? currentModel?.temperature
        : 0.3;
      similarityTopk = _.isNumber(currentModel?.similarity_topk)
        ? currentModel?.similarity_topk
        : 3;
      similarityScoreThreshold = _.isNumber(
        currentModel?.similarity_score_threshold,
      )
        ? currentModel?.similarity_score_threshold
        : 0.5;
    }

    if (botType === 'knowledge' && botModel !== config?.model) {
      template = currentModel?.prompt_template;
    }
    if (botType === 'common' && botChractor !== config?.charactor) {
      template = currentPromptTemplate?.prompt;
    }

    form.setFieldValue(['config', 'memory'], memory);
    form.setFieldValue(['config', 'llm', 'prompt_template'], template);

    form.setFieldValue(['config', 'llm', 'context_window'], contextWindow);
    form.setFieldValue(['config', 'llm', 'temperature'], temperature);
    form.setFieldValue(['config', 'llm', 'similarity_topk'], similarityTopk);
    form.setFieldValue(
      ['config', 'llm', 'similarity_score_threshold'],
      similarityScoreThreshold,
    );
  }, [botType, currentModel, currentPromptTemplate, values]);

  useEffect(() => {
    const config = values.config;
    if (_.isEmpty(config?.model)) {
      form.setFieldValue(['config', 'model'], `${models?.[0]?.model_service_provider}:${models?.[0]?.value}`);
    }
  }, [models]);

  useEffect(() => {
    const defaultCollection = collections?.find((c) => c.status === 'ACTIVE');
    if (_.isEmpty(values.collection_ids) && !_.isEmpty(defaultCollection)) {
      form.setFieldValue(['collection_ids'], [defaultCollection.id]);
    }
  }, [collections]);

  useEffect(() => {
    if (botModel) {
      const [model_service_provider, model_name] =
      botModel.split(':');
      console.log("model_service_provider", model_service_provider)
      console.log("model_name", model_name)
      form.setFieldValue(
        ['config', 'model_service_provider'],
        model_service_provider,
      );
      form.setFieldValue(['config', 'model_name'], model_name);
    }
  }, [botModel]);

  useEffect(() => {
    getCollections();
    getPromptTemplates();
    getModels();
  }, []);

  return (
    <>
      <Form
        autoComplete="off"
        onFinish={onFinish}
        layout="vertical"
        form={form}
        initialValues={values}
      >
        <Form.Item
          name="title"
          label={<FormattedMessage id="text.title" />}
          rules={[
            {
              required: true,
              message: <FormattedMessage id="text.title.required" />,
            },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="description"
          label={<FormattedMessage id="text.description" />}
        >
          <Input.TextArea />
        </Form.Item>

        <Form.Item
          label={<FormattedMessage id="bot.type" />}
          name="type"
          valuePropName="value"
          required
        >
          <CheckCard
            options={[
              {
                label: <FormattedMessage id="bot.type_knowledge" />,
                icon: iconKnowledge,
                value: 'knowledge',
              },
              {
                label: <FormattedMessage id="bot.type_common" />,
                icon: iconCommon,
                value: 'common',
              },
            ]}
          />
        </Form.Item>
        <Row gutter={24}>
          <Col
            {...{
              xs: 24,
              sm: 24,
              md: 12,
              lg: 12,
              xl: 12,
              xxl: 12,
            }}
          >
            <Form.Item
              name={['config', 'model']}
              label={<FormattedMessage id="model.name" />}
              rules={[
                {
                  required: true,
                  message: <FormattedMessage id="model.required" />,
                },
              ]}
            >
              <Select
                options={modelsOptions}
                labelRender={({ label, value }) => {
                  const [model_service_provider, model_name] = (
                    value as string
                  ).split(':');
                  return (
                    <Space style={{ alignItems: 'center' }}>
                      <Avatar
                        size={24}
                        shape="square"
                        src={MODEL_PROVIDER_ICON[model_service_provider]}
                        style={{
                          transform: 'translateY(-1px)',
                        }}
                      />
                      {label || model_name}
                    </Space>
                  );
                }}
              />
            </Form.Item>

          </Col>
          <Col
            {...{
              xs: 24,
              sm: 24,
              md: 12,
              lg: 12,
              xl: 12,
              xxl: 12,
            }}
          >
            {botType === 'knowledge' ? (
              <Form.Item
                label={<FormattedMessage id="collection.name" />}
                required
                name="collection_ids"
                rules={[
                  {
                    required: true,
                    message: formatMessage({ id: 'bot.colelction_required' }),
                  },
                ]}
              >
                <Select
                  mode="multiple"
                  dropdownRender={(originNode) => {
                    return (
                      <div>
                        {originNode}
                        <Divider style={{ marginBlock: 4 }} />
                        <div style={{ textAlign: 'center', paddingBlock: 4 }}>
                          <Link to="/collections/new" target="_blank">
                            <FormattedMessage id="collection.add" />
                          </Link>
                        </div>
                      </div>
                    );
                  }}
                  suffixIcon={
                    <RefreshButton
                      loading={collectionsLoading}
                      shape="circle"
                      size="small"
                      onClick={() => getCollections()}
                      style={{ transform: 'translateX(8px)' }}
                    />
                  }
                  options={collections?.map((collection) => ({
                    ...collection,
                    label: collection.title,
                    value: collection.id,
                    disabled: collection.status !== 'ACTIVE',
                  }))}
                  optionRender={(option) => {
                    const config = option.data.config;
                    const status = option.data.status as CollectionStatusEnum;
                    const isActive = status === 'ACTIVE';
                    const embedding_model_service_provider =
                      config?.embedding_model_service_provider || '';
                    const embedding_model_name = config?.embedding_model_name;
                    return (
                      <Tooltip
                        placement="left"
                        title={
                          !isActive
                            ? formatMessage({
                                id: `collection.status.${status}`,
                              })
                            : undefined
                        }
                      >
                        <Space>
                          <Avatar
                            shape="square"
                            src={
                              MODEL_PROVIDER_ICON[
                                embedding_model_service_provider
                              ]
                            }
                          />
                          <div>
                            <div>
                              <Typography.Text
                                type={!isActive ? 'secondary' : undefined}
                              >
                                {option.data.title}
                              </Typography.Text>
                            </div>
                            <Typography.Text
                              type="secondary"
                              style={{ fontSize: 12 }}
                            >
                              <Space split={<Divider type="vertical" />}>
                                {embedding_model_name}
                                <Space>
                                  <Badge
                                    status={UI_COLLECTION_STATUS[status]}
                                  />
                                  <FormattedMessage
                                    id={`collection.status.${status}`}
                                  />
                                </Space>
                              </Space>
                            </Typography.Text>
                          </div>
                        </Space>
                      </Tooltip>
                    );
                  }}
                />
              </Form.Item>
            ) : (
              <Form.Item
                name={['config', 'chractor']}
                label={<FormattedMessage id="bot.character" />}
                rules={[
                  {
                    required: true,
                    message: formatMessage({ id: 'bot.character_required' }),
                  },
                ]}
              >
                <Select
                  options={promptTemplates?.map((item) => ({
                    label: item.name,
                    value: item.prompt,
                    description: item.description,
                  }))}
                  optionRender={(option) => (
                    <>
                      <Typography.Text style={{ display: 'block' }}>
                        {option.data.label}
                      </Typography.Text>
                      <Typography.Text
                        type="secondary"
                        style={{ fontSize: 12 }}
                      >
                        {option.data.description}
                      </Typography.Text>
                    </>
                  )}
                />
              </Form.Item>
            )}
          </Col>
        </Row>

        <Form.Item name={['config', 'model_service_provider']} hidden>
          <Input type="hidden" />
        </Form.Item>
        <Form.Item name={['config', 'model_name']} hidden>
          <Input type="hidden" />
        </Form.Item>

        <Form.Item
          label={<FormattedMessage id="model.prompt_template" />}
          name={['config', 'llm', 'prompt_template']}
        >
          <Input.TextArea
            rows={10}
            style={{ color: token.colorTextSecondary }}
          />
        </Form.Item>

        <Form.Item
          label={<FormattedMessage id="model.llm_params" />}
          className="form-item-children-wrap"
        >
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <label
                className="label-top"
                title={formatMessage({ id: 'model.llm_context_window' })}
              >
                <Form.Item
                  name={['config', 'llm', 'context_window']}
                  rules={[
                    {
                      required: true,
                      message: formatMessage({
                        id: 'model.llm_context_window_required',
                      }),
                    },
                  ]}
                >
                  <InputNumber style={{ width: '100%' }} />
                </Form.Item>
              </label>
            </Col>
            <Col span={6}>
              <label
                className="label-top"
                title={formatMessage({
                  id: 'model.llm_similarity_score_threshold',
                })}
              >
                <Form.Item
                  name={['config', 'llm', 'similarity_score_threshold']}
                  rules={[
                    {
                      required: true,
                      message: formatMessage({
                        id: 'model.llm_similarity_score_threshold_required',
                      }),
                    },
                  ]}
                >
                  <InputNumber
                    min={0.1}
                    max={1}
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </label>
            </Col>
            <Col span={6}>
              <label
                className="label-top"
                title={formatMessage({ id: 'model.llm_similarity_topk' })}
              >
                <Form.Item
                  name={['config', 'llm', 'similarity_topk']}
                  rules={[
                    {
                      required: true,
                      message: formatMessage({
                        id: 'model.llm_similarity_topk_required',
                      }),
                    },
                  ]}
                >
                  <InputNumber
                    min={1}
                    max={10}
                    step={1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </label>
            </Col>
            <Col span={6}>
              <label
                className="label-top"
                title={formatMessage({ id: 'model.llm_temperature' })}
              >
                <Form.Item
                  name={['config', 'llm', 'temperature']}
                  rules={[
                    {
                      required: true,
                      message: formatMessage({
                        id: 'model.llm_temperature_required',
                      }),
                    },
                  ]}
                >
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </label>
            </Col>
          </Row>
        </Form.Item>

        {currentModel?.memory ? (
          <>
            <Form.Item
              label={
                <Space>
                  <FormattedMessage id="model.memory" />
                  <Typography.Text type="secondary">
                    <FormattedMessage id="model.memory_tips" />
                  </Typography.Text>
                </Space>
              }
              valuePropName="checked"
              name={['config', 'memory']}
            >
              <Switch />
            </Form.Item>
            {botMemory ? (
              <Row gutter={[16, 16]}>
                <Col span={6}>
                  <Form.Item
                    rules={[
                      {
                        required: true,
                        type: 'integer',
                        message: formatMessage({
                          id: 'model.memory_should_integer',
                        }),
                        transform(value) {
                          return Number(value) > 0 ? Number(value) : 'error';
                        },
                      },
                    ]}
                    name={['config', 'memory_length']}
                  >
                    <InputNumber
                      min={1}
                      addonBefore={
                        <Typography.Text type="secondary">
                          {formatMessage({
                            id: 'model.memory_max_token',
                          })}
                        </Typography.Text>
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={6}>
                  <Form.Item
                    rules={[
                      {
                        required: true,
                        type: 'integer',
                        message: formatMessage({
                          id: 'model.memory_should_integer',
                        }),
                        transform(value) {
                          return Number(value) > 0 ? Number(value) : 'error';
                        },
                      },
                    ]}
                    name={['config', 'memory_count']}
                  >
                    <InputNumber
                      min={1}
                      addonBefore={
                        <Typography.Text type="secondary">
                          {formatMessage({
                            id: 'model.memory_max_session',
                          })}
                        </Typography.Text>
                      }
                    />
                  </Form.Item>
                </Col>
              </Row>
            ) : null}
          </>
        ) : null}

        {botType === 'knowledge' ? (
          <>
            <Form.Item
              label={
                <Space>
                  <FormattedMessage id="model.recall" />
                  <Typography.Text type="secondary">
                    <FormattedMessage id="model.recall_help" />
                  </Typography.Text>
                </Space>
              }
              valuePropName="checked"
              name={['config', 'llm', 'enable_keyword_recall']}
            >
              <Switch />
            </Form.Item>
            <Form.Item
              label={formatMessage({ id: 'model.use_related_question' })}
              valuePropName="checked"
              name={['config', 'use_related_question']}
            >
              <Switch />
            </Form.Item>
            {botUseRelatedQuestion ? (
              <>
                <Form.Item
                  name={['config', 'related_question_prompt']}
                  label={formatMessage({ id: 'model.related_question_tips' })}
                >
                  <Input
                    placeholder={formatMessage({
                      id: 'model.related_question_prompt',
                    })}
                  />
                </Form.Item>

                <Form.Item
                  label={formatMessage({ id: 'model.related_prompt_template' })}
                  name={['config', 'llm', 'related_prompt_template']}
                >
                  <Input.TextArea autoSize={{ minRows: 8, maxRows: 12 }} />
                </Form.Item>
              </>
            ) : null}
          </>
        ) : null}

        <br />
        <Divider />
        <div style={{ textAlign: 'right' }}>
          <Button
            loading={loading}
            style={{ minWidth: 160 }}
            type="primary"
            htmlType="submit"
          >
            <FormattedMessage
              id={action !== 'add' ? 'action.update' : 'action.save'}
            />
          </Button>
        </div>
      </Form>
    </>
  );
};
