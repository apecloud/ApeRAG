import { AvailableEmbedding, AvailableModel, SupportedModelServiceProvider } from '@/api';
import { ApeMarkdown, CheckCard } from '@/components';
import {
  COLLECTION_SOURCE,
  COLLECTION_SOURCE_EMAIL,
  MODEL_PROVIDER_ICON,
} from '@/constants';
import { api } from '@/services';
import { UndrawEmpty } from 'react-undraw-illustrations';

import {
  ApeCollection,
  CollectionConfigSource,
  CollectionEmailSource,
} from '@/types';
import {
  Alert,
  Avatar,
  Button,
  Card,
  Col,
  Divider,
  Form,
  FormInstance,
  Input,
  Radio,
  Row,
  Segmented,
  Select,
  Space,
  Switch,
  theme,
  Typography,
} from 'antd';
import _ from 'lodash';
import { useEffect, useMemo, useState } from 'react';
import { FormattedMessage, useIntl, useModel } from 'umi';
import DocumentCloudFormItems from './DocumentCloudFormItems';
import DocumentEmailFormItems from './DocumentEmailFormItems';
import DocumentFeishuFormItems from './DocumentFeishuFormItems';
import DocumentFtpFormItems from './DocumentFtpFormItems';
import DocumentGithubFormItems from './DocumentGithubFormItems';
import DocumentLocalFormItems from './DocumentLocalFormItems';

type Props = {
  action: 'add' | 'edit';
  values?: ApeCollection;
  form: FormInstance<ApeCollection>;
  onSubmit: (data: ApeCollection) => void;
};

export default ({ onSubmit, action, values, form }: Props) => {
  const { formatMessage } = useIntl();
  const { token } = theme.useToken();
  const { setLoading, loading } = useModel('global');

  const [availableEmbeddings, setAvailableEmbeddings] =
    useState<AvailableEmbedding[]>();
  const [availableModels, setAvailableModels] =
    useState<AvailableModel[]>();
  const [supportedModelServiceProviders, setSupportedModelServiceProviders] =
    useState<SupportedModelServiceProvider[]>();

  const source = Form.useWatch(['config', 'source'], form);
  const emailSource: CollectionEmailSource | undefined = Form.useWatch(
    ['config', 'email_source'],
    form,
  );
  const sensitiveProtect = Form.useWatch(['config', 'sensitive_protect'], form);
  const embeddingModel = Form.useWatch(['config', 'embedding_model'], form);
  const enableLightRAG = Form.useWatch(['config', 'enable_lightrag'], form);
  const lightRAGModel = Form.useWatch(['config', 'lightrag_model'], form);

  const getEmbeddingsAndModels = async () => {
    setLoading(true);
    const [availableEmbeddingsRes, availableModelsRes, supportedModelServiceProvidersRes] =
      await Promise.all([
        api.availableEmbeddingsGet(),
        api.availableModelsGet(),
        api.supportedModelServiceProvidersGet(),
      ]);
    setLoading(false);
    setAvailableEmbeddings(availableEmbeddingsRes.data.items);
    setAvailableModels(availableModelsRes.data.items);
    setSupportedModelServiceProviders(
      supportedModelServiceProvidersRes.data.items,
    );
  };

  const embeddingModelOptions = useMemo(
    () =>
      _.map(
        _.groupBy(availableEmbeddings, 'model_service_provider'),
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
                label: aeb.embedding_name,
                value: `${providerName}:${aeb.embedding_name}`,
              };
            }),
          };
        },
      ),
    [availableEmbeddings, supportedModelServiceProviders],
  );


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

  const onFinish = async () => {
    const data = await form.validateFields();
    onSubmit(data);
  };

  useEffect(() => {
    if (embeddingModel) {
      const [model_service_provider, embedding_name] =
        embeddingModel.split(':');
      form.setFieldValue(
        ['config', 'embedding_model_service_provider'],
        model_service_provider,
      );
      form.setFieldValue(['config', 'embedding_model_name'], embedding_name);
    }
  }, [embeddingModel]);


  useEffect(() => {
    if (lightRAGModel) {
      const [model_service_provider, model_name] = lightRAGModel.split(':');
      form.setFieldValue(
        ['config', 'lightrag_model_service_provider'],
        model_service_provider,
      );
      form.setFieldValue(['config', 'lightrag_model_name'], model_name);
    }
  }, [lightRAGModel]);

  useEffect(() => {
    if (source === 'ftp') {
      form.setFieldValue(['config', 'port'], 21);
    }
    if (source === 'email') {
      if (!emailSource) {
        form.setFieldValue(['config', 'email_source'], 'gmail');
      } else {
        form.setFieldValue(
          ['config', 'pop_server'],
          COLLECTION_SOURCE_EMAIL[emailSource].pop_server,
        );
        form.setFieldValue(
          ['config', 'port'],
          COLLECTION_SOURCE_EMAIL[emailSource].port,
        );
      }
    }
  }, [source, emailSource]);

  useEffect(() => {
    getEmbeddingsAndModels();
  }, []);

  return (
    <Form
      autoComplete="off"
      onFinish={onFinish}
      // disabled={readonly}
      layout="vertical"
      form={form}
      initialValues={values}
    >
      <Card style={{ marginBottom: 20 }}>
        <Form.Item
          name="title"
          label={formatMessage({ id: 'text.title' })}
          rules={[
            {
              required: true,
              message: formatMessage({ id: 'text.title.required' }),
            },
          ]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          name="description"
          label={formatMessage({ id: 'text.description' })}
        >
          <Input.TextArea maxLength={300} rows={3} />
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
              name={['config', 'embedding_model']}
              rules={[
                {
                  required: true,
                  message: formatMessage({
                    id: 'collection.embedding_model.required',
                  }),
                },
              ]}
              className="form-item-wrap"
              label={formatMessage({ id: 'collection.embedding_model' })}
            >
              <Select
                options={embeddingModelOptions}
                disabled={action == 'edit'}
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
                notFoundContent={
                  <Space
                    direction="vertical"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      marginBlock: 24,
                    }}
                  >
                    <UndrawEmpty
                      primaryColor={token.colorPrimary}
                      height="80px"
                    />
                    <Typography.Text type="secondary">
                      <FormattedMessage id="collection.embedding_model_not_found" />
                    </Typography.Text>
                  </Space>
                }
              />
            </Form.Item>

            <Form.Item
              name={['config', 'embedding_model_service_provider']}
              hidden
            >
              <Input type="hidden" />
            </Form.Item>
            <Form.Item name={['config', 'embedding_model_name']} hidden>
              <Input type="hidden" />
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
            <Form.Item
              label={formatMessage({ id: 'collection.enable_lightrag' })}
              valuePropName="checked"
              name={['config', 'enable_lightrag']}
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        {enableLightRAG ? (
          <>
            <Form.Item
              label={formatMessage({
                id: 'collection.lightrag_model',
              })}
              name={['config', 'lightrag_model']}
              required
              rules={[
                {
                  required: true,
                  message: formatMessage({
                    id: 'collection.lightrag_model.required',
                  }),
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
                notFoundContent={
                  <Space
                    direction="vertical"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      marginBlock: 24,
                    }}
                  >
                    <UndrawEmpty
                      primaryColor={token.colorPrimary}
                      height="80px"
                    />
                    <Typography.Text type="secondary">
                      <FormattedMessage id="collection.lightrag_model_not_found" />
                    </Typography.Text>
                  </Space>
                }
              />
            </Form.Item>
            <Form.Item
              name={['config', 'lightrag_model_service_provider']}
              hidden
            >
              <Input type="hidden" />
            </Form.Item>
            <Form.Item name={['config', 'lightrag_model_name']} hidden>
              <Input type="hidden" />
            </Form.Item>
          </>
        ) : null }

        <Form.Item
          label={
            <>
              <Typography.Text>
                {formatMessage({ id: 'collection.sensitive.protect' })}
              </Typography.Text>
              <Typography.Text type="secondary">
                （{formatMessage({ id: 'collection.sensitive.help' })}）
              </Typography.Text>
            </>
          }
          valuePropName="checked"
          name={['config', 'sensitive_protect']}
        >
          <Switch />
        </Form.Item>
        {sensitiveProtect ? (
          <>
            <Form.Item
              label={formatMessage({
                id: 'collection.sensitive.protect.method',
              })}
              name={['config', 'sensitive_protect_method']}
              required
              rules={[
                {
                  required: true,
                  message: formatMessage({
                    id: 'collection.sensitive.protect.method.required',
                  }),
                },
              ]}
            >
              <Radio.Group
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                }}
                options={[
                  {
                    label: formatMessage({
                      id: 'collection.sensitive.nostore',
                    }),
                    value: 'nostore',
                  },
                  {
                    label: formatMessage({
                      id: 'collection.sensitive.replace',
                    }),
                    value: 'replace',
                  },
                ]}
              />
            </Form.Item>
          </>
        ) : null}

        <Form.Item
          name={['config', 'source']}
          required
          label={formatMessage({ id: 'collection.source' })}
          rules={[
            {
              required: true,
              message: formatMessage({ id: 'collection.source.required' }),
            },
          ]}
        >
          <CheckCard
            options={Object.keys(COLLECTION_SOURCE).map((key) => {
              const config = values.config;
              return {
                label: formatMessage({ id: `collection.source.${key}` }),
                value: key,
                icon: COLLECTION_SOURCE[key as CollectionConfigSource].icon,
                disabled:
                  !COLLECTION_SOURCE[key as CollectionConfigSource].enabled ||
                  (action === 'edit' && key !== config?.source),
              };
            })}
          />
        </Form.Item>

        {source === 'local' ? <DocumentLocalFormItems /> : null}

        {source === 'email' ? (
          <>
            <Form.Item
              required
              label={formatMessage({ id: 'email.source' })}
              name={['config', 'email_source']}
            >
              <Segmented
                size="small"
                block
                options={_.map(COLLECTION_SOURCE_EMAIL, (conf, key) => ({
                  label: (
                    <Space style={{ padding: 10 }}>
                      <Avatar shape="square" src={conf.icon} size={24} />
                      <Typography.Text>
                        <FormattedMessage id={`email.${key}`} />
                      </Typography.Text>
                    </Space>
                  ),
                  value: key,
                }))}
              />
            </Form.Item>
            <DocumentEmailFormItems />
            {emailSource ? (
              <Form.Item label="">
                <Alert
                  message={formatMessage({
                    id: `email.${emailSource}.tips.title`,
                  })}
                  description={
                    <ApeMarkdown>
                      {formatMessage({
                        id: `email.${emailSource}.tips.description`,
                      })}
                    </ApeMarkdown>
                  }
                  type="info"
                  showIcon
                />
              </Form.Item>
            ) : null}
          </>
        ) : null}

        {source === 's3' || source === 'oss' ? (
          <DocumentCloudFormItems />
        ) : null}

        {source === 'ftp' ? <DocumentFtpFormItems /> : null}

        {source === 'feishu' ? <DocumentFeishuFormItems /> : null}

        {source === 'github' ? <DocumentGithubFormItems /> : null}

        <br />
        <Divider />
        <div style={{ textAlign: 'right' }}>
          <Button
            loading={loading}
            style={{ minWidth: 160 }}
            type="primary"
            htmlType="submit"
          >
            {formatMessage({ id: 'action.save' })}
          </Button>
        </div>
      </Card>
    </Form>
  );
};
