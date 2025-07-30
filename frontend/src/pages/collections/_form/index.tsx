import { Collection } from '@/api';
import { ApeMarkdown, CheckCard, ModelSelect } from '@/components';
import { COLLECTION_SOURCE, COLLECTION_SOURCE_EMAIL } from '@/constants';

import { CollectionConfigSource, CollectionEmailSource } from '@/types';
import { QuestionCircleOutlined } from '@ant-design/icons';
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
  Row,
  Segmented,
  Space,
  Switch,
  Tooltip,
  Typography,
  theme,
} from 'antd';
import _ from 'lodash';
import { useEffect } from 'react';
import { FormattedMessage, useIntl, useModel } from 'umi';
import DocumentCloudFormItems from './DocumentCloudFormItems';
import DocumentEmailFormItems from './DocumentEmailFormItems';
import DocumentFeishuFormItems from './DocumentFeishuFormItems';
import DocumentFtpFormItems from './DocumentFtpFormItems';
import DocumentGithubFormItems from './DocumentGithubFormItems';
import DocumentLocalFormItems from './DocumentLocalFormItems';

type Props = {
  action: 'add' | 'edit';
  values?: Collection;
  form: FormInstance<Collection>;
  onSubmit: (data: Collection) => void;
};

const configSourceKey = ['config', 'source'];

const configEmbeddingModelKey = ['config', 'embedding', 'model'];
const configEmbeddingModelServiceProviderKey = [
  'config',
  'embedding',
  'model_service_provider',
];
const configEmbeddingCustomLlmProviderKey = [
  'config',
  'embedding',
  'custom_llm_provider',
];

const configEnableKnowledgeGraphKey = ['config', 'enable_knowledge_graph'];
const configEnableSummaryKey = ['config', 'enable_summary'];
const configEnableVisionKey = ['config', 'enable_vision'];

const configCompletionModelKey = ['config', 'completion', 'model'];
const configCompletionModelServiceProviderKey = [
  'config',
  'completion',
  'model_service_provider',
];
const configCompletionCustomLlmProviderKey = [
  'config',
  'completion',
  'custom_llm_provider',
];

const configEmailSourceKey = ['config', 'email_source'];
export default ({ onSubmit, action, values, form }: Props) => {
  const { formatMessage } = useIntl();
  const { token } = theme.useToken();

  const {
    availableModels,
    getAvailableModels,
    getDefaultModelsFromAvailable,
    getProviderByModelName,
  } = useModel('models');
  // save collection
  const onFinish = async () => {
    const data = await form.validateFields();
    onSubmit(data);
  };

  // form field watch
  const source = Form.useWatch(configSourceKey, form);
  const emailSource: CollectionEmailSource | undefined = Form.useWatch(
    configEmailSourceKey,
    form,
  );
  const embeddingModel = Form.useWatch(configEmbeddingModelKey, form);

  const enableKnowledgeGraph = Form.useWatch(
    configEnableKnowledgeGraphKey,
    form,
  );

  const enableSummary = Form.useWatch(configEnableSummaryKey, form);

  const completionModel = Form.useWatch(configCompletionModelKey, form);

  // Set default models for new collection when availableModels are loaded
  useEffect(() => {
    if (action === 'add' && availableModels.length > 0) {
      const { defaultEmbeddingModel, defaultCompletionModel } =
        getDefaultModelsFromAvailable();

      // Set default embedding model if available and not already set
      if (
        defaultEmbeddingModel &&
        !form.getFieldValue(configEmbeddingModelKey)
      ) {
        form.setFieldValue(configEmbeddingModelKey, defaultEmbeddingModel);
      }

      // Set default completion model if available and not already set
      if (
        defaultCompletionModel &&
        !form.getFieldValue(configCompletionModelKey)
      ) {
        form.setFieldValue(configCompletionModelKey, defaultCompletionModel);
      }
    }
  }, [action, availableModels, form, getDefaultModelsFromAvailable]);

  useEffect(() => {
    if (embeddingModel) {
      const { provider, model } = getProviderByModelName(
        embeddingModel,
        'embedding',
      );
      form.setFieldValue(
        configEmbeddingModelServiceProviderKey,
        provider?.name,
      );
      form.setFieldValue(
        configEmbeddingCustomLlmProviderKey,
        model?.custom_llm_provider,
      );
    }
  }, [embeddingModel, availableModels]);

  useEffect(() => {
    if (completionModel) {
      const { provider, model } = getProviderByModelName(
        completionModel,
        'completion',
      );
      form.setFieldValue(
        configCompletionModelServiceProviderKey,
        provider?.name,
      );
      form.setFieldValue(
        configCompletionCustomLlmProviderKey,
        model?.custom_llm_provider,
      );
    }
  }, [completionModel, availableModels]);

  useEffect(() => {
    if (source === 'ftp') {
      form.setFieldValue(['config', 'port'], 21);
    }
    if (source === 'email') {
      if (!emailSource) {
        form.setFieldValue(configEmailSourceKey, 'gmail');
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
    getAvailableModels();
  }, []);

  return (
    <Form
      autoComplete="off"
      onFinish={onFinish}
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
          label={
            <span>
              {formatMessage({ id: 'text.description' })}
              <Tooltip
                title={
                  <div
                    style={{
                      maxWidth: 400,
                      whiteSpace: 'pre-line',
                      color: '#fff',
                      fontSize: token.fontSize,
                    }}
                  >
                    {formatMessage({ id: 'collection.description.tips' })}
                  </div>
                }
                overlayStyle={{
                  maxWidth: 400,
                }}
                overlayInnerStyle={{
                  backgroundColor: token.colorBgSpotlight,
                  color: '#fff',
                  borderRadius: token.borderRadius,
                  padding: '8px 12px',
                }}
              >
                <QuestionCircleOutlined
                  style={{ marginLeft: 6, color: token.colorTextSecondary }}
                />
              </Tooltip>
            </span>
          }
        >
          <Input.TextArea
            maxLength={300}
            rows={3}
            readOnly={enableSummary}
            style={{
              color: token.colorText,
              backgroundColor: token.colorBgContainer,
              cursor: enableSummary ? 'default' : 'text',
            }}
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
              name={configEmbeddingModelKey}
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
              <ModelSelect 
                model="embedding" 
                disabled={action === 'edit'} 
                tagFilters={[{
                  operation: "OR",
                  tags: ["enable_for_collection"]
                }]}
              />
            </Form.Item>

            <Form.Item name={configEmbeddingModelServiceProviderKey} hidden>
              <Input hidden />
            </Form.Item>
            <Form.Item name={configEmbeddingCustomLlmProviderKey} hidden>
              <Input hidden />
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
              label={formatMessage({ id: 'collection.enable_knowledge_graph' })}
              valuePropName="checked"
              name={configEnableKnowledgeGraphKey}
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>

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
              label={formatMessage({ id: 'collection.enable_auto_summary' })}
              valuePropName="checked"
              name={configEnableSummaryKey}
            >
              <Switch />
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
              label={formatMessage({ id: 'collection.enable_vision' })}
              valuePropName="checked"
              name={configEnableVisionKey}
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        {enableKnowledgeGraph || enableSummary ? (
          <>
            <Form.Item
              label={formatMessage({
                id: 'collection.completion_model',
              })}
              name={configCompletionModelKey}
              required
              rules={[
                {
                  required: true,
                  message: formatMessage({
                    id: 'collection.completion_model.required',
                  }),
                },
              ]}
            >
              <ModelSelect 
                model="completion" 
                tagFilters={[{
                  operation: "OR",
                  tags: ["enable_for_collection"]
                }]}
              />
            </Form.Item>
            <Form.Item name={configCompletionModelServiceProviderKey} hidden>
              <Input hidden />
            </Form.Item>
            <Form.Item name={configCompletionCustomLlmProviderKey} hidden>
              <Input hidden />
            </Form.Item>
          </>
        ) : null}

        <Form.Item
          name={configSourceKey}
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
              const config = values?.config;
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
              name={configEmailSourceKey}
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
          <Button style={{ minWidth: 160 }} type="primary" htmlType="submit">
            {formatMessage({ id: 'action.save' })}
          </Button>
        </div>
      </Card>
    </Form>
  );
};
