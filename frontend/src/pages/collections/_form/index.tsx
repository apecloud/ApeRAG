import { Collection } from '@/api';
import { IndexTypeSelector, ModelSelect } from '@/components';

import { QuestionCircleOutlined } from '@ant-design/icons';
import {
  Button,
  Card,
  Divider,
  Form,
  FormInstance,
  Input,
  Tooltip,
  Typography,
  theme,
} from 'antd';
import { useEffect } from 'react';
import { useIntl, useModel } from 'umi';

type Props = {
  action: 'add' | 'edit';
  values?: Collection;
  form: FormInstance<Collection>;
  onSubmit: (data: Collection) => void;
};

// Index types configuration
const configIndexTypesKey = ['config', 'index_types'];
const configEnableVectorKey = ['config', 'enable_vector'];
const configEnableFulltextKey = ['config', 'enable_fulltext'];
const configEnableKnowledgeGraphKey = ['config', 'enable_knowledge_graph'];
const configEnableSummaryKey = ['config', 'enable_summary'];
const configEnableVisionKey = ['config', 'enable_vision'];

// Model configuration
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
  const indexTypes =
    Form.useWatch(configIndexTypesKey, form) ||
    (action === 'add' ? ['vector', 'fulltext', 'graph'] : []);
  const embeddingModel = Form.useWatch(configEmbeddingModelKey, form);
  const completionModel = Form.useWatch(configCompletionModelKey, form);

  const enableKnowledgeGraph = indexTypes.includes('graph');
  const enableSummary = indexTypes.includes('summary');
  const enableVision = indexTypes.includes('vision');

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

  // Update individual index type flags when indexTypes changes
  useEffect(() => {
    if (indexTypes) {
      form.setFieldValue(configEnableVectorKey, indexTypes.includes('vector'));
      form.setFieldValue(
        configEnableFulltextKey,
        indexTypes.includes('fulltext'),
      );
      form.setFieldValue(
        configEnableKnowledgeGraphKey,
        indexTypes.includes('graph'),
      );
      form.setFieldValue(
        configEnableSummaryKey,
        indexTypes.includes('summary'),
      );
      form.setFieldValue(configEnableVisionKey, indexTypes.includes('vision'));
    }
  }, [indexTypes, form]);

  useEffect(() => {
    getAvailableModels([{"operation":"AND","tags":["enable_for_collection"]}]);
  }, []);

  // Set initial index types based on config values in edit mode
  useEffect(() => {
    if (action === 'edit' && values?.config) {
      const indexTypes = [];
      if (values.config.enable_vector) indexTypes.push('vector');
      if (values.config.enable_fulltext) indexTypes.push('fulltext');
      if (values.config.enable_knowledge_graph) indexTypes.push('graph');
      if (values.config.enable_summary) indexTypes.push('summary');
      if (values.config.enable_vision) indexTypes.push('vision');

      form.setFieldValue(configIndexTypesKey, indexTypes);
    }
  }, [action, values, form]);

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
          <div>
            <Input.TextArea
              maxLength={300}
              rows={3}
              readOnly={enableSummary}
              placeholder={
                enableSummary
                  ? formatMessage({
                      id: 'collection.description.auto_generated.placeholder',
                    })
                  : undefined
              }
              style={{
                color: enableSummary
                  ? token.colorTextDisabled
                  : token.colorText,
                backgroundColor: enableSummary
                  ? token.colorBgContainerDisabled
                  : token.colorBgContainer,
                cursor: enableSummary ? 'default' : 'text',
              }}
            />
          </div>
        </Form.Item>

        {/* Index Types Selector */}
        <Form.Item
          name={configIndexTypesKey}
          initialValue={
            action === 'add' ? ['vector', 'fulltext', 'graph'] : undefined
          }
        >
          <IndexTypeSelector disabled={action === 'edit'} />
        </Form.Item>

        {/* Model Configuration */}
        <div style={{ marginTop: 32, marginBottom: 24 }}>
          <Typography.Title level={4} style={{ marginBottom: 16 }}>
            {formatMessage({ id: 'collection.advanced_settings' })}
          </Typography.Title>
          <Typography.Paragraph
            style={{ marginBottom: 24, color: token.colorTextSecondary }}
          >
            {formatMessage({ id: 'collection.model_settings.description' })}
          </Typography.Paragraph>
        </div>

        {/* Embedding Model - Always required */}
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
          label={formatMessage({ id: 'collection.embedding_model' })}
        >
          <ModelSelect
            model="embedding"
            disabled={action === 'edit'}
            tagfilters={[
              {
                operation: 'OR',
                tags: ['enable_for_collection'],
              },
            ]}
          />
        </Form.Item>

        {/* Completion Model - Required for graph, summary, or vision */}
        {(enableKnowledgeGraph || enableSummary || enableVision) && (
          <Form.Item
            label={formatMessage({
              id: 'collection.completion_model',
            })}
            name={configCompletionModelKey}
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
              disabled={action === 'edit'}
              tagfilters={[
                {
                  operation: 'OR',
                  tags: ['enable_for_collection'],
                },
              ]}
            />
          </Form.Item>
        )}

        {/* Hidden fields for model providers */}
        <Form.Item name={configEmbeddingModelServiceProviderKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configEmbeddingCustomLlmProviderKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configCompletionModelServiceProviderKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configCompletionCustomLlmProviderKey} hidden>
          <Input hidden />
        </Form.Item>

        {/* Hidden index type flags */}
        <Form.Item name={configEnableVectorKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configEnableFulltextKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configEnableKnowledgeGraphKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configEnableSummaryKey} hidden>
          <Input hidden />
        </Form.Item>
        <Form.Item name={configEnableVisionKey} hidden>
          <Input hidden />
        </Form.Item>

        {/* Set source to 'system' for file upload only */}
        <Form.Item name={['config', 'source']} initialValue="system" hidden>
          <Input hidden />
        </Form.Item>



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
