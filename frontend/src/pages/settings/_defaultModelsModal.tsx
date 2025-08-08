import { api } from '@/services';
import { TagFilterConditionOperationEnum } from '@/api/models/tag-filter-condition';
import { DefaultApi } from '@/api/apis/default-api';
import { DefaultModelConfigScenarioEnum } from '@/api/models/default-model-config';
import { Form, Modal, Select, message } from 'antd';
import { useEffect, useState } from 'react';
import { useIntl, useModel } from 'umi';

interface DefaultModelsModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

interface ScenarioConfig {
  key: DefaultModelConfigScenarioEnum;
  label: string;
}

const DefaultModelsModal: React.FC<DefaultModelsModalProps> = ({
  visible,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const { formatMessage } = useIntl();
  const { getAvailableModels } = useModel('models');
  
  // Store filtered models for different scenarios
  const [collectionModels, setCollectionModels] = useState<any[]>([]);
  const [agentModels, setAgentModels] = useState<any[]>([]);
  
  // Create DefaultApi instance
  const defaultApi = new DefaultApi();

  const scenarios: ScenarioConfig[] = [
    {
      key: DefaultModelConfigScenarioEnum.default_for_collection_completion,
      label: formatMessage({ id: 'default.models.collection.completion' }),
    },
    {
      key: DefaultModelConfigScenarioEnum.default_for_agent_completion,
      label: formatMessage({ id: 'default.models.agent.completion' }),
    },
    {
      key: DefaultModelConfigScenarioEnum.default_for_embedding,
      label: formatMessage({ id: 'default.models.embedding' }),
    },
    {
      key: DefaultModelConfigScenarioEnum.default_for_rerank,
      label: formatMessage({ id: 'default.models.rerank' }),
    },
    {
      key: DefaultModelConfigScenarioEnum.default_for_background_task,
      label: formatMessage({ id: 'default.models.background_task' }),
    },
  ];

  // Fetch filtered models for different scenarios
  const fetchFilteredModels = async () => {
    try {
      // Fetch models for collection scenarios (completion, embedding, rerank)
      const collectionTagFilters = [{ operation: TagFilterConditionOperationEnum.AND, tags: ['enable_for_collection'] }];
      const collectionRes = await api.availableModelsPost({ 
        tagFilterRequest: { tag_filters: collectionTagFilters } 
      });
      setCollectionModels(collectionRes.data.items || []);
      
      // Fetch models for agent scenarios (completion only)
      const agentTagFilters = [{ operation: TagFilterConditionOperationEnum.AND, tags: ['enable_for_agent'] }];
      const agentRes = await api.availableModelsPost({ 
        tagFilterRequest: { tag_filters: agentTagFilters } 
      });
      setAgentModels(agentRes.data.items || []);
    } catch (error) {
      console.error('Failed to fetch filtered models:', error);
    }
  };

  // Get model options for specific scenario
  const getModelOptions = (scenario: string) => {
    const options: Array<{ label: string; value: string; provider: string }> = [];
    let modelData: any[] = [];
    let modelType: 'completion' | 'embedding' | 'rerank' = 'completion';
    
    // Determine which models to use based on scenario
    if (scenario === 'default_for_agent_completion') {
      modelData = agentModels;
      modelType = 'completion';
    } else if (scenario === 'default_for_collection_completion') {
      modelData = collectionModels;
      modelType = 'completion';
    } else if (scenario === 'default_for_embedding') {
      modelData = collectionModels;
      modelType = 'embedding';
    } else if (scenario === 'default_for_rerank') {
      modelData = collectionModels;
      modelType = 'rerank';
    } else if (scenario === 'default_for_background_task') {
      // Background task (e.g., auto title) uses completion models; prefer agent-enabled models
      modelData = agentModels;
      modelType = 'completion';
    }

    modelData?.forEach((provider) => {
      provider[modelType]?.forEach((model: any) => {
        if (model.model) {
          options.push({
            label: `${provider.label || provider.name} / ${model.model}`,
            value: model.model,
            provider: provider.name || '',
          });
        }
      });
    });

    return options;
  };

  // Handle save
  const handleSave = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();

      // Include all scenarios - null values will delete the default model configuration
      const defaults = scenarios.map((scenario) => ({
        scenario: scenario.key,
        provider_name: values[`${scenario.key}_provider`] || null,
        model: values[scenario.key] || null,
      }));

      // Call API to update default models
      await defaultApi.defaultModelsPut({
        defaultModelsUpdateRequest: { defaults }
      });
      
      message.success(formatMessage({ id: 'default.models.save.success' }));
      onSuccess();
    } catch (error: any) {
      console.error('Failed to save default models:', error);
      
      // Handle specific error types
      let errorMessageId = 'default.models.save.error';
      
      if (error?.response?.data?.error_code) {
        const errorCode = error.response.data.error_code;
        switch (errorCode) {
          case 'PROVIDER_NOT_PUBLIC':
            errorMessageId = 'default.models.provider.not.public';
            break;
          case 'LLM_MODEL_NOT_FOUND':
            errorMessageId = 'default.models.provider.not.found';
            break;
          default:
            errorMessageId = 'default.models.save.error';
        }
      }
      
      message.error(formatMessage({ id: errorMessageId }));
    } finally {
      setLoading(false);
    }
  };

  // Fetch current defaults
  const fetchCurrentDefaults = async () => {
    try {
      const response = await defaultApi.defaultModelsGet();
      const formValues: any = {};

      // Populate form with current default models
      if (response.data?.items) {
        response.data.items.forEach((item) => {
          if (item.model && item.provider_name) {
            formValues[item.scenario!] = item.model;
            formValues[`${item.scenario!}_provider`] = item.provider_name;
          }
        });
      }

      form.setFieldsValue(formValues);
    } catch (error) {
      console.error('Failed to fetch default models:', error);
      message.error(formatMessage({ id: 'default.models.fetch.error' }));
    }
  };

  useEffect(() => {
    if (visible) {
      // Fetch filtered models for different scenarios
      fetchFilteredModels();
      fetchCurrentDefaults();
    }
  }, [visible]);

  return (
    <Modal
      title={formatMessage({ id: 'default.models.modal.title' })}
      open={visible}
      onCancel={onCancel}
      onOk={handleSave}
      confirmLoading={loading}
      width={600}
      destroyOnClose
      maskClosable={false}
    >
      <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#f6f8fa', borderRadius: 6, border: '1px solid #e1e4e8' }}>
        <div style={{ fontSize: 14, color: '#586069', lineHeight: '1.5' }}>
          💡 {formatMessage({ id: 'default.models.clear.hint' })}
        </div>
      </div>
      
      <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
        {scenarios.map((scenario) => (
          <div key={scenario.key}>
            <Form.Item
              label={scenario.label}
              name={scenario.key}
            >
              <Select
                placeholder={formatMessage({ id: 'default.models.select.placeholder' })}
                allowClear
                showSearch
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
                options={getModelOptions(scenario.key)}
                onChange={(value, option) => {
                  // Set provider information when model is selected
                  if (option && 'provider' in option) {
                    form.setFieldValue(`${scenario.key}_provider`, option.provider);
                  } else {
                    // Clear provider when model is cleared
                    form.setFieldValue(`${scenario.key}_provider`, undefined);
                  }
                }}
              />
            </Form.Item>
            {/* Hidden field to store provider information */}
            <Form.Item
              name={`${scenario.key}_provider`}
              style={{ display: 'none' }}
            >
              <input type="hidden" />
            </Form.Item>
          </div>
        ))}
      </Form>
    </Modal>
  );
};

export default DefaultModelsModal;