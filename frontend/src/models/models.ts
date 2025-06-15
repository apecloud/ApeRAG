import { ModelConfig, ModelDefinition, PromptTemplate } from '@/api';
import { api } from '@/services';
import { useState } from 'react';
import { useModel } from 'umi';

export default () => {
  const { setLoading } = useModel('global');
  const [promptTemplates, setPromptTemplates] = useState<PromptTemplate[]>();
  const [availableModels, setAvailableModels] = useState<ModelConfig[]>([]);

  // get available models (recommend only by default)
  const getAvailableModels = async () => {
    setLoading(true);
    const res = await api.availableModelsPost({});
    setLoading(false);
    setAvailableModels(res.data.items || []);
  };

  const getProviderByModelName = (
    name: string = '',
    type: 'embedding' | 'completion' | 'rerank',
  ): { provider?: ModelConfig; model?: ModelDefinition } => {
    let provider;
    let model;
    availableModels.forEach((p) => {
      p[type]?.forEach((m) => {
        if (m.model === name) {
          provider = p;
          model = m;
        }
      });
    });
    return { provider, model };
  };

  // prompt templates
  const getPromptTemplates = async () => {
    setLoading(true);
    const res = await api.promptTemplatesGet();
    setLoading(false);
    setPromptTemplates(res.data.items);
  };

  return {
    promptTemplates,
    getPromptTemplates,
    setPromptTemplates,

    availableModels,
    getAvailableModels,
    getProviderByModelName,
  };
};
