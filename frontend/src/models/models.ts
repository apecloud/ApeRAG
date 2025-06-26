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

  // get default models from already loaded availableModels
  const getDefaultModelsFromAvailable = (): {
    defaultEmbeddingModel?: string;
    defaultCompletionModel?: string;
  } => {
    let defaultEmbeddingModel: string | undefined;
    let defaultCompletionModel: string | undefined;

    if (!availableModels?.length) {
      return { defaultEmbeddingModel, defaultCompletionModel };
    }

    // Find default embedding model, fallback to first available
    for (const provider of availableModels) {
      if (provider?.embedding?.length) {
        // Try to find model with default tag first
        const defaultModel = provider.embedding.find((model) =>
          model?.tags?.includes('default_for_embedding'),
        );
        if (defaultModel?.model) {
          defaultEmbeddingModel = defaultModel.model;
          break;
        }
        // Fallback to first model if no default found
        if (!defaultEmbeddingModel && provider.embedding[0]?.model) {
          defaultEmbeddingModel = provider.embedding[0].model;
        }
      }
    }

    // Find default completion model, fallback to first available
    for (const provider of availableModels) {
      if (provider?.completion?.length) {
        // Try to find model with default tag first
        const defaultModel = provider.completion.find((model) =>
          model?.tags?.includes('default_for_indexing'),
        );
        if (defaultModel?.model) {
          defaultCompletionModel = defaultModel.model;
          break;
        }
        // Fallback to first model if no default found
        if (!defaultCompletionModel && provider.completion[0]?.model) {
          defaultCompletionModel = provider.completion[0].model;
        }
      }
    }

    return { defaultEmbeddingModel, defaultCompletionModel };
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
    getDefaultModelsFromAvailable,
    getProviderByModelName,
  };
};
