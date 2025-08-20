import { apiClient } from '@/lib/api/client';
import { useCallback, useEffect, useState } from 'react';

export function useModels() {
  const [models, setModels] = useState([]);
  const [agentModels, setAgentModels] = useState([]);
  const [collectionModels, setCollectionModels] = useState([]);
  const [agentModels, setAgentModels] = useState([]);

  const getModels = useCallback(async () => {
    const availableModelsRes = await apiClient.defaultApi.availableModelsPost();
  }, []);

  useEffect(() => {
    getModels();
  }, [getModels]);

  return {
    agentModels,
    collectionModels,
    embeddingModels,
    rerankModels,
    getModels,
  };
}
