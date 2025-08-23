'use client';

import { apiClient } from '@/lib/api/client';
import { useCallback, useEffect, useState } from 'react';

import { Collection, ModelSpec } from '@/api';
import { createContext, useContext } from 'react';

export type ProviderModels = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
}[];

type AgentsContextProps = {
  collections: Collection[];
  providerModels: ProviderModels;
};

const AgentsContext = createContext<AgentsContextProps>({
  collections: [],
  providerModels: [],
});

export const useAgentsContext = () => useContext(AgentsContext);

export const AgentsProvider = ({
  children,
}: {
  children?: React.ReactNode;
}) => {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [providerModels, setProviderModels] = useState<ProviderModels>([]);

  const loadData = useCallback(async () => {
    const [modelRes, collectionsRes] = await Promise.all([
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
        },
      }),
      apiClient.defaultApi.collectionsGet(),
    ]);

    const items = modelRes.data.items?.map((m) => {
      return {
        label: m.label,
        name: m.name,
        models: m.completion,
      };
    });
    setCollections(collectionsRes.data.items || []);
    setProviderModels(items || []);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <AgentsContext.Provider value={{ collections, providerModels }}>
      {children}
    </AgentsContext.Provider>
  );
};
