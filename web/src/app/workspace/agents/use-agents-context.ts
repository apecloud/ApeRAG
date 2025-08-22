'use client';

import { Collection, ModelSpec } from '@/api';
import { createContext, useContext } from 'react';

export type ProviderModels = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
}[];
export type AgentsContextProps = {
  collections: Collection[];
  providerModels: ProviderModels;
};

export const AgentsContext = createContext<AgentsContextProps>({
  collections: [],
  providerModels: [],
});
export const useAgentsContext = () => useContext(AgentsContext);
