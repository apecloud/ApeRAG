'use client';

import { browserApiClient } from '@/lib/api/typed/browser';
import type {
  Model,
  ModelAccount,
  ModelAccountCreateInput,
  ModelCapability,
  ModelCreateInput,
  ModelProvider,
  ModelUse,
  ModelUseScenario,
} from './types';

const api = browserApiClient as any;

export async function getModelProviders(): Promise<ModelProvider[]> {
  const { data } = await api.GET('/api/v3/model-providers');
  return data?.items ?? [];
}

export async function getModelAccounts(): Promise<ModelAccount[]> {
  const { data } = await api.GET('/api/v3/model-accounts');
  return data?.items ?? [];
}

export async function createModelAccount(input: ModelAccountCreateInput) {
  const { data } = await api.POST('/api/v3/model-accounts', { body: input });
  return data;
}

export async function validateModelAccount(accountId: string) {
  const { data } = await api.POST('/api/v3/model-accounts/{account_id}/validate', {
    params: { path: { account_id: accountId } },
  });
  return data;
}

export async function getModels(): Promise<Model[]> {
  const { data } = await api.GET('/api/v3/models');
  return data?.items ?? [];
}

export async function createModel(input: ModelCreateInput) {
  const { data } = await api.POST('/api/v3/models', { body: input });
  return data;
}

export async function getModelUses(): Promise<ModelUse[]> {
  const { data } = await api.GET('/api/v3/model-uses');
  return data?.items ?? [];
}

export async function updateModelUse(
  scenario: ModelUseScenario,
  input: {
    capability: ModelCapability;
    primary_model_id?: string | null;
    fallback_model_ids?: string[];
    enabled?: boolean;
  },
) {
  const { data } = await api.PUT('/api/v3/model-uses/{scenario}', {
    params: { path: { scenario } },
    body: {
      strategy: 'single',
      enabled: input.enabled ?? true,
      primary_model_id: input.primary_model_id,
      fallback_model_ids: input.fallback_model_ids ?? [],
      capability: input.capability,
    },
  });
  return data;
}

export async function getAvailableModels(capabilities: ModelCapability[] = []) {
  const models = await getModels();
  if (!capabilities.length) return models;
  return models.filter((model) => capabilities.includes(model.capability));
}
