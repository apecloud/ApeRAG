'use client';

import { browserApiClient } from '@/lib/api/typed/browser';
import type {
  Model,
  ModelAccount,
  ModelAccountCreateInput,
  ModelAccountUpdateInput,
  ModelCapability,
  ModelCreateInput,
  ModelProvider,
  ModelUse,
  ModelUseScenario,
} from './types';

export async function getModelProviders(): Promise<ModelProvider[]> {
  const { data } = await browserApiClient.GET('/api/v2/model-providers');
  return (data?.items ?? []) as ModelProvider[];
}

export async function getModelAccounts(): Promise<ModelAccount[]> {
  const { data } = await browserApiClient.GET('/api/v2/model-accounts');
  return (data?.items ?? []) as ModelAccount[];
}

export async function createModelAccount(input: ModelAccountCreateInput) {
  const { data } = await browserApiClient.POST('/api/v2/model-accounts', {
    body: input,
  });
  return data;
}

export async function updateModelAccount(
  accountId: string,
  input: ModelAccountUpdateInput,
) {
  const { data } = await browserApiClient.PUT(
    '/api/v2/model-accounts/{account_id}',
    {
      params: { path: { account_id: accountId } },
      body: input,
    },
  );
  return data;
}

export async function deleteModelAccount(accountId: string) {
  await browserApiClient.DELETE('/api/v2/model-accounts/{account_id}', {
    params: { path: { account_id: accountId } },
  });
}

export async function validateModelAccount(accountId: string) {
  const { data } = await browserApiClient.POST(
    '/api/v2/model-accounts/{account_id}/validate',
    {
      params: { path: { account_id: accountId } },
    },
  );
  return data;
}

export async function getModels(): Promise<Model[]> {
  const { data } = await browserApiClient.GET('/api/v2/models');
  return (data?.items ?? []) as Model[];
}

export async function createModel(input: ModelCreateInput) {
  const { data } = await browserApiClient.POST('/api/v2/models', {
    body: input as never,
  });
  return data;
}

export async function getModelUses(): Promise<ModelUse[]> {
  const { data } = await browserApiClient.GET('/api/v2/model-uses');
  return (data?.items ?? []) as ModelUse[];
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
  const { data } = await browserApiClient.PUT('/api/v2/model-uses/{scenario}', {
    params: { path: { scenario } },
    body: {
      strategy: 'single',
      enabled: input.enabled ?? true,
      primary_model_id: input.primary_model_id,
      fallback_model_ids: input.fallback_model_ids ?? [],
      capability: input.capability,
    } as never,
  });
  return data;
}

export async function getAvailableModels(capabilities: ModelCapability[] = []) {
  const models = await getModels();
  if (!capabilities.length) return models;
  return models.filter((model) => capabilities.includes(model.capability));
}
