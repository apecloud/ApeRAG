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
  ModelUpdateInput,
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

export async function getModels(input?: {
  capability?: ModelCapability;
  scenario?: ModelUseScenario;
}): Promise<Model[]> {
  const { data } = await browserApiClient.GET('/api/v2/models', {
    params: { query: input },
  });
  return (data?.items ?? []) as Model[];
}

export async function createModel(input: ModelCreateInput) {
  const { data } = await browserApiClient.POST('/api/v2/models', {
    body: input as never,
  });
  return data;
}

export async function updateModel(modelId: string, input: ModelUpdateInput) {
  const { data } = await browserApiClient.PUT('/api/v2/models/{model_id}', {
    params: { path: { model_id: modelId } },
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

const scenarioCapability: Record<ModelUseScenario, ModelCapability> = {
  agent_chat: 'chat',
  collection_completion: 'chat',
  collection_embedding: 'embedding',
  background_task: 'chat',
};

export function getScenarioCapability(scenario: ModelUseScenario) {
  return scenarioCapability[scenario];
}

export function isModelAllowedForScenario(
  model: Pick<Model, 'capability' | 'allowed_scenarios'>,
  scenario: ModelUseScenario,
) {
  return (
    model.capability === scenarioCapability[scenario] &&
    Boolean(model.allowed_scenarios?.includes(scenario))
  );
}

export async function getScenarioModels(scenario: ModelUseScenario) {
  const models = await getModels({ scenario });
  return models.filter((model) => isModelAllowedForScenario(model, scenario));
}
