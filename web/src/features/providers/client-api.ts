'use client';

import { browserApiClient } from '@/lib/api/typed/browser';
import type {
  DefaultModelConfig,
  ModelConfig,
  Provider,
  ProviderModel,
  ProviderModelApi,
  ProviderModelFormInput,
  ProviderModelUpdateInput,
  ProviderFormInput,
} from './types';

type ProviderApiClient = {
  GET<T>(path: string, options?: unknown): Promise<{ data?: T }>;
  POST<T>(path: string, options?: unknown): Promise<{ data?: T }>;
  PUT<T>(path: string, options?: unknown): Promise<{ data?: T }>;
  DELETE(path: string, options?: unknown): Promise<{ data?: unknown }>;
};

const providerApiClient = browserApiClient as unknown as ProviderApiClient;

function providerCreatePayload(input: ProviderFormInput) {
  return {
    allow_custom_base_url: false,
    ...input,
  };
}

export async function createProvider(input: ProviderFormInput) {
  const { data } = await providerApiClient.POST<Provider>('/api/v1/llm_providers', {
    body: providerCreatePayload(input),
  });
  return data;
}

export async function updateProvider(
  providerName: string,
  input: ProviderFormInput,
) {
  const { data } = await providerApiClient.PUT<Provider>(
    '/api/v1/llm_providers/{provider_name}',
    {
      params: { path: { provider_name: providerName } },
      body: input,
    },
  );
  return data;
}

export async function deleteProvider(providerName: string) {
  await providerApiClient.DELETE('/api/v1/llm_providers/{provider_name}', {
    params: { path: { provider_name: providerName } },
  });
}

export async function publishProvider(providerName: string) {
  const { data } = await providerApiClient.POST<Provider>(
    '/api/v1/llm_providers/{provider_name}/publish',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data;
}

export async function getProvider(providerName: string) {
  const { data } = await providerApiClient.GET<Provider>(
    '/api/v1/llm_providers/{provider_name}',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data;
}

export async function getProviderModels(providerName: string) {
  const { data } = await providerApiClient.GET<{ items?: ProviderModel[] }>(
    '/api/v1/llm_providers/{provider_name}/models',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data?.items ?? [];
}

export async function createProviderModel(
  providerName: string,
  input: ProviderModelFormInput,
) {
  const { data } = await providerApiClient.POST<ProviderModel>(
    '/api/v1/llm_providers/{provider_name}/models',
    {
      params: { path: { provider_name: providerName } },
      body: input,
    },
  );
  return data;
}

export async function updateProviderModel(
  providerName: string,
  api: ProviderModelApi,
  model: string,
  input: ProviderModelUpdateInput,
) {
  const { data } = await providerApiClient.PUT<ProviderModel>(
    '/api/v1/llm_providers/{provider_name}/models/{api}/{model}',
    {
      params: {
        path: { provider_name: providerName, api, model },
      },
      body: input,
    },
  );
  return data;
}

export async function deleteProviderModel(
  providerName: string,
  api: ProviderModelApi,
  model: string,
) {
  await providerApiClient.DELETE(
    '/api/v1/llm_providers/{provider_name}/models/{api}/{model}',
    {
      params: {
        path: { provider_name: providerName, api, model },
      },
    },
  );
}

export async function updateProviderModelTags(
  providerName: string,
  model: ProviderModel,
  tags: string[],
) {
  return updateProviderModel(providerName, model.api, model.model, {
    custom_llm_provider: model.custom_llm_provider,
    context_window: model.context_window,
    max_input_tokens: model.max_input_tokens,
    max_output_tokens: model.max_output_tokens,
    tags,
  });
}

export async function getDefaultModels() {
  const { data } = await browserApiClient.GET('/api/v1/default_models', {});
  return data?.items ?? [];
}

export async function updateDefaultModels(defaults: DefaultModelConfig[]) {
  const { data } = await browserApiClient.PUT('/api/v1/default_models', {
    body: { defaults },
  });
  return data?.items ?? [];
}

export async function getAvailableModels(tagFilters: string[][]) {
  const { data } = await browserApiClient.POST('/api/v1/available_models', {
    body: {
      tag_filters: tagFilters.map((tags) => ({
        operation: 'AND' as const,
        tags,
      })),
    },
  });
  return (data?.items ?? []) as ModelConfig[];
}
