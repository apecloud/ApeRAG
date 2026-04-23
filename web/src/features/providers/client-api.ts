'use client';

import { browserApiClient } from '@/lib/api/typed/browser';
import type {
  DefaultModelConfig,
  ModelConfig,
  ProviderModel,
  ProviderModelApi,
  ProviderModelFormInput,
  ProviderModelUpdateInput,
  ProviderFormInput,
} from './types';

function providerCreatePayload(input: ProviderFormInput) {
  return {
    allow_custom_base_url: false,
    ...input,
  };
}

export async function createProvider(input: ProviderFormInput) {
  const { data } = await browserApiClient.POST('/api/v2/providers', {
    body: providerCreatePayload(input),
  });
  return data;
}

export async function updateProvider(
  providerName: string,
  input: ProviderFormInput,
) {
  const { data } = await browserApiClient.PUT(
    '/api/v2/providers/{provider_name}',
    {
      params: { path: { provider_name: providerName } },
      body: input,
    },
  );
  return data;
}

export async function deleteProvider(providerName: string) {
  await browserApiClient.DELETE('/api/v2/providers/{provider_name}', {
    params: { path: { provider_name: providerName } },
  });
}

export async function publishProvider(providerName: string) {
  const { data } = await browserApiClient.POST(
    '/api/v2/providers/{provider_name}/publish',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data;
}

export async function getProvider(providerName: string) {
  const { data } = await browserApiClient.GET(
    '/api/v2/providers/{provider_name}',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data;
}

export async function getProviderModels(providerName: string) {
  const { data } = await browserApiClient.GET(
    '/api/v2/providers/{provider_name}/models',
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
  const { data } = await browserApiClient.POST(
    '/api/v2/providers/{provider_name}/models',
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
  const { data } = await browserApiClient.PUT(
    '/api/v2/providers/{provider_name}/models/{api}/{model}',
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
  await browserApiClient.DELETE(
    '/api/v2/providers/{provider_name}/models/{api}/{model}',
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
  const { data } = await browserApiClient.GET('/api/v2/default-models', {});
  return data?.items ?? [];
}

export async function updateDefaultModels(defaults: DefaultModelConfig[]) {
  const { data } = await browserApiClient.PUT('/api/v2/default-models', {
    body: { defaults },
  });
  return data?.items ?? [];
}

export async function getAvailableModels(tagFilters: string[][]) {
  const { data } = await browserApiClient.POST(
    '/api/v2/providers/available-models',
    {
      body: {
        tag_filters: tagFilters.map((tags) => ({
          operation: 'AND' as const,
          tags,
        })),
      },
    },
  );
  return (data?.items ?? []) as ModelConfig[];
}
