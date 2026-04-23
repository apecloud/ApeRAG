import { createServerApiClient } from '@/lib/api/typed/server';
import type {
  Provider,
  ProviderCatalogViewModel,
  ProviderModel,
} from './types';

type ProviderApiClient = {
  GET<T>(path: string, options?: unknown): Promise<{ data?: T }>;
};

export async function getProviderCatalog(): Promise<ProviderCatalogViewModel> {
  const client = (await createServerApiClient()) as unknown as ProviderApiClient;
  const { data } = await client.GET<ProviderCatalogViewModel>(
    '/api/v1/llm_configuration',
  );

  return {
    providers: data?.providers ?? [],
    models: data?.models ?? [],
  };
}

export async function getProvider(providerName: string) {
  const client = (await createServerApiClient()) as unknown as ProviderApiClient;
  const { data } = await client.GET<Provider>(
    '/api/v1/llm_providers/{provider_name}',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data as Provider;
}

export async function getProviderModels(providerName: string) {
  const client = (await createServerApiClient()) as unknown as ProviderApiClient;
  const { data } = await client.GET<{ items?: ProviderModel[] }>(
    '/api/v1/llm_providers/{provider_name}/models',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data?.items ?? [];
}
