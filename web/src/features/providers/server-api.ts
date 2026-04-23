import { createServerApiClient } from '@/lib/api/typed/server';
import type {
  Provider,
  ProviderCatalogViewModel,
} from './types';

export async function getProviderCatalog(): Promise<ProviderCatalogViewModel> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/providers/configuration',
  );

  return {
    providers: data?.providers ?? [],
    models: data?.models ?? [],
  };
}

export async function getProvider(providerName: string) {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/providers/{provider_name}',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data as Provider;
}

export async function getProviderModels(providerName: string) {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/providers/{provider_name}/models',
    {
      params: { path: { provider_name: providerName } },
    },
  );
  return data?.items ?? [];
}
