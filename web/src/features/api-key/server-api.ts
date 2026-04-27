import { createServerApiClient } from '@/lib/api/typed/server';

import type { ApiKey } from './types';

export async function listApiKeys(): Promise<ApiKey[]> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v1/apikeys', {});
  return data?.items ?? [];
}
