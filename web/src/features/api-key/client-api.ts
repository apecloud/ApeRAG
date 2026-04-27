'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type { ApiKeyCreate, ApiKeyUpdate } from './types';

export async function createApiKey(input: ApiKeyCreate) {
  const { data } = await browserApiClient.POST('/api/v1/apikeys', {
    body: input,
  });
  return data;
}

export async function updateApiKey(apikeyId: string, input: ApiKeyUpdate) {
  const { data } = await browserApiClient.PUT('/api/v1/apikeys/{apikey_id}', {
    params: {
      path: { apikey_id: apikeyId },
    },
    body: input,
  });
  return data;
}

export async function deleteApiKey(apikeyId: string) {
  await browserApiClient.DELETE('/api/v1/apikeys/{apikey_id}', {
    params: {
      path: { apikey_id: apikeyId },
    },
  });
}
