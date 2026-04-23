'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  MineruTokenTestResponse,
  QuotaUpdateRequest,
  QuotaUpdateResponse,
  Settings,
  SystemDefaultQuotasResponse,
  SystemDefaultQuotasUpdateRequest,
  SystemDefaultQuotasUpdateResponse,
  UserQuotaInfo,
  UserQuotaList,
} from './types';

// Single canonical surface for admin control-plane client calls: settings,
// parser/MinerU token test, admin-side system default quotas, per-user
// quota CRUD + recalculate. Workspace-side quota (`features/quota/*`) is
// kept separate per design-lock msg=5f0a370b decision 2e.

export async function updateSettings(input: Settings): Promise<Settings> {
  const { data } = await browserApiClient.PUT('/api/v1/settings', {
    body: input,
  });
  if (!data) {
    throw new Error('updateSettings: empty response body');
  }
  return data;
}

// Note: `Settings_test_mineru_token` response body is typed as `unknown`
// in the public OpenAPI spec, but the backend returns the
// `MineruTokenTestResponse` shape at runtime. Cast at the adapter
// boundary so callers get the concrete type. Phase 4 governance may
// tighten the schema to remove this cast.
export async function testMineruToken(
  token: string,
): Promise<MineruTokenTestResponse> {
  const { data } = await browserApiClient.POST(
    '/api/v1/settings/test_mineru_token',
    {
      body: { token },
    },
  );
  if (!data) {
    throw new Error('testMineruToken: empty response body');
  }
  return data as MineruTokenTestResponse;
}

export async function updateSystemDefaultQuotas(
  input: SystemDefaultQuotasUpdateRequest,
): Promise<SystemDefaultQuotasUpdateResponse> {
  const { data } = await browserApiClient.PUT(
    '/api/v1/system/default-quotas',
    {
      body: input,
    },
  );
  if (!data) {
    throw new Error('updateSystemDefaultQuotas: empty response body');
  }
  return data;
}

export async function getSystemDefaultQuotas(): Promise<SystemDefaultQuotasResponse> {
  const { data } = await browserApiClient.GET(
    '/api/v1/system/default-quotas',
    {},
  );
  if (!data) {
    throw new Error('getSystemDefaultQuotas: empty response body');
  }
  return data;
}

// `GET /api/v1/quotas` returns `UserQuotaInfo | UserQuotaList` depending on
// caller role (workspace user sees own quota, admin sees full list). Admin
// adapter narrows to the list shape; if the backend returns a single user
// info (caller is not admin), that is a contract violation for this adapter
// and we surface an error rather than silently treat it as an empty list.
export async function listUserQuotas(): Promise<UserQuotaList> {
  const { data } = await browserApiClient.GET('/api/v1/quotas', {});
  if (!data) {
    throw new Error('listUserQuotas: empty response body');
  }
  if (!('items' in data)) {
    throw new Error(
      'listUserQuotas: expected UserQuotaList shape, got UserQuotaInfo',
    );
  }
  return data;
}

export async function getUserQuota(userId: string): Promise<UserQuotaInfo> {
  const { data } = await browserApiClient.GET('/api/v1/quotas', {
    params: { query: { user_id: userId } },
  });
  if (!data) {
    throw new Error('getUserQuota: empty response body');
  }
  if ('items' in data) {
    throw new Error(
      'getUserQuota: expected UserQuotaInfo shape, got UserQuotaList',
    );
  }
  return data;
}

export async function updateUserQuota(
  userId: string,
  input: QuotaUpdateRequest,
): Promise<QuotaUpdateResponse> {
  const { data } = await browserApiClient.PUT(
    '/api/v1/quotas/{user_id}',
    {
      params: { path: { user_id: userId } },
      body: input,
    },
  );
  if (!data) {
    throw new Error('updateUserQuota: empty response body');
  }
  return data;
}

// Backend currently returns `unknown` (no typed schema for recalculate
// response); adapter exposes the raw JSON and lets callers decide how to
// interpret it. If Phase 4 governance formalizes the response shape, tighten
// the return type.
export async function recalculateUserQuota(userId: string): Promise<unknown> {
  const { data } = await browserApiClient.POST(
    '/api/v1/quotas/{user_id}/recalculate',
    {
      params: { path: { user_id: userId } },
    },
  );
  return data;
}
