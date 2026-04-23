import { createServerApiClient } from '@/lib/api/typed/server';
import type { UserList } from '@/features/identity/types';

import type { Settings, SystemDefaultQuotasResponse } from './types';

// Admin control-plane server-side reads. Workspace-side reads stay in their
// own feature (`features/quota/server-api::getUserQuota`); this file does
// NOT re-export workspace surfaces.

export async function getSettings(): Promise<Settings | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v1/settings', {});
  return data ?? null;
}

export async function getSystemDefaultQuotas(): Promise<SystemDefaultQuotasResponse | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v1/system/default-quotas', {});
  return data ?? null;
}

export async function listUsers(): Promise<UserList> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v1/users', {});
  return (
    data ?? {
      items: [],
      pageResult: null,
    }
  );
}
