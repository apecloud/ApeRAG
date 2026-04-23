import { createServerApiClient } from '@/lib/api/typed/server';
import type { User } from '@/features/identity/types';

// `/api/v1/user` is the authenticated "me" endpoint. Used by root / workspace
// / admin layouts to resolve the current user and redirect to signin when
// unauthenticated. `createServerApiClient` throws on non-2xx, so we swallow
// the error and return `null` to preserve the layout's redirect-on-unauth
// control flow.
export async function getCurrentUser(): Promise<User | null> {
  const client = await createServerApiClient();
  try {
    const { data } = await client.GET('/api/v1/user', {});
    return data ?? null;
  } catch {
    return null;
  }
}
