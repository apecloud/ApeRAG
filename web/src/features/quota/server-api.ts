import { createServerApiClient } from '@/lib/api/typed/server';

import type { UserQuotaInfo } from './types';

// The typed `GET /api/v2/quotas` 200 response is a
// `UserQuotaInfo | UserQuotaList` union because the endpoint is shared
// between the workspace view (current user) and the admin list view
// (admin query params). TypeScript cannot narrow the union by query
// presence, so narrow at the adapter boundary: workspace callers never
// pass admin query params, so the server must return the single-user
// shape. If that contract is ever violated, throw explicitly — silently
// rendering an admin list would be a user-facing regression. Phase 8 #66
// restores the same-shape quota contract on `/api/v2`; any future REST
// split belongs in a separate quota API redesign.
export async function getUserQuota(): Promise<UserQuotaInfo> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v2/quotas', {});
  if (!data) {
    throw new Error('workspace quota endpoint returned no body');
  }
  if ('items' in data) {
    throw new Error(
      'workspace quota endpoint returned a list response; expected single user quota',
    );
  }
  return data;
}
