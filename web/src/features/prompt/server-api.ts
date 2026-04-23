import { createServerApiClient } from '@/lib/api/typed/server';

import type { UserPromptsResponse } from './types';

export async function getUserPrompts(): Promise<UserPromptsResponse> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v1/prompts/user', {});
  // The typed schema currently types the response as `{ [key: string]: unknown }`.
  // Cast at the adapter boundary so consumers see the runtime shape
  // (see `types.ts` for the technical-debt note).
  return (data ?? {}) as UserPromptsResponse;
}
