import { createServerApiClient } from '@/lib/api/typed/server';

import type { SearchResultList } from './types';

export async function listSearches(
  collectionId: string,
): Promise<SearchResultList> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/searches',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  // Lesson 9a: always return the typed empty shape, never `null` / `{}`.
  return data ?? { items: [] };
}
