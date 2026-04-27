'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type { SearchRequest, SearchResult, SearchResultList } from './types';

export async function listSearches(
  collectionId: string,
): Promise<SearchResultList | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/searches',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data;
}

export async function createSearch(
  collectionId: string,
  input: SearchRequest,
): Promise<SearchResult | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/searches',
    {
      params: { path: { collection_id: collectionId } },
      body: input,
    },
  );
  return data;
}

export async function deleteSearch(
  collectionId: string,
  searchId: string,
): Promise<void> {
  await browserApiClient.DELETE(
    '/api/v2/collections/{collection_id}/searches/{search_id}',
    {
      params: {
        path: { collection_id: collectionId, search_id: searchId },
      },
    },
  );
}
