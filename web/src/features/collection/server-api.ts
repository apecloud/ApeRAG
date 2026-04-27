import { createServerApiClient } from '@/lib/api/typed/server';

import type {
  Collection,
  CollectionViewList,
  SharingStatusResponse,
} from './types';

export async function listCollections(options?: {
  page?: number;
  pageSize?: number;
  includeSubscribed?: boolean;
}): Promise<CollectionViewList> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v2/collections', {
    params: {
      query: {
        page: options?.page ?? 1,
        page_size: options?.pageSize ?? 50,
        include_subscribed: options?.includeSubscribed ?? true,
      },
    },
  });
  return data ?? {};
}

export async function getCollection(
  collectionId: string,
): Promise<Collection | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v2/collections/{collection_id}', {
    params: { path: { collection_id: collectionId } },
  });
  return data ?? null;
}

export async function getCollectionSharingStatus(
  collectionId: string,
): Promise<SharingStatusResponse | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/sharing',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data ?? null;
}
