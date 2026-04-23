'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  Collection,
  CollectionCreate,
  CollectionSummaryTriggerResponse,
  CollectionUpdate,
  CollectionViewList,
  MineruTokenTestRequest,
  MineruTokenTestResponse,
  SharingStatusResponse,
} from './types';

export async function createCollection(
  input: CollectionCreate,
): Promise<Collection | undefined> {
  const { data } = await browserApiClient.POST('/api/v2/collections', {
    body: input,
  });
  return data;
}

export async function listCollections(options?: {
  page?: number;
  pageSize?: number;
  includeSubscribed?: boolean;
}): Promise<CollectionViewList | undefined> {
  const { data } = await browserApiClient.GET('/api/v2/collections', {
    params: {
      query: {
        page: options?.page ?? 1,
        page_size: options?.pageSize ?? 50,
        include_subscribed: options?.includeSubscribed ?? true,
      },
    },
  });
  return data;
}

export async function getCollection(
  collectionId: string,
): Promise<Collection | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data;
}

export async function updateCollection(
  collectionId: string,
  input: CollectionUpdate,
): Promise<Collection | undefined> {
  const { data } = await browserApiClient.PUT(
    '/api/v2/collections/{collection_id}',
    {
      params: { path: { collection_id: collectionId } },
      body: input,
    },
  );
  return data;
}

export async function deleteCollection(collectionId: string): Promise<void> {
  await browserApiClient.DELETE('/api/v2/collections/{collection_id}', {
    params: { path: { collection_id: collectionId } },
  });
}

export async function triggerCollectionSummary(
  collectionId: string,
): Promise<CollectionSummaryTriggerResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/summary/generate',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data;
}

export async function testMineruToken(
  input: MineruTokenTestRequest,
): Promise<MineruTokenTestResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/test-mineru-token',
    {
      body: input,
    },
  );
  return data;
}

export async function getCollectionSharingStatus(
  collectionId: string,
): Promise<SharingStatusResponse | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/sharing',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data;
}

export async function publishCollectionSharing(
  collectionId: string,
): Promise<void> {
  await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/sharing',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
}

export async function unpublishCollectionSharing(
  collectionId: string,
): Promise<void> {
  await browserApiClient.DELETE(
    '/api/v2/collections/{collection_id}/sharing',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
}
