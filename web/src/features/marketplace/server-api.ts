import { createServerApiClient } from '@/lib/api/typed/server';

import type {
  MarketplaceDocumentList,
  MarketplaceDocumentPreview,
  SharedCollection,
  SharedCollectionList,
} from './types';

export async function listMarketplaceCollections(options?: {
  page?: number;
  pageSize?: number;
}): Promise<SharedCollectionList> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v1/marketplace/collections', {
    params: {
      query: {
        page: options?.page ?? 1,
        page_size: options?.pageSize ?? 100,
      },
    },
  });
  return data ?? {};
}

export async function getMarketplaceCollection(
  collectionId: string,
): Promise<SharedCollection | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v1/marketplace/collections/{collection_id}',
    {
      params: {
        path: { collection_id: collectionId },
      },
    },
  );
  return data ?? null;
}

export async function listMarketplaceCollectionDocuments(
  collectionId: string,
  options?: {
    page?: number;
    pageSize?: number;
    sortBy?: string;
    sortOrder?: string;
    search?: string;
  },
): Promise<MarketplaceDocumentList> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v1/marketplace/collections/{collection_id}/documents',
    {
      params: {
        path: { collection_id: collectionId },
        query: {
          page: options?.page ?? 1,
          page_size: options?.pageSize ?? 20,
          sort_by: options?.sortBy,
          sort_order: options?.sortOrder,
          search: options?.search,
        },
      },
    },
  );
  return data ?? {};
}

export async function getMarketplaceCollectionDocumentPreview(
  collectionId: string,
  documentId: string,
): Promise<MarketplaceDocumentPreview | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v1/marketplace/collections/{collection_id}/documents/{document_id}/preview',
    {
      params: {
        path: {
          collection_id: collectionId,
          document_id: documentId,
        },
      },
    },
  );
  return data ?? null;
}
