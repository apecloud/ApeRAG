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
  const page = options?.page ?? 1;
  const pageSize = options?.pageSize ?? 100;
  const { data } = await client.GET('/api/v1/marketplace/collections', {
    params: {
      query: {
        page,
        page_size: pageSize,
      },
    },
  });
  // `SharedCollectionList` has `items/total/page/page_size` all required;
  // absent body means "no collections yet", so return a typed empty list
  // rather than `{}` (which would violate the type contract and silently
  // mask a malformed response with missing pagination).
  return (
    data ?? {
      items: [],
      total: 0,
      page,
      page_size: pageSize,
    }
  );
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
  const page = options?.page ?? 1;
  const pageSize = options?.pageSize ?? 20;
  const { data } = await client.GET(
    '/api/v1/marketplace/collections/{collection_id}/documents',
    {
      params: {
        path: { collection_id: collectionId },
        query: {
          page,
          page_size: pageSize,
          sort_by: options?.sortBy,
          sort_order: options?.sortOrder,
          search: options?.search,
        },
      },
    },
  );
  // `DocumentList` fields are all optional, so `{}` is technically valid,
  // but fall back to a typed empty list for consistency with
  // `listMarketplaceCollections` and so consumers can rely on `items`
  // being iterable.
  return (
    data ?? {
      items: [],
      total: 0,
      page,
      page_size: pageSize,
      total_pages: 0,
    }
  );
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
