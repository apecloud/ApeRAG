import { createServerApiClient } from '@/lib/api/typed/server';

import type { Document, DocumentList, DocumentPreview } from './types';

export type ListDocumentsServerOptions = {
  page?: number;
  pageSize?: number;
  sortBy?: 'name' | 'created' | 'updated' | 'size' | 'status';
  sortOrder?: 'asc' | 'desc';
  search?: string;
};

export async function listDocuments(
  collectionId: string,
  options?: ListDocumentsServerOptions,
): Promise<DocumentList> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/documents',
    {
      params: {
        path: { collection_id: collectionId },
        query: {
          page: options?.page ?? 1,
          page_size: options?.pageSize ?? 10,
          sort_by: options?.sortBy ?? 'created',
          sort_order: options?.sortOrder ?? 'desc',
          search: options?.search,
        },
      },
    },
  );
  return data ?? {};
}

export async function getDocument(
  collectionId: string,
  documentId: string,
): Promise<Document | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/documents/{document_id}',
    {
      params: {
        path: { collection_id: collectionId, document_id: documentId },
      },
    },
  );
  return data ?? null;
}

export async function getDocumentPreview(
  collectionId: string,
  documentId: string,
): Promise<DocumentPreview | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/documents/{document_id}/preview',
    {
      params: {
        path: { collection_id: collectionId, document_id: documentId },
      },
    },
  );
  return data ?? null;
}
