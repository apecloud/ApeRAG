'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  DeleteDocumentsRequest,
  DeleteDocumentsResponse,
  Document,
  DocumentList,
  DocumentPreview,
  RebuildIndexesRequest,
  RebuildIndexesResponse,
} from './types';

export type ListDocumentsOptions = {
  page?: number;
  pageSize?: number;
  sortBy?: 'name' | 'created' | 'updated' | 'size' | 'status';
  sortOrder?: 'asc' | 'desc';
  search?: string;
};

export async function listDocuments(
  collectionId: string,
  options?: ListDocumentsOptions,
): Promise<DocumentList | undefined> {
  const { data } = await browserApiClient.GET(
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
  return data;
}

export async function getDocument(
  collectionId: string,
  documentId: string,
): Promise<Document | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/documents/{document_id}',
    {
      params: {
        path: { collection_id: collectionId, document_id: documentId },
      },
    },
  );
  return data;
}

export async function getDocumentPreview(
  collectionId: string,
  documentId: string,
): Promise<DocumentPreview | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/documents/{document_id}/preview',
    {
      params: {
        path: { collection_id: collectionId, document_id: documentId },
      },
    },
  );
  return data;
}

export async function deleteDocument(
  collectionId: string,
  documentId: string,
): Promise<void> {
  await browserApiClient.DELETE(
    '/api/v2/collections/{collection_id}/documents/{document_id}',
    {
      params: {
        path: { collection_id: collectionId, document_id: documentId },
      },
    },
  );
}

export async function deleteDocuments(
  collectionId: string,
  input: DeleteDocumentsRequest,
): Promise<DeleteDocumentsResponse | undefined> {
  const { data } = await browserApiClient.DELETE(
    '/api/v2/collections/{collection_id}/documents',
    {
      params: { path: { collection_id: collectionId } },
      body: input,
    },
  );
  return data;
}

export async function rebuildDocumentIndexes(
  collectionId: string,
  documentId: string,
  input: RebuildIndexesRequest,
): Promise<RebuildIndexesResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/documents/{document_id}/rebuild_indexes',
    {
      params: {
        path: { collection_id: collectionId, document_id: documentId },
      },
      body: input,
    },
  );
  return data;
}

export async function rebuildFailedDocumentIndexes(
  collectionId: string,
): Promise<RebuildIndexesResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/rebuild_failed_indexes',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data;
}

// Binary routes (download / object) are consumed directly as URLs by the
// browser or embedded viewers (e.g. react-pdf), not through the typed JSON
// client. Keep the URL shape as first-class helpers so all call sites agree
// on the v2 path and the `path` query param on /object.

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function buildDocumentDownloadUrl(
  collectionId: string,
  documentId: string,
): string {
  return `${BASE_PATH}/api/v2/collections/${collectionId}/documents/${documentId}/download`;
}

export function buildDocumentObjectUrl(
  collectionId: string,
  documentId: string,
  objectPath: string,
): string {
  const query = new URLSearchParams({ path: objectPath });
  return `${BASE_PATH}/api/v2/collections/${collectionId}/documents/${documentId}/object?${query.toString()}`;
}
