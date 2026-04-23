'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  ConfirmDocumentsResponse,
  DeleteDocumentsRequest,
  DeleteDocumentsResponse,
  Document,
  DocumentList,
  DocumentPreview,
  FetchUrlResponse,
  RebuildIndexesRequest,
  RebuildIndexesResponse,
  StagedDocumentsResponse,
  UploadDocumentResponse,
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

// Upload flow (staged list / upload / confirm / fetch-url). Phase 1b typed
// wrap: these wrap the still-v1 endpoints without renaming the path. All
// four returns are single-resource shapes with required fields, so an empty
// body means the backend broke its contract — the adapter throws instead
// of returning a typed-shape-missing `{}` (new server/client adapter gate,
// lesson 9a).

export async function listStagedDocuments(
  collectionId: string,
  signal?: AbortSignal,
): Promise<StagedDocumentsResponse> {
  const { data } = await browserApiClient.GET(
    '/api/v1/collections/{collection_id}/documents/staged',
    {
      params: { path: { collection_id: collectionId } },
      signal,
    },
  );
  if (!data) {
    throw new Error('listStagedDocuments: empty response body');
  }
  return data;
}

// `openapi-fetch` 0.17 does not auto-encode `body: { file }` as
// `multipart/form-data`; we supply an explicit `bodySerializer` that builds
// a `FormData` and lets `fetch` set the correct `Content-Type` boundary.
// The body type literally matches the generated
// `Body_documents_upload_document_view` (`{ file: string /* binary */ }`),
// but at runtime we pass a real `File` under `file`; that's why we cast to
// `unknown` to satisfy the typed contract and then read `body.file` as the
// actual File inside the serializer.
export async function uploadDocument(
  collectionId: string,
  file: File,
  options?: { signal?: AbortSignal; timeout?: number },
): Promise<UploadDocumentResponse> {
  const { data } = await browserApiClient.POST(
    '/api/v1/collections/{collection_id}/documents/upload',
    {
      params: { path: { collection_id: collectionId } },
      body: { file: file as unknown as string },
      bodySerializer: (body) => {
        const fd = new FormData();
        fd.append('file', (body as unknown as { file: File }).file);
        return fd;
      },
      signal: options?.signal,
    },
  );
  if (!data) {
    throw new Error('uploadDocument: empty response body');
  }
  return data;
}

export async function confirmDocuments(
  collectionId: string,
  documentIds: string[],
): Promise<ConfirmDocumentsResponse> {
  const { data } = await browserApiClient.POST(
    '/api/v1/collections/{collection_id}/documents/confirm',
    {
      params: { path: { collection_id: collectionId } },
      body: { document_ids: documentIds },
    },
  );
  if (!data) {
    throw new Error('confirmDocuments: empty response body');
  }
  return data;
}

export async function fetchUrlDocuments(
  collectionId: string,
  urls: string[],
  options?: { signal?: AbortSignal },
): Promise<FetchUrlResponse> {
  const { data } = await browserApiClient.POST(
    '/api/v1/collections/{collection_id}/documents/fetch-url',
    {
      params: { path: { collection_id: collectionId } },
      body: { urls },
      signal: options?.signal,
    },
  );
  if (!data) {
    throw new Error('fetchUrlDocuments: empty response body');
  }
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
