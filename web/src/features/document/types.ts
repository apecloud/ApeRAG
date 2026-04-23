import type { components } from '@/api-v2/schema';

export type Document = components['schemas']['Document'];
export type DocumentList = components['schemas']['DocumentList'];
export type DocumentPreview = components['schemas']['DocumentPreview'];

export type DocumentStatus = NonNullable<Document['status']>;
export type DocumentIndexStatus = NonNullable<Document['vector_index_status']>;

export type RebuildIndexesRequest =
  components['schemas']['RebuildIndexesRequest'];
export type RebuildIndexesResponse =
  components['schemas']['RebuildIndexesResponse'];

export type DocumentIndexType = RebuildIndexesRequest['index_types'][number];

export const DOCUMENT_INDEX_TYPES = [
  'VECTOR',
  'FULLTEXT',
  'GRAPH',
  'SUMMARY',
  'VISION',
] as const satisfies readonly DocumentIndexType[];

export type DeleteDocumentsRequest =
  components['schemas']['DeleteDocumentsRequest'];
export type DeleteDocumentsResponse =
  components['schemas']['DeleteDocumentsResponse'];
