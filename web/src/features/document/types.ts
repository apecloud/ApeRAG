import type { components } from '@/api-v2/schema';

export type Document = components['schemas']['Document'];
export type DocumentList = components['schemas']['DocumentList'];
export type DocumentPreview = components['schemas']['DocumentPreview'];

export type RebuildIndexesRequest =
  components['schemas']['RebuildIndexesRequest'];
export type RebuildIndexesResponse =
  components['schemas']['RebuildIndexesResponse'];

export type DeleteDocumentsRequest =
  components['schemas']['DeleteDocumentsRequest'];
export type DeleteDocumentsResponse =
  components['schemas']['DeleteDocumentsResponse'];
