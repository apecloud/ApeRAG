import type { components } from '@/api-v2/schema';

export type SearchRequest = NonNullable<
  components['schemas']['SearchRequest']
>;
export type SearchResult = NonNullable<components['schemas']['SearchResult']>;
export type SearchResultItem = NonNullable<
  components['schemas']['SearchResultItem']
>;
export type SearchResultList = NonNullable<
  components['schemas']['SearchResultList']
>;
export type SearchResultMetadata = NonNullable<
  components['schemas']['SearchResultMetadata']
>;

export type VectorSearchParams = NonNullable<
  components['schemas']['VectorSearchParams']
>;
export type FulltextSearchParams = NonNullable<
  components['schemas']['FulltextSearchParams']
>;
export type GraphSearchParams = NonNullable<
  components['schemas']['GraphSearchParams']
>;
export type SummarySearchParams = NonNullable<
  components['schemas']['SummarySearchParams']
>;
export type VisionSearchParams = NonNullable<
  components['schemas']['VisionSearchParams']
>;

export type SearchRecallType = NonNullable<SearchResultItem['recall_type']>;

// Public recall-type vocabulary the FE actively renders. Kept as a
// `satisfies readonly SearchRecallType[]` list so the compile step
// errors if the backend enum loses a member the UI depends on.
export const SEARCH_RECALL_TYPES = [
  'vector_search',
  'graph_search',
  'fulltext_search',
  'summary_search',
  'vision_search',
] as const satisfies readonly SearchRecallType[];
