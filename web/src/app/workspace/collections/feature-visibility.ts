import type { DocumentIndexType } from '@/features/document/types';
import type {
  SearchRecallType,
  SearchResultItem,
} from '@/features/retrieval/types';

const hiddenCollectionConfigKeys = new Set([
  'config.enable_summary',
  'config.enable_vision',
]);

const hiddenDocumentIndexTypes = new Set<DocumentIndexType>([
  'SUMMARY',
  'VISION',
]);

const hiddenSearchRecallTypes = new Set<SearchRecallType>([
  'summary_search',
  'vision_search',
]);

export const isVisibleCollectionConfigKey = (key: string) =>
  !hiddenCollectionConfigKeys.has(key);

export const isVisibleDocumentIndexType = (indexType: string) =>
  !hiddenDocumentIndexTypes.has(indexType as DocumentIndexType);

export const filterVisibleSearchItems = (items?: SearchResultItem[]) =>
  (items || []).filter(
    (item) => !hiddenSearchRecallTypes.has(item.recall_type as SearchRecallType),
  );
