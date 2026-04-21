import {
  RebuildIndexesRequestIndexTypesEnum,
  SearchResultItem,
  SearchResultItemRecallTypeEnum,
} from '@/api';

const hiddenCollectionConfigKeys = new Set([
  'config.enable_summary',
  'config.enable_vision',
]);

const hiddenDocumentIndexTypes = new Set<string>([
  RebuildIndexesRequestIndexTypesEnum.SUMMARY,
  RebuildIndexesRequestIndexTypesEnum.VISION,
]);

const hiddenSearchRecallTypes = new Set<string>([
  SearchResultItemRecallTypeEnum.summary_search,
  SearchResultItemRecallTypeEnum.vision_search,
]);

export const isVisibleCollectionConfigKey = (key: string) =>
  !hiddenCollectionConfigKeys.has(key);

export const isVisibleDocumentIndexType = (indexType: string) =>
  !hiddenDocumentIndexTypes.has(indexType);

export const filterVisibleSearchItems = (items?: SearchResultItem[]) =>
  (items || []).filter(
    (item) => !hiddenSearchRecallTypes.has(item.recall_type || ''),
  );
