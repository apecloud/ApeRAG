import {
  RebuildIndexesRequestIndexTypesEnum,
  SearchResultItem,
  SearchResultItemRecallTypeEnum,
} from '@/api';

const hiddenCollectionConfigKeys = new Set([
  'config.enable_summary',
  'config.enable_vision',
]);

const hiddenDocumentIndexTypes = new Set([
  RebuildIndexesRequestIndexTypesEnum.SUMMARY,
  RebuildIndexesRequestIndexTypesEnum.VISION,
]);

const hiddenSearchRecallTypes = new Set([
  SearchResultItemRecallTypeEnum.summary_search,
  SearchResultItemRecallTypeEnum.vision_search,
]);

export const isVisibleCollectionConfigKey = (key: string) =>
  !hiddenCollectionConfigKeys.has(key);

export const isVisibleDocumentIndexType = (indexType: string) =>
  !hiddenDocumentIndexTypes.has(
    indexType as (typeof RebuildIndexesRequestIndexTypesEnum)[keyof typeof RebuildIndexesRequestIndexTypesEnum],
  );

export const filterVisibleSearchItems = (items?: SearchResultItem[]) =>
  (items || []).filter(
    (item) =>
      !hiddenSearchRecallTypes.has(
        item.recall_type as (typeof SearchResultItemRecallTypeEnum)[keyof typeof SearchResultItemRecallTypeEnum],
      ),
  );
