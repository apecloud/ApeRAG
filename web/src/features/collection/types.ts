import type { components } from '@/api-v2/schema';

export type Collection = components['schemas']['Collection'];
export type CollectionCreate = components['schemas']['CollectionCreate'];
export type CollectionUpdate = components['schemas']['CollectionUpdate'];
export type CollectionView = components['schemas']['CollectionView'];
export type CollectionViewList = components['schemas']['CollectionViewList'];

export type CollectionStatus = NonNullable<CollectionView['status']>;

export type SharingStatusResponse =
  components['schemas']['SharingStatusResponse'];
export type CollectionSummaryTriggerResponse =
  components['schemas']['CollectionSummaryTriggerResponse'];

export type MineruTokenTestRequest =
  components['schemas']['MineruTokenTestRequest'];
export type MineruTokenTestResponse =
  components['schemas']['MineruTokenTestResponse'];

export type TitleLanguage = NonNullable<
  components['schemas']['TitleGenerateRequest']['language']
>;

export const TITLE_LANGUAGES = [
  'zh-CN',
  'en-US',
  'ja-JP',
  'ko-KR',
] as const satisfies readonly TitleLanguage[];

export type ExportTaskResponse = components['schemas']['ExportTaskResponse'];
