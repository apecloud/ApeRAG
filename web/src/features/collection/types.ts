import type { components } from '@/api-v2/schema';

export type CollectionCreate = components['schemas']['CollectionCreate'];
export type CollectionUpdate = components['schemas']['CollectionUpdate'];
export type CollectionView = components['schemas']['CollectionView'];
export type CollectionViewList = components['schemas']['CollectionViewList'];

// Backend ``Collection`` schema was removed (Wave 8 — renamed/refactored
// to ``CollectionView`` as the read-side view shape). Existing FE call
// sites (collection-form / search-table / search-test / collection-
// provider) still expect the full ``Collection`` shape including
// ``config`` (CollectionConfig with enable_fulltext / enable_kg / etc.)
// which ``CollectionView`` does not surface. Define a permissive local
// type that covers ``CollectionView`` plus the legacy ``config`` field
// so the FE compiles; index signature absorbs further drift (TODO
// W9-4: align FE call sites to ``CollectionView`` + a separate
// ``CollectionConfigView`` instead of one combined shape).
export type Collection = Partial<CollectionView> & {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config?: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

export type CollectionStatus = NonNullable<CollectionView['status']>;

export type SharingStatusResponse =
  components['schemas']['SharingStatusResponse'];
// Wave 10 §K.13 Chunk D — explicit operator override regen response
// for the ``/summary/regen`` and ``/description/regen`` endpoints. The
// pre-Wave-10 ``CollectionSummaryTriggerResponse`` shape was removed
// alongside the old ``CollectionSummary`` ORM (Wave 10 §K.13 hard-cut).
export type CollectionRegenTriggerResponse =
  components['schemas']['CollectionRegenTriggerResponse'];

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
