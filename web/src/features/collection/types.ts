import type { components } from '@/api-v2/schema';

export type CollectionCreate = components['schemas']['CollectionCreate'];
export type CollectionUpdate = components['schemas']['CollectionUpdate'];
export type CollectionView = components['schemas']['CollectionView'];
export type CollectionViewList = components['schemas']['CollectionViewList'];

// task #61 P1-D3 (PR for #87): typed mirrors of the deployment vector
// backend identity + capability matrix that the BE projects onto every
// collection detail read. The FE only consumes them as read-only display
// — there is no input shape for these (the BE does not accept them on
// create/update). Per architect msg=0044261f + dongdong msg=c2593fdd
// the schemas live on ``Collection`` (output projection) and never on
// ``CollectionConfig`` (which would leak them onto the OpenAPI input).
export type VectorBackendInfo = NonNullable<
  components['schemas']['VectorBackendInfo']
>;
export type VectorBackendCapabilities = NonNullable<
  components['schemas']['VectorBackendCapabilities']
>;
export type VectorBackendType = VectorBackendInfo['type'];

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
