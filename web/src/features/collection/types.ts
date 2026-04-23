import type { components } from '@/api-v2/schema';

export type Collection = components['schemas']['Collection'];
export type CollectionCreate = components['schemas']['CollectionCreate'];
export type CollectionUpdate = components['schemas']['CollectionUpdate'];
export type CollectionView = components['schemas']['CollectionView'];
export type CollectionViewList = components['schemas']['CollectionViewList'];

export type SharingStatusResponse =
  components['schemas']['SharingStatusResponse'];
export type CollectionSummaryTriggerResponse =
  components['schemas']['CollectionSummaryTriggerResponse'];

export type MineruTokenTestRequest =
  components['schemas']['MineruTokenTestRequest'];
export type MineruTokenTestResponse =
  components['schemas']['MineruTokenTestResponse'];
