import type { components } from '@/api-v2/schema';

export type SharedCollection = components['schemas']['SharedCollection'];

export type SharedCollectionList =
  components['schemas']['SharedCollectionList'];

// Marketplace endpoints return document/preview schemas shared with the
// document feature but consumed here only under marketplace semantics.
// Alias them locally with a marketplace prefix so the marketplace callers
// never reach into `features/document/types` and never re-import `@/api`
// legacy types. The underlying schema components are the source of truth;
// the alias is purely a naming boundary.
export type MarketplaceDocument = components['schemas']['Document'];

export type MarketplaceDocumentList = components['schemas']['DocumentList'];

export type MarketplaceDocumentPreview =
  components['schemas']['DocumentPreview'];
