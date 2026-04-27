import type { components } from '@/api-v2/schema';

// Settings (parser / MinerU / MarkItDown admin config).
export type Settings = components['schemas']['Settings'];

// Admin-side system default quota configuration (request input shape).
export type SystemDefaultQuotas = components['schemas']['SystemDefaultQuotas'];
// Wrapped read/update response shapes exposed by the backend.
export type SystemDefaultQuotasResponse =
  components['schemas']['SystemDefaultQuotasResponse'];
export type SystemDefaultQuotasUpdateRequest =
  components['schemas']['SystemDefaultQuotasUpdateRequest'];
export type SystemDefaultQuotasUpdateResponse =
  components['schemas']['SystemDefaultQuotasUpdateResponse'];

// Admin-side per-user quota update input + response.
export type QuotaUpdateRequest = components['schemas']['QuotaUpdateRequest'];
export type QuotaUpdateResponse = components['schemas']['QuotaUpdateResponse'];

// Admin-side per-user quota read shape (reused from workspace quota domain
// because the `/api/v1/quotas` endpoint returns the same `UserQuotaInfo`
// shape regardless of caller). Re-exported here so `features/admin/*` is the
// single canonical surface for admin control-plane types.
export type UserQuotaInfo = components['schemas']['UserQuotaInfo'];
export type UserQuotaList = components['schemas']['UserQuotaList'];

// MinerU token test response (shared with collection feature).
export type MineruTokenTestResponse =
  components['schemas']['MineruTokenTestResponse'];
