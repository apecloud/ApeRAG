import type { components } from '@/api-v2/schema';

// Settings (parser / MinerU / MarkItDown admin config).
export type Settings = components['schemas']['Settings'];

// Admin-side system default quota configuration.
export type SystemDefaultQuotas = components['schemas']['SystemDefaultQuotas'];

// Admin-side per-user quota update input.
export type QuotaUpdateRequest = components['schemas']['QuotaUpdateRequest'];

// Admin-side per-user quota read shape (reused from workspace quota domain
// because the `/api/v1/quotas` endpoint returns the same `UserQuotaInfo`
// shape regardless of caller). Re-exported here so `features/admin/*` is the
// single canonical surface for admin control-plane types.
export type UserQuotaInfo = components['schemas']['UserQuotaInfo'];
export type UserQuotaList = components['schemas']['UserQuotaList'];
