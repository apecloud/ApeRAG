'use client';

import type { AuditLog, AuditLogList, ListAuditLogsParams } from './types';

// Raw fetch: audit logs is hidden from public OpenAPI per Phase 4
// governance (see `aperag/openapi_spec.py::HIDDEN_FROM_PUBLIC_PATH_PREFIXES`).
// Phase 4 will promote to typed contract. The feature adapter boundary
// keeps the raw fetch confined to this file + `./server-api.ts` so the
// rest of the FE does not hand-build audit URLs (enforced by G17 static
// gate `test_audit_raw_fetch_confined_to_features_audit`).

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || '';

function buildAuditLogsUrl(params?: ListAuditLogsParams): string {
  const qs = new URLSearchParams();
  if (params?.page != null) qs.set('page', String(params.page));
  if (params?.pageSize != null) qs.set('page_size', String(params.pageSize));
  if (params?.userId) qs.set('user_id', params.userId);
  if (params?.resourceType) qs.set('resource_type', params.resourceType);
  if (params?.apiName) qs.set('api_name', params.apiName);
  if (params?.httpMethod) qs.set('http_method', params.httpMethod);
  if (params?.statusCode != null)
    qs.set('status_code', String(params.statusCode));
  if (params?.startTime != null) qs.set('start_time', String(params.startTime));
  if (params?.endTime != null) qs.set('end_time', String(params.endTime));
  const query = qs.toString();
  return `${BASE_PATH}/api/v1/audit-logs${query ? `?${query}` : ''}`;
}

export async function listAuditLogs(
  params?: ListAuditLogsParams,
): Promise<AuditLogList> {
  const response = await fetch(buildAuditLogsUrl(params), {
    credentials: 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`listAuditLogs: HTTP ${response.status}`);
  }
  return (await response.json()) as AuditLogList;
}

export async function getAuditLog(id: string): Promise<AuditLog> {
  const response = await fetch(
    `${BASE_PATH}/api/v1/audit-logs/${encodeURIComponent(id)}`,
    { credentials: 'same-origin' },
  );
  if (!response.ok) {
    throw new Error(`getAuditLog: HTTP ${response.status}`);
  }
  return (await response.json()) as AuditLog;
}
