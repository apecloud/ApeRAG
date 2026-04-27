import { cookies } from 'next/headers';

import { getLocale } from '@/services/cookies';

import type { AuditLogList, ListAuditLogsParams } from './types';

// Raw fetch: audit logs is hidden from public OpenAPI per Phase 4
// governance (see `aperag/openapi_spec.py::HIDDEN_FROM_PUBLIC_PATH_PREFIXES`).
// Phase 4 will promote to typed contract. Server-side variant mirrors the
// cookie/lang injection done by `lib/api/typed/server::createServerApiClient`
// so the request is authenticated and localized like every other typed
// server fetch in the app.

const API_SERVER_ENDPOINT =
  process.env.API_SERVER_ENDPOINT || 'http://localhost:8000';

function mergeHeaders(
  initHeaders: HeadersInit | undefined,
  injected: Record<string, string>,
) {
  const headers = new Headers(initHeaders);
  for (const [key, value] of Object.entries(injected)) {
    if (value) {
      headers.set(key, value);
    }
  }
  return headers;
}

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
  if (params?.startDate) qs.set('start_date', params.startDate);
  if (params?.endDate) qs.set('end_date', params.endDate);
  if (params?.sortBy) qs.set('sort_by', params.sortBy);
  if (params?.sortOrder) qs.set('sort_order', params.sortOrder);
  const query = qs.toString();
  return `${API_SERVER_ENDPOINT}/api/v1/audit-logs${query ? `?${query}` : ''}`;
}

export async function listAuditLogs(
  params?: ListAuditLogsParams,
): Promise<AuditLogList> {
  const lang = await getLocale();
  const cookieHeader = (await cookies())
    .getAll()
    .map((item) => `${item.name}=${item.value}`)
    .join('; ');

  const response = await fetch(buildAuditLogsUrl(params), {
    cache: 'no-store',
    headers: mergeHeaders(undefined, {
      Cookie: cookieHeader,
      Lang: lang,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `listAuditLogs (server): HTTP ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as AuditLogList;
}
