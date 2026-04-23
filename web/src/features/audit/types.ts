// Audit logs are intentionally hidden from the public OpenAPI spec
// (`HIDDEN_FROM_PUBLIC_PATH_PREFIXES` in `aperag/openapi_spec.py`), so no
// `components['schemas']['AuditLog']` exists in `@/api-v2/schema`. These
// mirrors reflect the current backend shape and will be promoted to
// schema-derived aliases in Phase 4 governance (see
// `features/audit/client-api.ts` + `server-api.ts` for the matching raw
// fetch comment). Hand-writing the mirror keeps the FE domain contract
// explicit without committing the backend to a public API shape.
//
// Counting note: this file intentionally does NOT import from
// `@/api-v2/schema`. When Phase 4 unhides `/api/v1/audit-logs` and this
// mirror is promoted to schema-derived aliases, the
// `web_raw_schema_allowlist.txt` target will rise by 1 (current #13
// canonical is `raw_schema 13 → 16`; Phase 4 takes it to `16 → 17`).

export type AuditLogResourceType =
  | 'collection'
  | 'document'
  | 'bot'
  | 'chat'
  | 'message'
  | 'api_key'
  | 'llm'
  | 'llm_provider'
  | 'llm_provider_model'
  | 'model_service_provider'
  | 'user'
  | 'flow'
  | 'search'
  | 'index';

export type AuditLog = {
  id?: string;
  user_id?: string;
  username?: string;
  resource_type?: AuditLogResourceType;
  resource_id?: string;
  api_name?: string;
  http_method?: string;
  path?: string;
  status_code?: number;
  start_time?: number;
  end_time?: number;
  duration_ms?: number;
  request_data?: string;
  response_data?: string;
  error_message?: string;
  ip_address?: string;
  user_agent?: string;
  request_id?: string;
  created?: string;
};

export type AuditLogList = {
  items?: AuditLog[];
  total?: number;
  page?: number;
  page_size?: number;
  total_pages?: number;
};

export type ListAuditLogsParams = {
  page?: number;
  pageSize?: number;
  userId?: string;
  resourceType?: AuditLogResourceType;
  apiName?: string;
  httpMethod?: string;
  statusCode?: number;
  startTime?: number;
  endTime?: number;
  /** ISO date filters accepted by the backend alongside numeric start/end time. */
  startDate?: string;
  endDate?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
};

export type AuditLogListResponse = AuditLogList;
