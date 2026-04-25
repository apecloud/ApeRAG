'use client';

// HTTP client for the agent runtime (D8.4a). Wraps the FastAPI surface
// declared in `aperag/domains/agent_runtime/api/routes.py`. The SSE
// stream itself is handled separately by the stream client; this file
// only covers the JSON request/response endpoints the chat UI calls
// alongside the stream.

import type { AgentMessagePart, ToolConsentData, ElicitationData } from './types';

const DEFAULT_BASE = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/agent`;

export type AgentTurnEnvelope = {
  schema_version: string;
  turn_id: string;
  chat_id: string;
  user_id: string;
  bot_id: string;
  request_id: string;
  client_idempotency_key: string;
  status: string;
  input_text: string;
  model_profile: Record<string, unknown>;
  error_code?: string | null;
  error_message?: string | null;
  answer_artifact_id?: string | null;
  reference_bundle_artifact_id?: string | null;
  timeline_cursor: number;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentTimelineEventEnvelope = {
  schema_version: string;
  event_id: string;
  turn_id: string;
  sequence: number;
  timestamp: string;
  type: string;
  technical_type?: string | null;
  label?: string | null;
  status?: string | null;
  actor: 'agent' | 'tool' | 'system';
  data: Record<string, unknown>;
};

export type AgentArtifactEnvelope = {
  schema_version: string;
  artifact_id: string;
  turn_id: string;
  artifact_type: string;
  summary?: string | null;
  payload: Record<string, unknown>;
  storage_ref?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

// Phase 8 D8.4d (#90) + D8.5-BE (#92): the snapshot endpoint and
// `getBotChat()` history payload both return the canonical UIMessage
// at-rest envelope (matches
// `aperag.domains.agent_runtime.uimessage.AgentTurnSnapshot`).
// Per D8 §2 wire / at-rest are byte-equal, so the FE consumes the same
// `AgentMessagePart` discriminated union from both the live SSE stream
// and this reload payload — no client-side converter is required.
//
// `runtime_kind` (D8.5-BE) is a forward-compat discriminator for
// future direct/RAG chat paths; today's production code only emits
// `agent_runtime`. The FE does NOT branch on it (per PM lock
// msg=38e116e5 — kept BE-internal).
//
// `input_text` is the user-side bubble content; the FE renders it as
// the `MessagePartsUser` paired with the AI card so historical and
// live turns share one render path.
export type AgentRuntimeKind = 'agent_runtime' | 'direct_chat' | 'rag_chat';

export type AgentTurnSnapshotEnvelope = {
  schema_version: string;
  turn_id: string;
  chat_id: string;
  runtime_kind: AgentRuntimeKind;
  input_text?: string | null;
  role: 'assistant';
  status: string;
  parts: AgentMessagePart[];
  error_text?: string | null;
  timeline_cursor: number;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentTurnCreateInput = {
  query: string;
  client_idempotency_key?: string;
  // Forward-compat: any extra fields the chat input feeds into
  // CreateTurnRequest (model_profile / locale / etc.) flow through
  // unchanged.
  [key: string]: unknown;
};

export type AgentTurnCreateResponse = {
  turn: AgentTurnEnvelope;
  stream_url: string;
};

type RequestOptions = Omit<RequestInit, 'body'> & {
  body?: unknown;
};

async function request<T>(url: string, init?: RequestOptions): Promise<T> {
  const headers = new Headers(init?.headers);
  let serializedBody: BodyInit | null | undefined;
  const rawBody = init?.body;
  if (rawBody == null) {
    serializedBody = undefined;
  } else if (
    rawBody instanceof FormData ||
    rawBody instanceof Blob ||
    rawBody instanceof URLSearchParams ||
    rawBody instanceof ArrayBuffer ||
    typeof rawBody === 'string'
  ) {
    serializedBody = rawBody as BodyInit;
  } else {
    headers.set('Content-Type', 'application/json');
    serializedBody = JSON.stringify(rawBody);
  }

  const { body: _ignored, ...rest } = init ?? {};
  void _ignored;
  const response = await fetch(url, {
    credentials: 'same-origin',
    cache: 'no-store',
    ...rest,
    headers,
    body: serializedBody,
  });

  if (!response.ok) {
    let detail: unknown;
    try {
      detail = await response.json();
    } catch {
      detail = undefined;
    }
    throw new Error(extractError(detail, response.status));
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

function extractError(detail: unknown, status: number): string {
  if (typeof detail === 'string' && detail) return detail;
  if (
    typeof detail === 'object' &&
    detail &&
    'detail' in detail &&
    typeof (detail as { detail?: unknown }).detail === 'string'
  ) {
    return (detail as { detail: string }).detail;
  }
  return `Request failed with status ${status}`;
}

export async function createAgentTurn(
  chatId: string,
  input: AgentTurnCreateInput,
): Promise<AgentTurnCreateResponse> {
  return request<AgentTurnCreateResponse>(
    `${DEFAULT_BASE}/chats/${chatId}/turns`,
    {
      method: 'POST',
      body: {
        client_idempotency_key:
          input.client_idempotency_key ??
          (typeof crypto !== 'undefined' && 'randomUUID' in crypto
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random()}`),
        ...input,
      },
    },
  );
}

export async function getAgentTurnSnapshot(
  chatId: string,
  turnId: string,
): Promise<AgentTurnSnapshotEnvelope> {
  return request<AgentTurnSnapshotEnvelope>(
    `${DEFAULT_BASE}/chats/${chatId}/turns/${turnId}`,
    { method: 'GET' },
  );
}

export async function cancelAgentTurn(
  chatId: string,
  turnId: string,
): Promise<{ turn_id: string; status: string }> {
  return request<{ turn_id: string; status: string }>(
    `${DEFAULT_BASE}/chats/${chatId}/turns/${turnId}/cancel`,
    { method: 'POST', body: {} },
  );
}

export async function getAgentArtifact(
  artifactId: string,
): Promise<AgentArtifactEnvelope> {
  return request<AgentArtifactEnvelope>(
    `${DEFAULT_BASE}/artifacts/${artifactId}`,
    { method: 'GET' },
  );
}

// D9 §3 — consent decision (placeholder client for #78 chenyexuan).
export async function decideToolConsent(
  chatId: string,
  turnId: string,
  toolCallId: string,
  decision: 'approved' | 'denied',
): Promise<{ consent_id: string; decision: string; state: ToolConsentData['state'] }> {
  return request(
    `${DEFAULT_BASE}/chats/${chatId}/turns/${turnId}/consent/${toolCallId}`,
    {
      method: 'POST',
      body: { decision },
    },
  );
}

// D9 §5 — elicitation submission (placeholder client for #78).
export async function submitElicitation(
  chatId: string,
  turnId: string,
  elicitationId: string,
  response: Record<string, unknown>,
): Promise<{ elicitation_id: string; state: ElicitationData['state'] }> {
  return request(
    `${DEFAULT_BASE}/chats/${chatId}/turns/${turnId}/elicit/${elicitationId}`,
    {
      method: 'POST',
      body: { response },
    },
  );
}
