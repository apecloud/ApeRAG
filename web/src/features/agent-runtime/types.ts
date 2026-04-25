'use client';

// Wire-shape and at-rest types for the agent runtime stream client (D8.4a).
// Mirrors aperag/domains/agent_runtime/wire/parts.py and uimessage.py byte-for-byte
// so wire frames round-trip without a translator on the FE side.

// --- Citation location (Anthropic-shape, D8 §2.5) -------------------------

export type CitationLocation =
  | {
      type: 'char_location';
      start_char_index?: number;
      end_char_index?: number;
      doc_index?: number;
      doc_title?: string | null;
      url?: string | null;
      title?: string | null;
    }
  | {
      type: 'page_location';
      page_number?: number;
      doc_index?: number;
      doc_title?: string | null;
      url?: string | null;
      title?: string | null;
    }
  | {
      type: 'content_block_location';
      content_block_index?: number;
      doc_index?: number;
      doc_title?: string | null;
      url?: string | null;
      title?: string | null;
    }
  | {
      type: 'url_citation';
      url?: string | null;
      title?: string | null;
    };

export type CitationData = {
  cited_text: string;
  location: CitationLocation;
};

// --- Transient activity (data-activity, D8.1) -----------------------------

export type ActivityIntent =
  | 'thinking'
  | 'searching_knowledge'
  | 'reading_source'
  | 'comparing_results'
  | 'writing_answer'
  | 'waiting'
  | 'completed'
  | 'error';

export type ActivityContext = {
  source_name?: string | null;
  keyword?: string | null;
  count?: number | null;
  target_type?: 'knowledge_base' | 'document' | 'web' | null;
  scope_label?: string | null;
};

export type UserActivityEnvelope = {
  intent: ActivityIntent;
  title_key?: string;
  subtitle_key?: string;
  detail_key?: string | null;
  context?: ActivityContext | null;
};

export type ActivityData = {
  activity?: UserActivityEnvelope;
  intent?: string;
  label?: string | null;
};

// --- Tool consent + elicitation (D9 §3 + §5) ------------------------------

export type ToolConsentRisk =
  | 'writes_user_data'
  | 'calls_external_api'
  | 'modifies_system'
  | 'admin_only';

export type ToolConsentData = {
  toolCallId: string;
  toolName: string;
  metadata?: Record<string, unknown>;
  argsPreview: string;
  argsHash: string;
  risk: ToolConsentRisk;
  requestedAt: string;
  state: 'pending' | 'approved' | 'denied' | 'expired';
};

export type ElicitationData = {
  elicitationId: string;
  serverName: string;
  prompt: string;
  schema: Record<string, unknown>;
  response?: Record<string, unknown> | null;
  state: 'pending' | 'answered' | 'cancelled';
};

// --- Wire StreamPart discriminated union ----------------------------------
// Mirrors wire/parts.py StreamPart 1:1. Always camelCase outer keys; inner
// `data` payloads carry whatever shape the BE Pydantic model declared
// (snake_case for citation/activity, camelCase for consent/elicitation).

export type StreamPart =
  | { type: 'start'; messageId?: string | null }
  | { type: 'start-step' }
  | { type: 'finish-step' }
  | { type: 'finish' }
  | { type: 'abort' }
  | { type: 'error'; errorText: string }
  | { type: 'text-start'; id: string }
  | { type: 'text-delta'; id: string; delta: string }
  | { type: 'text-end'; id: string }
  | {
      type: 'tool-input-start';
      toolCallId: string;
      toolName: string;
      metadata?: Record<string, unknown>;
    }
  | { type: 'tool-input-delta'; toolCallId: string; inputTextDelta: string }
  | {
      type: 'tool-input-available';
      toolCallId: string;
      toolName: string;
      input: unknown;
    }
  | {
      type: 'tool-output-available';
      toolCallId: string;
      output: unknown;
      // BE `parts.py` (#73) currently sets `errorText` here on failure;
      // AI SDK v5 strict spec migrates failure to `tool-output-error`
      // (see task #89 D8.0c+ hygiene fix-forward). The reducer accepts
      // both shapes so the FE rolls forward without coupling to the
      // BE's split timing.
      errorText?: string | null;
    }
  | { type: 'tool-output-error'; toolCallId: string; errorText: string }
  | { type: 'source-url'; sourceId: string; url: string; title?: string | null }
  | {
      type: 'source-document';
      sourceId: string;
      mediaType: string;
      title: string;
    }
  | { type: 'data-citation'; data: CitationData }
  | { type: 'data-activity'; data: ActivityData; transient: true }
  | { type: 'data-tool-consent'; data: ToolConsentData }
  | { type: 'data-elicitation'; data: ElicitationData };

// --- At-rest agent message parts (consumed by #77 renderer + #78 UI) ------
// These are the dedup'd, lifecycle-collapsed parts the hook publishes.
// Shape aligns with AI SDK v5 UIMessagePart so the renderer can use SDK
// type guards (isTextUIPart / isToolUIPart / etc.) where convenient.

export type AgentTextPart = {
  kind: 'text';
  id: string;
  text: string;
  state: 'streaming' | 'done';
};

export type AgentToolPart = {
  kind: 'tool';
  toolCallId: string;
  toolName: string;
  metadata?: Record<string, unknown>;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  state:
    | 'input-streaming'
    | 'input-available'
    | 'output-available'
    | 'output-error';
};

export type AgentSourceUrlPart = {
  kind: 'source-url';
  sourceId: string;
  url: string;
  title?: string | null;
};

export type AgentSourceDocumentPart = {
  kind: 'source-document';
  sourceId: string;
  mediaType: string;
  title: string;
};

export type AgentCitationPart = {
  kind: 'citation';
  key: string;
  data: CitationData;
};

export type AgentToolConsentPart = {
  kind: 'tool-consent';
  toolCallId: string;
  data: ToolConsentData;
};

export type AgentElicitationPart = {
  kind: 'elicitation';
  elicitationId: string;
  data: ElicitationData;
};

export type AgentMessagePart =
  | AgentTextPart
  | AgentToolPart
  | AgentSourceUrlPart
  | AgentSourceDocumentPart
  | AgentCitationPart
  | AgentToolConsentPart
  | AgentElicitationPart;

// --- Stream status ---------------------------------------------------------

export type AgentStreamStatus =
  | 'idle'
  | 'connecting'
  | 'streaming'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'aborted';

// AI SDK v5 protocol marker (BE response header value).
export const AI_SDK_V5_HEADER = 'x-vercel-ai-ui-message-stream';
export const AI_SDK_V5_HEADER_VALUE = 'v1';
