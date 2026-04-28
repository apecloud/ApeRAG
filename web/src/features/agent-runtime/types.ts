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
  /** User-visible short summary extracted by the backend (query/url/title). */
  summary?: string;
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
      summary?: string | null;
    }
  | {
      type: 'tool-output-available';
      toolCallId: string;
      output: unknown;
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
// Shape aligns 1:1 with AI SDK v5 `UIMessagePart` discriminators so the
// renderer can use the SDK's `isTextUIPart` / `isToolUIPart` /
// `isDataUIPart` guards directly. The compile-time `_AgentMessagePartIsSDKCompatible`
// assertion at the bottom of this file enforces this contract — if a
// shape drifts, type-check fails.

export type AgentTextPart = {
  type: 'text';
  /** Wire `text-block id` (also re-exposed for #77 dedup-by-id rendering). */
  id: string;
  text: string;
  state: 'streaming' | 'done';
};

/**
 * Tool part. Type discriminator follows the AI SDK v5 dynamic-tool
 * pattern: `tool-${SafeToolName}` (per D8 §2.4 / D9 §A1+A6).
 * `isToolUIPart` from `ai` accepts any `tool-${string}` literal so
 * this shape is exchangeable with the SDK's `ToolUIPart`.
 */
export type AgentToolPart = {
  type: `tool-${string}`;
  toolCallId: string;
  toolName: string;
  metadata?: Record<string, unknown>;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  /** User-visible short summary extracted by the backend (query/url/title). */
  summary?: string;
  state:
    | 'input-streaming'
    | 'input-available'
    | 'output-available'
    | 'output-error';
};

export type AgentSourceUrlPart = {
  type: 'source-url';
  sourceId: string;
  url: string;
  title?: string;
};

export type AgentSourceDocumentPart = {
  type: 'source-document';
  sourceId: string;
  mediaType: string;
  title: string;
};

/** Anthropic-shape citation, wrapped per AI SDK v5 `DataUIPart`. */
export type AgentCitationPart = {
  type: 'data-citation';
  /** Synthetic stable fingerprint (`cited_text + canonical(location)`). */
  id: string;
  data: CitationData;
};

/** Tool-consent prompt, wrapped per AI SDK v5 `DataUIPart`. */
export type AgentToolConsentPart = {
  type: 'data-tool-consent';
  /** `data.toolCallId` — re-exposed at the part envelope so SDK
   *  consumers can dedup without inspecting the inner payload. */
  id: string;
  data: ToolConsentData;
};

/** Elicitation prompt, wrapped per AI SDK v5 `DataUIPart`. */
export type AgentElicitationPart = {
  type: 'data-elicitation';
  /** `data.elicitationId` — see comment on `AgentToolConsentPart.id`. */
  id: string;
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

// --- Compile-time SDK compatibility assertion (Weston msg=63a796f3 B2) ---
//
// Each part type below must be structurally assignable to the
// corresponding shape in `@ai-sdk/react` (sourced from the `ai`
// package). If any shape drifts, this file fails to type-check and
// the renderer's `isTextUIPart` / `isToolUIPart` / `isDataUIPart`
// guards would silently misclassify our parts at runtime.

import type {
  DataUIPart as _SDKDataUIPart,
  SourceDocumentUIPart as _SDKSourceDocumentUIPart,
  SourceUrlUIPart as _SDKSourceUrlUIPart,
  TextUIPart as _SDKTextUIPart,
} from 'ai';

// The data discriminators we actually emit. Used to parameterize
// the SDK's `DataUIPart<DATA_TYPES>` for the assertion below.
export type ApeRAGUIDataTypes = {
  citation: CitationData;
  'tool-consent': ToolConsentData;
  elicitation: ElicitationData;
};

type _AssertText = AgentTextPart extends _SDKTextUIPart ? true : never;
type _AssertSourceUrl = AgentSourceUrlPart extends _SDKSourceUrlUIPart
  ? true
  : never;
type _AssertSourceDocument =
  AgentSourceDocumentPart extends _SDKSourceDocumentUIPart ? true : never;
type _AssertCitation =
  AgentCitationPart extends _SDKDataUIPart<ApeRAGUIDataTypes> ? true : never;
type _AssertToolConsent =
  AgentToolConsentPart extends _SDKDataUIPart<ApeRAGUIDataTypes> ? true : never;
type _AssertElicitation =
  AgentElicitationPart extends _SDKDataUIPart<ApeRAGUIDataTypes> ? true : never;
// Tool parts are intentionally not asserted against `ToolUIPart<TOOLS>`
// (which requires a static `TOOLS` schema) — the SDK's `isToolUIPart`
// guard branches on `type.startsWith('tool-')`, which our
// `tool-${string}` template-literal discriminator satisfies.

// One read of every assertion alias so the compiler keeps them around.
// Each member here would be `never` if the corresponding `extends`
// failed, which would propagate a type error to anyone importing this
// constant — making the compatibility check load-bearing.
export const _AgentMessagePartIsSDKCompatible: [
  _AssertText,
  _AssertSourceUrl,
  _AssertSourceDocument,
  _AssertCitation,
  _AssertToolConsent,
  _AssertElicitation,
] = [true, true, true, true, true, true];

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
