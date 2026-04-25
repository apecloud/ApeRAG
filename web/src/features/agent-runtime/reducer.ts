'use client';

// Stream reducer: collapses lifecycle wire parts into the consolidated,
// dedup'd `AgentMessagePart[]` shape that downstream renderers (#77)
// and interactive UI (#78) consume. Output shapes are SDK-compatible
// (see `types.ts` `_AgentMessagePartIsSDKCompatible` assertion).
//
// Dedup contract (architect lock msg=f35c5a3d / Lock C):
//   The BE replays an entire envelope on disconnect — every part
//   produced by the same envelope is re-emitted on resume. The client
//   is responsible for deduping by **stable part identifier**:
//     * text parts: `id` (text-block id)
//     * tool parts: `toolCallId`
//     * source-url / source-document: `sourceId`
//     * data-tool-consent: `data.toolCallId` (re-exposed as part `id`)
//     * data-elicitation: `data.elicitationId` (re-exposed as part `id`)
//     * data-citation: `cited_text + canonical(location)` fingerprint
//       (re-exposed as part `id`) — there is no native id on the wire,
//       and the BE always re-emits the same payload on replay so a
//       content fingerprint is stable.
//     * data-activity: TRANSIENT — replace-last only, never persisted
//       into the parts array.
//
// State semantics:
//   * Status follows the lifecycle envelope: `connecting` → `streaming`
//     (after first non-lifecycle part) → `completed` / `failed` /
//     `aborted` / `cancelled`.
//   * `errorText` is set by the `error` part; the stream client closes
//     the connection right after dispatching it.
//   * `lastSequence` is updated only when a frame carries an SSE
//     `id:` field (the LAST part of a fan-out group), so reconnect
//     resumes at the next envelope, never mid-fanout.

import type {
  ActivityData,
  AgentMessagePart,
  AgentStreamStatus,
  AgentTextPart,
  AgentToolPart,
  CitationData,
  StreamPart,
} from './types';

export type ReducerState = {
  parts: AgentMessagePart[];
  transientActivity: ActivityData | null;
  status: AgentStreamStatus;
  errorText: string | null;
  lastSequence: number;
};

export const initialReducerState = (): ReducerState => ({
  parts: [],
  transientActivity: null,
  status: 'idle',
  errorText: null,
  lastSequence: 0,
});

export function applyPart(
  state: ReducerState,
  part: StreamPart,
  eventId: number | null,
): ReducerState {
  // Lifecycle frames first — they drive status, never appear in `parts`.
  switch (part.type) {
    case 'start':
    case 'start-step':
      return advanceStatus(state, 'streaming', eventId);
    case 'finish-step':
      // finish-step closes the inner step, but the turn may still emit
      // more parts; status stays streaming.
      return advanceStatus(state, state.status, eventId);
    case 'finish':
      return advanceStatus(state, 'completed', eventId);
    case 'abort':
      return advanceStatus(state, 'aborted', eventId);
    case 'error':
      return {
        ...state,
        status: 'failed',
        errorText: part.errorText,
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
  }

  // Content frames — collapse + dedup.
  switch (part.type) {
    case 'text-start':
      return {
        ...state,
        status: state.status === 'idle' ? 'streaming' : state.status,
        parts: upsertText(state.parts, part.id, '', 'streaming'),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'text-delta':
      return {
        ...state,
        status: state.status === 'idle' ? 'streaming' : state.status,
        parts: appendTextDelta(state.parts, part.id, part.delta),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'text-end':
      return {
        ...state,
        parts: closeText(state.parts, part.id),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'tool-input-start':
      return {
        ...state,
        parts: upsertTool(state.parts, part.toolCallId, {
          toolName: part.toolName,
          metadata: part.metadata,
          state: 'input-streaming',
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'tool-input-delta':
      // Args streaming — we don't surface partial JSON to the UI today
      // (BE batches on `tool-input-available`), but we still keep the
      // part registered so a later renderer can show "input
      // streaming…" affordances.
      return {
        ...state,
        parts: upsertTool(state.parts, part.toolCallId, {
          state: 'input-streaming',
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'tool-input-available':
      return {
        ...state,
        parts: upsertTool(state.parts, part.toolCallId, {
          toolName: part.toolName,
          input: part.input,
          state: 'input-available',
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'tool-output-available': {
      // BE today (#73) emits failures here too with `errorText` set;
      // task #89 splits failures onto `tool-output-error`. We accept
      // both shapes so the FE rolls forward without coupling to BE
      // timing.
      const failed =
        typeof part.errorText === 'string' && part.errorText.length > 0;
      return {
        ...state,
        parts: upsertTool(state.parts, part.toolCallId, {
          output: part.output,
          errorText: failed ? part.errorText! : undefined,
          state: failed ? 'output-error' : 'output-available',
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    }
    case 'tool-output-error':
      // Strict AI SDK v5 failure shape (task #89 fix-forward target).
      return {
        ...state,
        parts: upsertTool(state.parts, part.toolCallId, {
          errorText: part.errorText,
          state: 'output-error',
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'source-url':
      return {
        ...state,
        parts: upsertSourceUrl(state.parts, part.sourceId, {
          url: part.url,
          title: nullToUndefined(part.title),
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'source-document':
      return {
        ...state,
        parts: upsertSourceDocument(state.parts, part.sourceId, {
          mediaType: part.mediaType,
          title: part.title,
        }),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'data-citation': {
      const id = citationFingerprint(part.data);
      return {
        ...state,
        parts: upsertCitation(state.parts, id, part.data),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    }

    case 'data-tool-consent':
      return {
        ...state,
        parts: upsertToolConsent(state.parts, part.data),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'data-elicitation':
      return {
        ...state,
        parts: upsertElicitation(state.parts, part.data),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'data-activity':
      // Transient — replace-last only, never persisted to `parts`. The
      // architect lock (msg=f35c5a3d) makes this explicit: the
      // wire-side `transient: true` flag means the consumer must NOT
      // include the part in any persistent message history.
      return {
        ...state,
        transientActivity: part.data,
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
  }

  return state;
}

function advanceStatus(
  state: ReducerState,
  next: AgentStreamStatus,
  eventId: number | null,
): ReducerState {
  return {
    ...state,
    status: next,
    lastSequence: maxSeq(state.lastSequence, eventId),
  };
}

function maxSeq(current: number, candidate: number | null): number {
  if (candidate == null) return current;
  return candidate > current ? candidate : current;
}

function nullToUndefined<T>(value: T | null | undefined): T | undefined {
  return value == null ? undefined : value;
}

// -- text helpers ---------------------------------------------------------

function isTextPart(part: AgentMessagePart): part is AgentTextPart {
  return part.type === 'text';
}

function isToolPart(part: AgentMessagePart): part is AgentToolPart {
  return part.type.startsWith('tool-');
}

function upsertText(
  parts: AgentMessagePart[],
  id: string,
  initial: string,
  state: AgentTextPart['state'],
): AgentMessagePart[] {
  const index = parts.findIndex((p) => isTextPart(p) && p.id === id);
  if (index >= 0) {
    const existing = parts[index] as AgentTextPart;
    return replaceAt(parts, index, { ...existing, state });
  }
  return [...parts, { type: 'text', id, text: initial, state }];
}

function appendTextDelta(
  parts: AgentMessagePart[],
  id: string,
  delta: string,
): AgentMessagePart[] {
  const index = parts.findIndex((p) => isTextPart(p) && p.id === id);
  if (index < 0) {
    // text-delta arrived without text-start (BE emits text-start once
    // per turn but defends against missing it). Synthesize a streaming
    // block so the delta isn't dropped.
    return [...parts, { type: 'text', id, text: delta, state: 'streaming' }];
  }
  const existing = parts[index] as AgentTextPart;
  return replaceAt(parts, index, {
    ...existing,
    text: existing.text + delta,
    state: 'streaming',
  });
}

function closeText(parts: AgentMessagePart[], id: string): AgentMessagePart[] {
  const index = parts.findIndex((p) => isTextPart(p) && p.id === id);
  if (index < 0) return parts;
  const existing = parts[index] as AgentTextPart;
  return replaceAt(parts, index, { ...existing, state: 'done' });
}

// -- tool helpers ---------------------------------------------------------

type ToolPatch = Partial<
  Pick<
    AgentToolPart,
    'toolName' | 'metadata' | 'input' | 'output' | 'errorText' | 'state'
  >
>;

function upsertTool(
  parts: AgentMessagePart[],
  toolCallId: string,
  patch: ToolPatch,
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => isToolPart(p) && p.toolCallId === toolCallId,
  );
  if (index < 0) {
    const toolName = patch.toolName ?? '';
    const created: AgentToolPart = {
      type: `tool-${toolName}`,
      toolCallId,
      toolName,
      metadata: patch.metadata,
      input: patch.input,
      output: patch.output,
      errorText: patch.errorText,
      state: patch.state ?? 'input-streaming',
    };
    return [...parts, created];
  }
  const existing = parts[index] as AgentToolPart;
  // Preserve toolName once it's known (input-start / input-available
  // both carry it; output-* parts do not). The `type` discriminator is
  // recomputed when toolName changes so SDK guards still see a stable
  // `tool-${name}` shape after the toolName resolves.
  const nextToolName = patch.toolName ?? existing.toolName;
  const merged: AgentToolPart = {
    ...existing,
    ...patch,
    toolName: nextToolName,
    type: `tool-${nextToolName}`,
    state: patch.state ?? existing.state,
  };
  return replaceAt(parts, index, merged);
}

// -- source helpers -------------------------------------------------------

function upsertSourceUrl(
  parts: AgentMessagePart[],
  sourceId: string,
  fields: { url: string; title?: string },
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => p.type === 'source-url' && p.sourceId === sourceId,
  );
  if (index < 0) {
    return [...parts, { type: 'source-url', sourceId, ...fields }];
  }
  // Already seen — replay. Same payload byte-stable per
  // envelope-atomic replay; keep existing.
  return parts;
}

function upsertSourceDocument(
  parts: AgentMessagePart[],
  sourceId: string,
  fields: { mediaType: string; title: string },
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => p.type === 'source-document' && p.sourceId === sourceId,
  );
  if (index < 0) {
    return [...parts, { type: 'source-document', sourceId, ...fields }];
  }
  return parts;
}

// -- data part helpers ----------------------------------------------------

function upsertCitation(
  parts: AgentMessagePart[],
  id: string,
  data: CitationData,
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => p.type === 'data-citation' && p.id === id,
  );
  if (index < 0) {
    return [...parts, { type: 'data-citation', id, data }];
  }
  // Already seen — replay; data is byte-identical per envelope-atomic
  // replay so we keep the existing entry.
  return parts;
}

function upsertToolConsent(
  parts: AgentMessagePart[],
  data: import('./types').ToolConsentData,
): AgentMessagePart[] {
  const id = data.toolCallId;
  const index = parts.findIndex(
    (p) => p.type === 'data-tool-consent' && p.id === id,
  );
  if (index < 0) {
    return [...parts, { type: 'data-tool-consent', id, data }];
  }
  // Consent decision can transition (pending → approved/denied/expired);
  // the BE re-emits the part with the new state, so we replace data.
  return replaceAt(parts, index, { type: 'data-tool-consent', id, data });
}

function upsertElicitation(
  parts: AgentMessagePart[],
  data: import('./types').ElicitationData,
): AgentMessagePart[] {
  const id = data.elicitationId;
  const index = parts.findIndex(
    (p) => p.type === 'data-elicitation' && p.id === id,
  );
  if (index < 0) {
    return [...parts, { type: 'data-elicitation', id, data }];
  }
  // Same reasoning as consent: state transitions (pending → answered /
  // cancelled) are re-emitted; replace data on hit.
  return replaceAt(parts, index, { type: 'data-elicitation', id, data });
}

function citationFingerprint(data: CitationData): string {
  // Stable fingerprint: cited_text + JSON.stringify(location). The BE
  // always replays the identical payload on reconnect, so this collides
  // exactly when (and only when) we've already seen the citation.
  // Symphony msg=2f9225f5: JSON.stringify key order is engine-stable
  // for primitive-keyed objects (matches BE Pydantic shape) within a
  // process; if cross-session ghost duplicates surface later, swap to
  // a canonical sorted-keys hash.
  try {
    return `${data.cited_text}${JSON.stringify(data.location)}`;
  } catch {
    return data.cited_text;
  }
}

function replaceAt<T>(items: T[], index: number, value: T): T[] {
  const next = items.slice();
  next[index] = value;
  return next;
}
