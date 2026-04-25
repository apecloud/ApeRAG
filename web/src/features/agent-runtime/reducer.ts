'use client';

// Stream reducer: collapses lifecycle wire parts into the consolidated,
// dedup'd `AgentMessagePart[]` shape that downstream renderers (#77)
// and interactive UI (#78) consume.
//
// Dedup contract (architect lock msg=f35c5a3d / Lock C):
//   The BE replays an entire envelope on disconnect — every part
//   produced by the same envelope is re-emitted on resume. The client
//   is responsible for deduping by **stable part identifier**:
//     * text-* parts: `id`
//     * tool-* parts: `toolCallId`
//     * source-* parts: `sourceId`
//     * data-tool-consent: `data.toolCallId`
//     * data-elicitation: `data.elicitationId`
//     * data-citation: `cited_text + canonical(location)` fingerprint
//       — there is no native id; the BE always re-emits the same
//       payload on replay, so a content fingerprint is stable.
//     * data-activity: TRANSIENT — replace-last only, never persisted.
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
  AgentMessagePart,
  AgentStreamStatus,
  AgentTextPart,
  AgentToolPart,
  ActivityData,
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
      // Args streaming — we don't surface partial JSON to the UI
      // today (BE batches on `tool-input-available`), but we still
      // keep the part registered so a later renderer can show "input
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

    case 'source-url':
      return {
        ...state,
        parts: upsertById(state.parts, 'sourceId', part.sourceId, () => ({
          kind: 'source-url',
          sourceId: part.sourceId,
          url: part.url,
          title: part.title,
        })),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    case 'source-document':
      return {
        ...state,
        parts: upsertById(state.parts, 'sourceId', part.sourceId, () => ({
          kind: 'source-document',
          sourceId: part.sourceId,
          mediaType: part.mediaType,
          title: part.title,
        })),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'data-citation': {
      const key = citationKey(part.data);
      return {
        ...state,
        parts: upsertCitation(state.parts, key, part.data),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };
    }

    case 'data-tool-consent':
      return {
        ...state,
        parts: upsertById(
          state.parts,
          'toolCallId',
          part.data.toolCallId,
          () => ({
            kind: 'tool-consent',
            toolCallId: part.data.toolCallId,
            data: part.data,
          }),
          (existing) =>
            existing.kind === 'tool-consent'
              ? { ...existing, data: part.data }
              : existing,
        ),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'data-elicitation':
      return {
        ...state,
        parts: upsertById(
          state.parts,
          'elicitationId',
          part.data.elicitationId,
          () => ({
            kind: 'elicitation',
            elicitationId: part.data.elicitationId,
            data: part.data,
          }),
          (existing) =>
            existing.kind === 'elicitation'
              ? { ...existing, data: part.data }
              : existing,
        ),
        lastSequence: maxSeq(state.lastSequence, eventId),
      };

    case 'data-activity':
      // Transient — replace-last only, never persisted to `parts`.
      // The architect lock (msg=f35c5a3d) makes this explicit: the
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

// -- text helpers ---------------------------------------------------------

function upsertText(
  parts: AgentMessagePart[],
  id: string,
  initial: string,
  state: AgentTextPart['state'],
): AgentMessagePart[] {
  const index = parts.findIndex((p) => p.kind === 'text' && p.id === id);
  if (index >= 0) {
    const existing = parts[index] as AgentTextPart;
    return replaceAt(parts, index, { ...existing, state });
  }
  return [...parts, { kind: 'text', id, text: initial, state }];
}

function appendTextDelta(
  parts: AgentMessagePart[],
  id: string,
  delta: string,
): AgentMessagePart[] {
  const index = parts.findIndex((p) => p.kind === 'text' && p.id === id);
  if (index < 0) {
    // text-delta arrived without text-start (BE emits text-start once
    // per turn but defends against missing it). Synthesize a streaming
    // block so the delta isn't dropped.
    return [...parts, { kind: 'text', id, text: delta, state: 'streaming' }];
  }
  const existing = parts[index] as AgentTextPart;
  return replaceAt(parts, index, {
    ...existing,
    text: existing.text + delta,
    state: 'streaming',
  });
}

function closeText(parts: AgentMessagePart[], id: string): AgentMessagePart[] {
  const index = parts.findIndex((p) => p.kind === 'text' && p.id === id);
  if (index < 0) return parts;
  const existing = parts[index] as AgentTextPart;
  return replaceAt(parts, index, { ...existing, state: 'done' });
}

// -- tool helpers ---------------------------------------------------------

function upsertTool(
  parts: AgentMessagePart[],
  toolCallId: string,
  patch: Partial<Omit<AgentToolPart, 'kind' | 'toolCallId'>>,
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => p.kind === 'tool' && p.toolCallId === toolCallId,
  );
  if (index < 0) {
    return [
      ...parts,
      {
        kind: 'tool',
        toolCallId,
        toolName: patch.toolName ?? '',
        metadata: patch.metadata,
        input: patch.input,
        output: patch.output,
        errorText: patch.errorText,
        state: patch.state ?? 'input-streaming',
      },
    ];
  }
  const existing = parts[index] as AgentToolPart;
  return replaceAt(parts, index, {
    ...existing,
    ...patch,
    // Preserve toolName once it's known (input-start → input-available
    // both carry it; output-available does not).
    toolName: patch.toolName ?? existing.toolName,
    state: patch.state ?? existing.state,
  });
}

// -- generic helpers ------------------------------------------------------

type AgentMessagePartByKey<K extends string> = Extract<
  AgentMessagePart,
  Record<K, string>
>;

function upsertById<K extends 'sourceId' | 'toolCallId' | 'elicitationId'>(
  parts: AgentMessagePart[],
  key: K,
  value: string,
  create: () => AgentMessagePartByKey<K>,
  update?: (existing: AgentMessagePart) => AgentMessagePart,
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => (p as Record<string, unknown>)[key] === value,
  );
  if (index < 0) return [...parts, create()];
  if (!update) return parts;
  return replaceAt(parts, index, update(parts[index]));
}

function upsertCitation(
  parts: AgentMessagePart[],
  key: string,
  data: import('./types').CitationData,
): AgentMessagePart[] {
  const index = parts.findIndex(
    (p) => p.kind === 'citation' && p.key === key,
  );
  if (index < 0) {
    return [...parts, { kind: 'citation', key, data }];
  }
  // Already seen — replay. Keep existing (data is byte-identical
  // per envelope-atomic replay).
  return parts;
}

function citationKey(data: import('./types').CitationData): string {
  // Stable fingerprint: cited_text + JSON.stringify(location). The BE
  // always replays the identical payload on reconnect, so this collides
  // exactly when (and only when) we've already seen the citation.
  // JSON.stringify is deterministic for primitive-keyed objects, which
  // matches the BE Pydantic model shape.
  try {
    return `${data.cited_text}${JSON.stringify(data.location)}`;
  } catch {
    return data.cited_text;
  }
}

function replaceAt<T>(items: T[], index: number, value: T): T[] {
  const next = items.slice();
  next[index] = value;
  return next;
}
