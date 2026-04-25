'use client';

// AI SDK-compatible stream transport for the ApeRAG agent runtime (D8.4a).
//
// Why fetch + ReadableStream instead of `EventSource` / `useChat`:
//   * Contract #2 — we must validate the
//     `x-vercel-ai-ui-message-stream: v1` response header before
//     consuming the stream. EventSource exposes no response headers.
//   * Contract #3 — Last-Event-ID resume is driven explicitly from the
//     persisted last `id:` field; we cannot rely on the browser's
//     EventSource auto-reconnect.
//   * `useChat`'s default lifecycle is single-step POST+stream;
//     ApeRAG's runtime is two-phase (POST `/turns` returns a
//     `stream_url`, then GET that URL begins the SSE body), so adapting
//     `useChat` would mean writing a custom `ChatTransport` of
//     comparable complexity to this consumer.
//
// The 6 client-side contracts (architect msg=bad0cd0f) are enforced
// here:
//   1. AI SDK v5 typed parts surface — we deserialize into `StreamPart`
//      which mirrors `aperag/domains/agent_runtime/wire/parts.py`. The
//      reducer downstream emits `AgentMessagePart` shaped to align with
//      `@ai-sdk/react`'s `UIMessagePart` so the renderer can lean on
//      the SDK's type guards.
//   2. Header marker validation — `x-vercel-ai-ui-message-stream: v1`
//      verified before any `onPart` dispatch.
//   3. Resume / error / abort — `Last-Event-ID` header on reconnect;
//      `error` part dispatched then connection terminates (no
//      auto-retry); `abort` part terminates the stream and the consumer
//      cleans up.
//   4. Part-level dedup — handled by the reducer; the client is
//      authoritative on `lastEventId` so `Last-Event-ID` always points
//      at the next envelope.
//   5. Wire shape adoption — wrapped `{type, data:{...}}` for
//      `data-citation/data-tool-consent/data-elicitation/data-activity`
//      passes through unchanged (typed in `types.ts`).
//   6. Transient `data-activity` — emitted to `onPart` as usual; the
//      reducer keeps it out of the persistent parts list.

import { parseSseChunk } from './stream-parser';
import {
  AI_SDK_V5_HEADER,
  AI_SDK_V5_HEADER_VALUE,
  type StreamPart,
} from './types';

export type AgentStreamClientOptions = {
  url: string;
  signal: AbortSignal;
  /**
   * Last sequence id observed by the consumer. Sent as both the
   * `Last-Event-ID` HTTP header and the `after_sequence` query
   * parameter — the BE accepts either, but the header is canonical
   * per the AI SDK v5 spec.
   */
  lastEventId?: number;
  onPart: (part: StreamPart, eventId: number | null) => void;
  /**
   * Called when the BE response advertises a `Last-Event-ID`-style
   * marker on a frame; the client uses this to keep `lastEventId` in
   * sync so the next reconnect resumes correctly.
   */
  onEventId?: (eventId: number) => void;
};

export type AgentStreamCloseReason =
  | 'completed' // BE closed the stream cleanly (terminal envelope reached)
  | 'aborted' // local AbortController fired
  | 'error'; // network or protocol error — `error` is set

export type AgentStreamCloseInfo = {
  reason: AgentStreamCloseReason;
  error?: string;
};

/**
 * Open a single SSE connection and pump frames into `onPart`. Resolves
 * once the stream is fully drained (BE-closed or aborted) and rejects
 * only on protocol-level failures (header mismatch, fetch error).
 *
 * The caller is expected to handle reconnect: this function performs
 * exactly one HTTP request.
 */
export async function consumeAgentStream(
  options: AgentStreamClientOptions,
): Promise<AgentStreamCloseInfo> {
  const url = buildStreamUrl(options.url, options.lastEventId);
  const headers: Record<string, string> = {
    Accept: 'text/event-stream',
  };
  if (options.lastEventId != null && options.lastEventId > 0) {
    headers['Last-Event-ID'] = String(options.lastEventId);
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers,
      signal: options.signal,
    });
  } catch (error) {
    if (options.signal.aborted) {
      return { reason: 'aborted' };
    }
    return { reason: 'error', error: errorText(error) };
  }

  if (!response.ok) {
    return {
      reason: 'error',
      error: `stream HTTP ${response.status}`,
    };
  }

  const protocolHeader = response.headers.get(AI_SDK_V5_HEADER);
  if (protocolHeader !== AI_SDK_V5_HEADER_VALUE) {
    return {
      reason: 'error',
      error: `protocol header mismatch (${AI_SDK_V5_HEADER}=${protocolHeader ?? 'missing'})`,
    };
  }

  const body = response.body;
  if (!body) {
    return { reason: 'error', error: 'empty stream body' };
  }

  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let carry = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      carry += value;
      const { frames, carry: nextCarry } = parseSseChunk(carry);
      carry = nextCarry;
      for (const frame of frames) {
        if (frame.id != null) {
          options.onEventId?.(frame.id);
        }
        options.onPart(frame.part, frame.id);
        if (isTerminalPart(frame.part)) {
          // Terminal part dispatched — the BE will close TCP next
          // tick. We return early so the hook flips to the matching
          // status without waiting on EOF.
          return { reason: 'completed' };
        }
      }
    }
  } catch (error) {
    if (options.signal.aborted) {
      return { reason: 'aborted' };
    }
    return { reason: 'error', error: errorText(error) };
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // already released
    }
  }

  // EOF without a terminal part — the HTTP stream closed cleanly but
  // the turn was not finished/aborted/errored. Treat as recoverable
  // stream loss so the hook's reconnect loop resumes from
  // `lastEventId`. Without this guard a clean mid-turn TCP close
  // would mark the turn completed (Weston msg=63a796f3 blocker #1).
  return {
    reason: 'error',
    error: 'stream closed before terminal frame',
  };
}

function buildStreamUrl(rawUrl: string, lastEventId?: number): string {
  // Resume hint as a query param too — the BE accepts `after_sequence`
  // as a fallback path for clients that cannot set `Last-Event-ID`
  // (e.g. when reconnecting through a same-origin redirect that drops
  // custom headers).
  let url: URL;
  try {
    url = new URL(rawUrl, window.location.origin);
  } catch {
    return rawUrl;
  }
  if (lastEventId != null && lastEventId > 0) {
    url.searchParams.set('after_sequence', String(lastEventId));
  }
  return url.toString();
}

function isTerminalPart(part: StreamPart): boolean {
  // The BE always closes the stream after `finish` / `error` / `abort`;
  // detecting these client-side lets us return early without waiting
  // for the BE's TCP close. `error` and `abort` are surfaced through
  // `onPart` first so the reducer can still react.
  return part.type === 'finish' || part.type === 'error' || part.type === 'abort';
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
