'use client';

// React hook around the agent runtime stream client (D8.4a).
//
// Public seam consumed by #77 (parts renderer) and #78 (interactive
// consent + elicitation UI):
//
//   const { parts, transientActivity, status, errorText, lastSequence,
//           abort } = useAgentTurnStream({ chatId, turnId, streamUrl,
//                                          initialSequence });
//
// `streamUrl == null` means "do not connect" — used for terminal turns
// loaded from snapshot (status will resolve from the snapshot fetch in
// the caller).

import { useEffect, useReducer, useRef } from 'react';

import { consumeAgentStream } from './stream-client';
import {
  applyPart,
  initialReducerState,
  type ReducerState,
} from './reducer';
import type { ActivityData, AgentMessagePart, AgentStreamStatus, StreamPart } from './types';

type ReducerAction =
  | { kind: 'reset'; initialSequence: number }
  | { kind: 'connect' }
  | { kind: 'part'; part: StreamPart; eventId: number | null }
  | { kind: 'closed'; reason: 'completed' | 'aborted' | 'error'; error?: string };

function reduce(state: ReducerState, action: ReducerAction): ReducerState {
  switch (action.kind) {
    case 'reset':
      return {
        ...initialReducerState(),
        lastSequence: action.initialSequence,
      };
    case 'connect':
      // Only flip to connecting if we are not already mid-stream; this
      // keeps `status` stable across silent reconnects (the consumer
      // hides reconnect blips from the UI).
      if (state.status === 'streaming') return state;
      if (
        state.status === 'completed' ||
        state.status === 'failed' ||
        state.status === 'cancelled' ||
        state.status === 'aborted'
      ) {
        return state;
      }
      return { ...state, status: 'connecting' };
    case 'part':
      return applyPart(state, action.part, action.eventId);
    case 'closed':
      if (action.reason === 'completed') {
        // BE-side terminal — reducer already moved status off
        // 'streaming' via the finish/error/abort frame. If we missed
        // it, pin to 'completed' so the UI doesn't get stuck.
        if (state.status === 'streaming' || state.status === 'connecting') {
          return { ...state, status: 'completed' };
        }
        return state;
      }
      if (action.reason === 'aborted') {
        // Local abort triggered by the user; we don't clobber a
        // BE-driven completion that landed in the same tick.
        if (
          state.status === 'completed' ||
          state.status === 'failed' ||
          state.status === 'cancelled'
        ) {
          return state;
        }
        return { ...state, status: 'aborted' };
      }
      // 'error'
      return {
        ...state,
        status: 'failed',
        errorText: action.error ?? state.errorText ?? 'stream error',
      };
  }
}

export type UseAgentTurnStreamInput = {
  chatId: string;
  turnId: string;
  /** Absolute or relative SSE URL for the turn stream. `null` skips connection. */
  streamUrl: string | null;
  /** Sequence id of the last envelope already in `parts` (snapshot reload, etc.). */
  initialSequence?: number;
};

export type UseAgentTurnStreamResult = {
  parts: AgentMessagePart[];
  transientActivity: ActivityData | null;
  status: AgentStreamStatus;
  errorText: string | null;
  lastSequence: number;
  abort: () => void;
};

export function useAgentTurnStream(
  input: UseAgentTurnStreamInput,
): UseAgentTurnStreamResult {
  const { chatId: _chatId, turnId, streamUrl, initialSequence = 0 } = input;
  void _chatId; // chatId is currently identity-bound via the same-origin cookie; reserved for future scoping.

  const [state, dispatch] = useReducer(
    reduce,
    initialSequence,
    (seq): ReducerState => ({
      ...initialReducerState(),
      lastSequence: seq,
    }),
  );

  // Keep the latest sequence in a ref so the reconnect loop can read
  // it without re-running effects on every part arrival.
  const sequenceRef = useRef(initialSequence);
  sequenceRef.current = state.lastSequence;

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!streamUrl || !turnId) return;

    let cancelled = false;
    const abortController = new AbortController();
    abortRef.current = abortController;
    dispatch({ kind: 'reset', initialSequence });

    (async () => {
      // Reconnect loop with exponential-ish backoff. We stop as soon as
      // the stream closes for a non-recoverable reason (completed,
      // aborted, or after `MAX_ATTEMPTS` consecutive errors).
      let attempt = 0;
      const MAX_ATTEMPTS = 5;
      while (!cancelled && !abortController.signal.aborted) {
        dispatch({ kind: 'connect' });
        const closeInfo = await consumeAgentStream({
          url: streamUrl,
          signal: abortController.signal,
          lastEventId: sequenceRef.current,
          onPart: (part, eventId) => {
            if (cancelled) return;
            dispatch({ kind: 'part', part, eventId });
          },
          onEventId: (eventId) => {
            if (cancelled) return;
            // Track the highest seen id directly; the reducer also
            // bumps lastSequence on parts that carry an id, but the
            // ref needs to lead so the next reconnect picks up
            // immediately.
            if (eventId > sequenceRef.current) {
              sequenceRef.current = eventId;
            }
          },
        });

        if (cancelled) break;

        if (closeInfo.reason === 'completed') {
          dispatch({ kind: 'closed', reason: 'completed' });
          break;
        }
        if (closeInfo.reason === 'aborted') {
          dispatch({ kind: 'closed', reason: 'aborted' });
          break;
        }
        attempt += 1;
        if (attempt >= MAX_ATTEMPTS) {
          dispatch({
            kind: 'closed',
            reason: 'error',
            error: closeInfo.error,
          });
          break;
        }
        await sleep(Math.min(2000, 200 * 2 ** (attempt - 1)));
      }
    })();

    return () => {
      cancelled = true;
      abortController.abort();
      if (abortRef.current === abortController) {
        abortRef.current = null;
      }
    };
    // initialSequence is read once at hook-mount time via the lazy
    // reducer initializer; we intentionally do not re-run on changes
    // because the runtime always increments the cursor and we don't
    // want to silently drop accumulated parts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamUrl, turnId]);

  return {
    parts: state.parts,
    transientActivity: state.transientActivity,
    status: state.status,
    errorText: state.errorText,
    lastSequence: state.lastSequence,
    abort: () => abortRef.current?.abort(),
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
