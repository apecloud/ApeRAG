'use client';

// TODO(#77 dongdong): delete this file when the parts renderer ships.
//
// Narrow projection from the new `AgentMessagePart[]` seam back into
// the legacy `AgentTurnSnapshot { turn, timeline, artifacts }` shape so
// that `AgentTurnCard` keeps rendering during the D8.4a/4b transition.
// Boundary (per architect lock msg=ed98280c + dongdong msg=f33e9039):
//
//   * Coverage: `streamingAnswer` (concatenation of every text-block in
//     declaration order, joined with `\n\n`); patched turn `status` /
//     `error_message` / `timeline_cursor` from the live stream;
//     `timeline` and `artifacts` pass-through from `baselineSnapshot`
//     only.
//   * NOT covered: rich activity inference, debug previews, reference
//     bundle items rendered from the new parts stream, error_summary
//     translation, timeline merging from new parts. Those belong to
//     the D8.4b renderer (#77).
//
// This shim has zero callers outside `chat-messages.tsx`. Do not
// extend its surface — every new field would otherwise become a soft
// requirement for the renderer rewrite.

import type {
  AgentMessagePart,
  AgentStreamStatus,
  AgentTextPart,
  AgentToolPart,
} from './types';
import type {
  AgentTurnEnvelope,
  AgentTurnSnapshotEnvelope,
} from './api';

export type LegacySnapshotShim = {
  /**
   * Streaming answer text — concatenation of every text block in
   * declaration order. AI SDK v5 spec allows multiple text blocks per
   * turn (`text-start{id}` → `text-delta{id}` → `text-end{id}`); we
   * group by id and join with `\n\n` so multi-block streams still
   * read naturally in the existing card.
   */
  streamingAnswer: string;
  pending: boolean;
  /** Patched envelope with status reflecting the live stream state. */
  legacySnapshot: AgentTurnSnapshotEnvelope;
};

const TERMINAL_STATUSES: ReadonlySet<AgentStreamStatus> = new Set([
  'completed',
  'failed',
  'cancelled',
  'aborted',
]);

const STREAM_TO_LEGACY_STATUS: Record<AgentStreamStatus, string> = {
  idle: 'QUEUED',
  connecting: 'RUNNING',
  streaming: 'RUNNING',
  completed: 'COMPLETED',
  failed: 'FAILED',
  cancelled: 'CANCELLED',
  aborted: 'CANCELLED',
};

export function projectToLegacySnapshot(input: {
  turn: AgentTurnEnvelope;
  parts: AgentMessagePart[];
  status: AgentStreamStatus;
  errorText: string | null;
  lastSequence: number;
  baselineSnapshot?: AgentTurnSnapshotEnvelope;
}): LegacySnapshotShim {
  const streamingAnswer = collectStreamingAnswer(input.parts);

  const liveStatus =
    input.status === 'idle'
      ? input.turn.status
      : STREAM_TO_LEGACY_STATUS[input.status];

  const turn: AgentTurnEnvelope = {
    ...input.turn,
    status: liveStatus,
    error_message:
      input.status === 'failed'
        ? input.errorText ?? input.turn.error_message ?? null
        : input.turn.error_message ?? null,
    timeline_cursor: Math.max(
      input.turn.timeline_cursor || 0,
      input.lastSequence,
    ),
  };

  return {
    streamingAnswer,
    pending: !TERMINAL_STATUSES.has(input.status),
    legacySnapshot: {
      turn,
      timeline: input.baselineSnapshot?.timeline ?? [],
      artifacts: input.baselineSnapshot?.artifacts ?? [],
    },
  };
}

function collectStreamingAnswer(parts: AgentMessagePart[]): string {
  const texts: string[] = [];
  for (const part of parts) {
    if (part.type === 'text') {
      const value = (part as AgentTextPart).text;
      if (value) texts.push(value);
    }
  }
  return texts.join('\n\n');
}

// Re-exported helper: surface the running tool name (if any) so the
// existing card's loading affordance stays meaningful pre-#77.
export function getRunningToolName(parts: AgentMessagePart[]): string | null {
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    const part = parts[i];
    if (!part.type.startsWith('tool-')) continue;
    const tool = part as AgentToolPart;
    if (
      tool.state === 'input-streaming' ||
      tool.state === 'input-available'
    ) {
      return tool.toolName || null;
    }
  }
  return null;
}
