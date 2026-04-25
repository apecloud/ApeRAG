'use client';

// Reload-path fallback for terminal historical turns.
//
// The agent runtime snapshot endpoint
// (`GET /agent/chats/{cid}/turns/{tid}`) still returns the legacy
// `{turn, timeline, artifacts}` envelope; D8.2 (#74) added at-rest
// UIMessage storage but the read path that the FE calls on reload
// has not yet been migrated to expose UIMessage parts. Until that BE
// change lands, terminal historical turns reload with an empty live
// stream (`streamUrl: null` ⇒ hook never connects ⇒ `parts: []` and
// `status: 'idle'`), which would render as an empty queued card —
// the regression dongdong called out (msg=97336fb9).
//
// Fix: when the hook is dormant for a terminal turn, synthesize a
// minimal `AgentMessagePart[]` from the snapshot's legacy artifacts
// (answer text + reference bundle items) so the renderer shows the
// completed answer + references instead of an empty idle state.
// The synthesis is read-only and never feeds back into the live
// reducer; once the BE snapshot endpoint returns UIMessages, this
// file is a one-line delete.

import type { AgentMessagePart, AgentStreamStatus } from './types';
import type { AgentArtifactEnvelope, AgentTurnSnapshotEnvelope } from './api';

type ReferenceBundleItem = {
  source_id?: string | null;
  title?: string | null;
  snippet?: string | null;
  uri?: string | null;
  score?: number | null;
  metadata?: Record<string, unknown>;
};

const ANSWER_ARTIFACT_TYPE = 'answer';
const REFERENCE_BUNDLE_ARTIFACT_TYPE = 'reference_bundle';

function findArtifact(
  artifacts: AgentArtifactEnvelope[],
  artifactType: string,
): AgentArtifactEnvelope | undefined {
  return artifacts.find((a) => a.artifact_type === artifactType);
}

function extractAnswerText(artifact: AgentArtifactEnvelope): string {
  const payload = artifact.payload || {};
  if (typeof payload.text === 'string') return payload.text;
  if (typeof payload.content === 'string') return payload.content;
  return '';
}

function extractReferenceItems(
  artifact: AgentArtifactEnvelope,
): ReferenceBundleItem[] {
  const items = artifact.payload?.items;
  if (!Array.isArray(items)) return [];
  return items.filter(
    (item): item is ReferenceBundleItem =>
      typeof item === 'object' && item !== null,
  );
}

/**
 * Build a minimal `AgentMessagePart[]` from a terminal turn's legacy
 * snapshot. Currently emits at most:
 *   * one `text` part (from the `answer` artifact's payload.text /
 *     .content)
 *   * one `source-url` per reference bundle item with a usable URL
 *   * one `data-citation` per reference bundle item with a snippet
 *
 * Tool call timeline is intentionally NOT replayed — historical tool
 * calls would need lifecycle reconstruction the snapshot endpoint
 * doesn't provide cheaply. Renderer falls back to an empty activity
 * stream for these turns, which matches the legacy `agent-turn-card`
 * behaviour (it also did not show tool calls for past turns once the
 * answer artifact had landed).
 */
export function synthesizePartsFromSnapshot(
  snapshot: AgentTurnSnapshotEnvelope,
): AgentMessagePart[] {
  const parts: AgentMessagePart[] = [];
  const turnId = snapshot.turn.turn_id;

  const answerArtifact = findArtifact(snapshot.artifacts, ANSWER_ARTIFACT_TYPE);
  if (answerArtifact) {
    const text = extractAnswerText(answerArtifact);
    if (text) {
      parts.push({
        type: 'text',
        id: turnId,
        text,
        state: 'done',
      });
    }
  }

  const referenceArtifact = findArtifact(
    snapshot.artifacts,
    REFERENCE_BUNDLE_ARTIFACT_TYPE,
  );
  if (referenceArtifact) {
    const items = extractReferenceItems(referenceArtifact);
    items.forEach((item, index) => {
      const sourceId =
        (item.source_id ? String(item.source_id) : '') ||
        `${turnId}-ref-${index}`;
      const title = item.title ? String(item.title) : undefined;
      const url = item.uri ? String(item.uri) : undefined;
      if (url) {
        parts.push({
          type: 'source-url',
          sourceId,
          url,
          title,
        });
      }
      const snippet = item.snippet ? String(item.snippet) : '';
      if (snippet) {
        parts.push({
          type: 'data-citation',
          id: `${sourceId}-citation`,
          data: {
            cited_text: snippet,
            location: {
              type: 'url_citation',
              url: url ?? '',
              title,
            },
          },
        });
      }
    });
  }

  return parts;
}

const TERMINAL_BACKEND_STATUSES = new Set([
  'COMPLETED',
  'FAILED',
  'CANCELLED',
]);

export function isTerminalBackendStatus(status: string | undefined): boolean {
  if (!status) return false;
  return TERMINAL_BACKEND_STATUSES.has(status.toUpperCase());
}

/**
 * Map a backend `AgentTurnEnvelope.status` string back into the
 * stream-side `AgentStreamStatus` enum so the renderer's status
 * branching (badge, answer section heading, etc.) stays consistent
 * with live-stream turns.
 */
export function mapBackendTurnStatus(status: string): AgentStreamStatus {
  switch (status.toUpperCase()) {
    case 'COMPLETED':
      return 'completed';
    case 'FAILED':
      return 'failed';
    case 'CANCELLED':
      return 'cancelled';
    case 'RUNNING':
    case 'QUEUED':
      // For non-terminal turns the live stream should be active and
      // drive status; this is a defensive fallback only.
      return 'streaming';
    default:
      return 'idle';
  }
}
