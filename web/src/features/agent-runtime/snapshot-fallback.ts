'use client';

// Phase 8 D8.4d (#90): the BE snapshot endpoint now returns canonical
// UIMessage parts directly, so the artifact synthesis adapters
// (`synthesizePartsFromSnapshot` + `extractErrorTextFromSnapshot`) are
// gone. The remaining helpers translate the BE turn status into the
// FE stream status enum so the renderer's status branching stays
// consistent for reload paths.

import type { AgentStreamStatus } from './types';

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
 * Map a backend turn status string back into the stream-side
 * `AgentStreamStatus` enum so the renderer's status branching (badge,
 * answer section heading, etc.) stays consistent with live-stream
 * turns.
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
