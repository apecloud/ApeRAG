'use client';

// SSE frame parser used by the agent runtime stream client.
//
// AI SDK v5 wire shape: `id: <seq>\ndata: <single-line JSON>\n\n`.
// `event:` is never set (the BE keys discrimination on the JSON `type`
// field). Only the LAST part of a fan-out group carries `id:`; mid-group
// frames are continuation frames without it. SSE comments (`: ...`) and
// blank-only buffers are heartbeats — we keep the connection alive but
// do not surface them.
//
// We deliberately re-implement the wire framing instead of using the AI
// SDK's `readUIMessageStream` because (a) we need to validate the
// response header before consuming, which the SDK helper does not
// expose, (b) the SDK helper expects a `ReadableStream<UIMessageChunk>`
// already deserialized — we are the layer that does the deserialize.

import type { StreamPart } from './types';

export type SseFrame = {
  id: number | null;
  part: StreamPart;
};

export type SseParseResult = {
  frames: SseFrame[];
  carry: string;
};

const FRAME_DELIMITER = /\r?\n\r?\n/;

/**
 * Split an SSE buffer into frames at blank-line delimiters and parse
 * each frame's `id:` and `data:` fields. The trailing partial frame
 * (if any) is returned in `carry` so the caller can prepend it to the
 * next chunk. Comment-only frames (heartbeats) are skipped.
 */
export function parseSseChunk(buffer: string): SseParseResult {
  const segments = buffer.split(FRAME_DELIMITER);
  const carry = segments.pop() ?? '';
  const frames: SseFrame[] = [];

  for (const segment of segments) {
    if (!segment) continue;
    const frame = parseSseFrame(segment);
    if (frame) frames.push(frame);
  }

  return { frames, carry };
}

function parseSseFrame(segment: string): SseFrame | null {
  let id: number | null = null;
  const dataLines: string[] = [];

  for (const rawLine of segment.split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    if (!line) continue;
    if (line.startsWith(':')) continue; // comment / heartbeat
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    const value =
      colon < 0
        ? ''
        : line[colon + 1] === ' '
          ? line.slice(colon + 2)
          : line.slice(colon + 1);

    if (field === 'data') {
      dataLines.push(value);
    } else if (field === 'id') {
      const parsed = Number.parseInt(value, 10);
      if (Number.isFinite(parsed)) id = parsed;
    }
  }

  if (dataLines.length === 0) return null;

  let part: StreamPart;
  try {
    part = JSON.parse(dataLines.join('\n')) as StreamPart;
  } catch {
    return null;
  }

  if (!part || typeof (part as { type?: unknown }).type !== 'string') {
    return null;
  }

  return { id, part };
}
