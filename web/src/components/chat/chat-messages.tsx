'use client';

import type { ChatDetails, ChatMessage, Feedback } from '@/features/bot/types';
import {
  cancelAgentTurn,
  createAgentTurn,
  getAgentTurnSnapshot,
  mapBackendTurnStatus,
  useAgentTurnStream,
  type AgentStreamStatus,
  type AgentTurnEnvelope,
  type AgentTurnSnapshotEnvelope,
} from '@/features/agent-runtime';
import { useBotContext } from '@/components/providers/bot-provider';
import _ from 'lodash';
import { useParams } from 'next/navigation';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { animateScroll as scroll } from 'react-scroll';
import { toast } from 'sonner';
import { AgentTurnRenderer } from './agent-turn-renderer';
import { ConsentPrompt } from './consent-prompt';
import { ElicitationForm } from './elicitation-form';
import { ChatInput, ChatInputSubmitParams } from './chat-input';
import { MessagePartsUser } from './message-parts-user';

const API_BASE_PATH = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2`;
const ACTIVE_TURN_STORAGE_PREFIX = 'agent-runtime-v3:active-turn:';

type LiveTurn = {
  envelope: AgentTurnEnvelope;
  baselineSnapshot?: AgentTurnSnapshotEnvelope;
  streamUrl: string | null; // null = no live connection (terminal / paused)
  pending: boolean;
};

type PendingUserMessage = {
  key: string;
  query: string;
  timestamp: number;
};

type TurnFeedbackListResponse = {
  items: Array<
    Feedback & {
      turn_id: string;
    }
  >;
};

function isTerminalStatus(status?: string) {
  const upper = String(status ?? '').toUpperCase();
  return upper === 'COMPLETED' || upper === 'FAILED' || upper === 'CANCELLED';
}

function getActiveTurnStorageKey(chatId?: string) {
  return `${ACTIVE_TURN_STORAGE_PREFIX}${chatId || 'unknown'}`;
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) return error.message;
  return fallback;
}

function buildStreamUrl(
  chatId: string | undefined,
  turnId: string,
): string | null {
  if (!chatId) return null;
  const base = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/agent`;
  return `${base}/chats/${chatId}/turns/${turnId}/events`;
}

/**
 * Phase 8 D8.5-FE (#93): synthesize an `AgentTurnEnvelope` from the
 * canonical `AgentTurnSnapshot` returned by `getBotChat()` /
 * `getAgentTurnSnapshot()`. Reload-side fields the renderer needs
 * (status / timestamps / error_message / timeline_cursor) come from
 * the snapshot directly; turn-shape fields the renderer does NOT
 * consume on reload (`bot_id` / `user_id` / `request_id` /
 * `client_idempotency_key` / `model_profile`) are filled with empty
 * placeholders. Live turn submissions still hand the renderer the
 * full envelope returned by `createAgentTurn`.
 */
function envelopeFromSnapshot(
  snapshot: AgentTurnSnapshotEnvelope,
): AgentTurnEnvelope {
  return {
    schema_version: snapshot.schema_version,
    turn_id: snapshot.turn_id,
    chat_id: snapshot.chat_id,
    user_id: '',
    bot_id: '',
    request_id: '',
    client_idempotency_key: '',
    status: snapshot.status,
    input_text: snapshot.input_text ?? '',
    model_profile: {},
    error_code: null,
    error_message: snapshot.error_text ?? null,
    answer_artifact_id: null,
    reference_bundle_artifact_id: null,
    timeline_cursor: snapshot.timeline_cursor,
    started_at: snapshot.started_at,
    finished_at: snapshot.finished_at,
    created_at: snapshot.created_at,
    updated_at: snapshot.updated_at,
  };
}

function liveTurnFromSnapshot(
  chatId: string | undefined,
  snapshot: AgentTurnSnapshotEnvelope,
): LiveTurn {
  const terminal = isTerminalStatus(snapshot.status);
  return {
    envelope: envelopeFromSnapshot(snapshot),
    baselineSnapshot: snapshot,
    streamUrl: terminal ? null : buildStreamUrl(chatId, snapshot.turn_id),
    pending: !terminal,
  };
}

function seedFromHistory(
  chatId: string | undefined,
  history: AgentTurnSnapshotEnvelope[] | null | undefined,
): { liveTurns: Record<string, LiveTurn>; turnOrder: string[] } {
  const liveTurns: Record<string, LiveTurn> = {};
  const turnOrder: string[] = [];
  for (const snapshot of history ?? []) {
    if (!snapshot?.turn_id) continue;
    liveTurns[snapshot.turn_id] = liveTurnFromSnapshot(chatId, snapshot);
    turnOrder.push(snapshot.turn_id);
  }
  return { liveTurns, turnOrder };
}

function userMessagePartsFromText(
  text: string | null | undefined,
  timestamp: number | null | undefined,
): ChatMessage[] {
  if (!text) return [];
  return [
    {
      type: 'message',
      role: 'human',
      data: text,
      timestamp:
        typeof timestamp === 'number' && Number.isFinite(timestamp)
          ? timestamp
          : Math.floor(Date.now() / 1000),
    },
  ];
}

export const ChatMessages = ({ chat }: { chat: ChatDetails }) => {
  const { chatRename } = useBotContext();
  const { chatId } = useParams<{ chatId: string }>();

  // Phase 8 D8.5-FE (#93): `chat.history` now ships canonical
  // `AgentTurnSnapshot[]`; seed `liveTurns` directly from it, no
  // separate per-turn snapshot fetch on first render.
  //
  // The OpenAPI-generated `ChatDetails.history` shape and the
  // FE-curated `AgentTurnSnapshotEnvelope` shape carry the same
  // wire bytes (D8 §2 byte-equal canonical) but are nominally
  // distinct TypeScript types — the OpenAPI union for `parts`
  // references generated `TextPart` / `ToolPart` / etc. while the
  // FE renderer consumes the SDK-aligned `AgentMessagePart` union
  // (with its compile-time `_AgentMessagePartIsSDKCompatible`
  // assertion). The `unknown` cast bridges the two without losing
  // run-time safety; this seam disappears whenever the FE part
  // union is regenerated from the OpenAPI schema.
  const historicalTurns = (chat.history ?? null) as
    | AgentTurnSnapshotEnvelope[]
    | null;
  const initialSeed = useMemo(
    () => seedFromHistory(chatId, historicalTurns),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const [liveTurns, setLiveTurns] = useState<Record<string, LiveTurn>>(
    initialSeed.liveTurns,
  );
  const [turnOrder, setTurnOrder] = useState<string[]>(initialSeed.turnOrder);
  const [pendingUserMessages, setPendingUserMessages] = useState<
    PendingUserMessage[]
  >([]);
  const [feedbackByTurnId, setFeedbackByTurnId] = useState<
    Record<string, Feedback>
  >({});
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);

  const liveTurnsRef = useRef(liveTurns);
  liveTurnsRef.current = liveTurns;

  const activeTurnStorageKey = useMemo(
    () => getActiveTurnStorageKey(chat.id ?? undefined),
    [chat.id],
  );

  const loading = useMemo(
    () =>
      pendingUserMessages.length > 0 ||
      Object.values(liveTurns).some((turn) => turn.pending),
    [liveTurns, pendingUserMessages.length],
  );

  const updateLiveTurn = useCallback(
    (
      turnId: string,
      updater: (previous: LiveTurn | undefined) => LiveTurn | undefined,
    ) => {
      setLiveTurns((previous) => {
        const next = updater(previous[turnId]);
        if (!next) {
          if (!(turnId in previous)) return previous;
          const rest = { ...previous };
          delete rest[turnId];
          return rest;
        }
        return { ...previous, [turnId]: next };
      });
    },
    [],
  );

  const recordTurn = useCallback(
    (turnId: string, liveTurn: LiveTurn) => {
      setLiveTurns((previous) => ({ ...previous, [turnId]: liveTurn }));
      setTurnOrder((previous) =>
        previous.includes(turnId) ? previous : [...previous, turnId],
      );
    },
    [],
  );

  const updateActiveTurn = useCallback(
    (turnId: string | null) => {
      setActiveTurnId(turnId);
      if (typeof window === 'undefined') return;
      if (turnId) {
        window.sessionStorage.setItem(activeTurnStorageKey, turnId);
      } else {
        window.sessionStorage.removeItem(activeTurnStorageKey);
      }
    },
    [activeTurnStorageKey],
  );

  const fetchTurnFeedbacks = useCallback(
    async () =>
      fetch(`${API_BASE_PATH}/chats/${chatId}/feedback`, {
        method: 'GET',
        credentials: 'same-origin',
      }).then(async (response) => {
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        return (await response.json()) as TurnFeedbackListResponse;
      }),
    [chatId],
  );

  const handleSendMessage = useCallback(
    async (params: ChatInputSubmitParams) => {
      if (!chatId) return;

      const pendingKey = `pending-${Date.now()}-${Math.random()
        .toString(36)
        .slice(2, 8)}`;
      const optimistic: PendingUserMessage = {
        key: pendingKey,
        query: params.query,
        timestamp: Math.floor(Date.now() / 1000),
      };
      setPendingUserMessages((previous) => [...previous, optimistic]);

      try {
        const response = await createAgentTurn(chatId, { ...params });
        const turnId = response.turn.turn_id;
        const terminal = isTerminalStatus(response.turn.status);
        recordTurn(turnId, {
          envelope: response.turn,
          streamUrl: terminal ? null : response.stream_url,
          pending: !terminal,
        });
        setPendingUserMessages((previous) =>
          previous.filter((m) => m.key !== pendingKey),
        );
        updateActiveTurn(terminal ? null : turnId);
      } catch (error) {
        console.error('Failed to create agent turn', error);
        setPendingUserMessages((previous) =>
          previous.filter((m) => m.key !== pendingKey),
        );
        toast.error(
          getErrorMessage(error, 'Failed to create a new agent turn.'),
        );
      }
    },
    [chatId, recordTurn, updateActiveTurn],
  );

  const handleCancel = useCallback(async () => {
    if (!activeTurnId || !chatId) return;
    try {
      await cancelAgentTurn(chatId, activeTurnId);
    } catch (error) {
      console.error('Failed to cancel turn', error);
      toast.error(getErrorMessage(error, 'Failed to cancel the running turn.'));
    }
  }, [activeTurnId, chatId]);

  const handleStreamTerminal = useCallback(
    (turnId: string, finalEnvelope: AgentTurnEnvelope) => {
      updateLiveTurn(turnId, (previous) => {
        if (!previous) return previous;
        return {
          ...previous,
          envelope: finalEnvelope,
          streamUrl: null,
          pending: false,
        };
      });
      if (activeTurnId === turnId) {
        updateActiveTurn(null);
      }
      if (chatRename && chat.id) {
        chatRename(chat);
      }
    },
    [activeTurnId, chat, chatRename, updateActiveTurn, updateLiveTurn],
  );

  const handleMessageFeedback = useCallback(
    async (turnId: string, feedback: Feedback) => {
      if (!chatId || !turnId) return;

      const init: RequestInit = {
        method: feedback.type ? 'POST' : 'DELETE',
        credentials: 'same-origin',
        headers: feedback.type
          ? { 'Content-Type': 'application/json' }
          : undefined,
        body: feedback.type ? JSON.stringify(feedback) : undefined,
      };

      const response = await fetch(
        `${API_BASE_PATH}/chats/${chatId}/turns/${turnId}/feedback`,
        init,
      );
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      if (feedback.type) {
        setFeedbackByTurnId((previous) => ({
          ...previous,
          [turnId]: feedback,
        }));
      } else {
        setFeedbackByTurnId((previous) => {
          const next = { ...previous };
          delete next[turnId];
          return next;
        });
      }
    },
    [chatId],
  );

  const isEmpty = turnOrder.length === 0 && pendingUserMessages.length === 0;

  useEffect(() => {
    if (isEmpty) return;
    scroll.scrollToBottom({ duration: 0 });
  }, [isEmpty, liveTurns, pendingUserMessages, turnOrder]);

  useEffect(() => {
    scroll.scrollToBottom({ duration: 0 });
  }, []);

  useEffect(() => {
    if (!chatId) return;

    let cancelled = false;
    setFeedbackByTurnId({});

    const loadTurnFeedbacks = async () => {
      try {
        const response = await fetchTurnFeedbacks();
        if (cancelled) return;
        setFeedbackByTurnId(
          Object.fromEntries(
            response.items.map((item) => [
              item.turn_id,
              {
                type: item.type,
                tag: item.tag,
                message: item.message,
              },
            ]),
          ),
        );
      } catch (error) {
        console.error('Failed to load turn feedback', error);
      }
    };

    void loadTurnFeedbacks();
    return () => {
      cancelled = true;
    };
  }, [chatId, fetchTurnFeedbacks]);

  // Phase 8 D8.5-FE (#93): historical turns now arrive populated in
  // `chat.history` (canonical `AgentTurnSnapshot[]`). The
  // `recoverActiveTurn` effect below still re-fetches the snapshot for
  // the session-storage active turn id so a mid-stream reload picks up
  // any timeline_cursor / status drift since the page load.
  useEffect(() => {
    if (typeof window === 'undefined' || !chat.id || !chatId) return;

    const storedTurnId = window.sessionStorage.getItem(activeTurnStorageKey);
    if (!storedTurnId) return;

    let cancelled = false;
    const recoverActiveTurn = async () => {
      try {
        const snapshot = await getAgentTurnSnapshot(chatId, storedTurnId);
        if (cancelled) return;
        recordTurn(storedTurnId, liveTurnFromSnapshot(chatId, snapshot));
        if (!isTerminalStatus(snapshot.status)) {
          updateActiveTurn(snapshot.turn_id);
        } else {
          updateActiveTurn(null);
        }
      } catch {
        if (!cancelled) updateActiveTurn(null);
      }
    };
    void recoverActiveTurn();
    return () => {
      cancelled = true;
    };
  }, [
    activeTurnStorageKey,
    chat.id,
    chatId,
    recordTurn,
    updateActiveTurn,
  ]);

  return (
    <div className="flex flex-col gap-6 pb-70">
      {turnOrder.map((turnId) => {
        const liveTurn = liveTurns[turnId];
        if (!liveTurn) return null;
        return (
          <AgentTurnStreamCard
            key={turnId}
            chatId={chatId || ''}
            liveTurn={liveTurn}
            feedback={feedbackByTurnId[turnId]}
            onFeedback={handleMessageFeedback}
            onTerminal={handleStreamTerminal}
          />
        );
      })}
      {pendingUserMessages.map((msg) => (
        <MessagePartsUser
          key={msg.key}
          parts={userMessagePartsFromText(msg.query, msg.timestamp)}
        />
      ))}
      <ChatInput
        chat={chat}
        welcome={_.isEmpty(turnOrder) && _.isEmpty(pendingUserMessages)}
        onSubmit={handleSendMessage}
        disabled={false}
        loading={loading}
        onCancel={handleCancel}
      />
    </div>
  );
};

// ---------------------------------------------------------------------------
// AgentTurnStreamCard — renders one turn (historical or live). Owns one
// `useAgentTurnStream` hook per turn; for terminal historical turns the
// hook stays dormant (`streamUrl: null`) and the card falls back to
// `baselineSnapshot.parts` directly.
//
// User input bubble (`MessagePartsUser` driven by `envelope.input_text`)
// is rendered inline above the AI card so historical and live turns
// share one render path. The legacy `MessagePartsAi` branch is gone —
// historical turns now consume canonical UIMessage parts the same way
// live turns do.
// ---------------------------------------------------------------------------

function AgentTurnStreamCard({
  chatId,
  liveTurn,
  feedback,
  onFeedback,
  onTerminal,
}: {
  chatId: string;
  liveTurn: LiveTurn;
  feedback?: Feedback;
  onFeedback: (turnId: string, feedback: Feedback) => Promise<void>;
  onTerminal: (turnId: string, finalEnvelope: AgentTurnEnvelope) => void;
}) {
  const { envelope, baselineSnapshot, streamUrl } = liveTurn;
  const initialSequence =
    baselineSnapshot?.timeline_cursor || envelope.timeline_cursor || 0;

  const stream = useAgentTurnStream({
    chatId,
    turnId: envelope.turn_id,
    streamUrl,
    initialSequence,
  });

  // Phase 8 D8.4d (#90) + D8.5-FE (#93): for dormant terminal turns
  // (`streamUrl == null` and live stream produced nothing), fall back
  // to `baselineSnapshot.parts` and the snapshot's status / error_text
  // so the renderer reads canonical UIMessage parts uniformly.
  const useFallback =
    streamUrl == null &&
    stream.parts.length === 0 &&
    Boolean(baselineSnapshot);
  const displayParts =
    useFallback && baselineSnapshot ? baselineSnapshot.parts : stream.parts;
  const displayStatus: AgentStreamStatus = useFallback
    ? mapBackendTurnStatus(envelope.status)
    : stream.status;
  const displayErrorText =
    displayStatus === 'failed'
      ? (stream.errorText ??
        (useFallback && baselineSnapshot
          ? (baselineSnapshot.error_text ?? null)
          : null) ??
        envelope.error_message ??
        null)
      : stream.errorText;

  const onTerminalRef = useRef(onTerminal);
  onTerminalRef.current = onTerminal;

  useEffect(() => {
    if (
      stream.status === 'completed' ||
      stream.status === 'failed' ||
      stream.status === 'cancelled' ||
      stream.status === 'aborted'
    ) {
      onTerminalRef.current(envelope.turn_id, envelope);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.status, envelope.turn_id]);

  const userParts = useMemo(
    () =>
      userMessagePartsFromText(
        envelope.input_text,
        envelope.created_at
          ? Math.floor(new Date(envelope.created_at).getTime() / 1000)
          : null,
      ),
    [envelope.created_at, envelope.input_text],
  );

  return (
    <div className="flex flex-col gap-4">
      {userParts.length > 0 && <MessagePartsUser parts={userParts} />}
      <AgentTurnRenderer
        chatId={chatId}
        turn={envelope}
        parts={displayParts}
        transientActivity={stream.transientActivity}
        status={displayStatus}
        errorText={displayErrorText}
        feedback={feedback}
        onFeedback={onFeedback}
        ConsentSlot={ConsentPrompt}
        ElicitationSlot={ElicitationForm}
      />
    </div>
  );
}
