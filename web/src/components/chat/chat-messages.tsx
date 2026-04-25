'use client';

import type { ChatDetails, ChatMessage, Feedback } from '@/features/bot/types';
import {
  cancelAgentTurn,
  createAgentTurn,
  extractErrorTextFromSnapshot,
  getAgentTurnSnapshot,
  mapBackendTurnStatus,
  synthesizePartsFromSnapshot,
  useAgentTurnStream,
  type AgentMessagePart,
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
import { MessagePartsAi } from './message-parts-ai';
import { MessagePartsUser } from './message-parts-user';

const API_BASE_PATH = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2`;
const ACTIVE_TURN_STORAGE_PREFIX = 'agent-runtime-v3:active-turn:';

type LiveTurn = {
  envelope: AgentTurnEnvelope;
  baselineSnapshot?: AgentTurnSnapshotEnvelope;
  streamUrl: string | null; // null = no live connection (terminal / paused)
  pending: boolean;
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

function createUserParts(query: string): ChatMessage[] {
  return [
    {
      type: 'message',
      role: 'human',
      data: query,
      timestamp: Math.floor(Date.now() / 1000),
    },
  ];
}

function createAiPlaceholder(turnId: string): ChatMessage[] {
  return [
    {
      id: turnId,
      type: 'start',
      role: 'ai',
      data: '',
    },
  ];
}

function getAiGroupId(parts: ChatMessage[]) {
  const aiIds = [
    ...new Set(
      parts
        .filter((part) => part.role === 'ai' && part.id)
        .map((part) => part.id as string),
    ),
  ];
  return aiIds.length === 1 ? aiIds[0] : undefined;
}

function getErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) return error.message;
  return fallback;
}

function cloneMessages(messages: ChatMessage[][]) {
  return messages.map((parts) => [...parts]);
}

export const ChatMessages = ({ chat }: { chat: ChatDetails }) => {
  const { chatRename } = useBotContext();
  const { chatId } = useParams<{ chatId: string }>();
  const [messages, setMessages] = useState<ChatMessage[][]>(chat.history || []);
  const [liveTurns, setLiveTurns] = useState<Record<string, LiveTurn>>({});
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
    () => Object.values(liveTurns).some((turn) => turn.pending),
    [liveTurns],
  );

  const updateLiveTurn = useCallback(
    (turnId: string, updater: (previous: LiveTurn | undefined) => LiveTurn | undefined) => {
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

  const ensureTurnGroups = useCallback(
    (turn: AgentTurnEnvelope, includeUserQuery: boolean) => {
      setMessages((previous) => {
        const next = cloneMessages(previous);
        const aiIndex = next.findIndex(
          (parts) => getAiGroupId(parts) === turn.turn_id,
        );

        if (includeUserQuery) {
          const hasUserMessage = next.some((parts) =>
            parts.some(
              (part) =>
                part.role === 'human' &&
                part.type === 'message' &&
                part.data === turn.input_text,
            ),
          );
          if (!hasUserMessage && turn.input_text) {
            next.push(createUserParts(turn.input_text));
          }
        }

        if (aiIndex === -1) {
          next.push(createAiPlaceholder(turn.turn_id));
        }

        return next;
      });
    },
    [],
  );

  const seedFromSnapshot = useCallback(
    (snapshot: AgentTurnSnapshotEnvelope) => {
      const turnId = snapshot.turn.turn_id;
      const terminal = isTerminalStatus(snapshot.turn.status);
      updateLiveTurn(turnId, () => ({
        envelope: snapshot.turn,
        baselineSnapshot: snapshot,
        streamUrl: terminal ? null : buildStreamUrl(chatId, turnId),
        pending: !terminal,
      }));
      ensureTurnGroups(snapshot.turn, false);
      if (terminal && activeTurnId === turnId) {
        updateActiveTurn(null);
      }
    },
    [activeTurnId, chatId, ensureTurnGroups, updateActiveTurn, updateLiveTurn],
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

      setMessages((previous) => [
        ...cloneMessages(previous),
        createUserParts(params.query),
      ]);

      try {
        const response = await createAgentTurn(chatId, {
          ...params,
        });
        const turnId = response.turn.turn_id;
        ensureTurnGroups(response.turn, false);
        const terminal = isTerminalStatus(response.turn.status);
        updateLiveTurn(turnId, () => ({
          envelope: response.turn,
          streamUrl: terminal ? null : response.stream_url,
          pending: !terminal,
        }));
        updateActiveTurn(terminal ? null : turnId);
      } catch (error) {
        console.error('Failed to create agent turn', error);
        toast.error(getErrorMessage(error, 'Failed to create a new agent turn.'));
      }
    },
    [chatId, ensureTurnGroups, updateActiveTurn, updateLiveTurn],
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
        headers: feedback.type ? { 'Content-Type': 'application/json' } : undefined,
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
        setFeedbackByTurnId((previous) => ({ ...previous, [turnId]: feedback }));
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

  useEffect(() => {
    if (!messages.length && !Object.keys(liveTurns).length) return;
    scroll.scrollToBottom({ duration: 0 });
  }, [messages, liveTurns]);

  useEffect(() => {
    scroll.scrollToBottom({ duration: 0 });
  }, []);

  useEffect(() => {
    setMessages(chat.history || []);
  }, [chat.history]);

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

  useEffect(() => {
    if (!chatId) return;

    const storedActiveTurnId =
      typeof window === 'undefined'
        ? null
        : window.sessionStorage.getItem(activeTurnStorageKey);

    const potentialTurnIds = [
      ...new Set(
        (chat.history || [])
          .map((parts) => getAiGroupId(parts))
          .filter(
            (value): value is string =>
              Boolean(value) && value !== storedActiveTurnId,
          ),
      ),
    ];

    if (!potentialTurnIds.length) return;

    let cancelled = false;
    const loadSnapshots = async () => {
      for (const turnId of potentialTurnIds) {
        if (cancelled || liveTurnsRef.current[turnId]) continue;
        try {
          const snapshot = await getAgentTurnSnapshot(chatId, turnId);
          if (!cancelled) seedFromSnapshot(snapshot);
        } catch (error) {
          console.error('Failed to load historical turn snapshot', turnId, error);
        }
      }
    };
    void loadSnapshots();
    return () => {
      cancelled = true;
    };
  }, [activeTurnStorageKey, chat.history, chatId, seedFromSnapshot]);

  useEffect(() => {
    if (typeof window === 'undefined' || !chat.id || !chatId) return;

    const storedTurnId = window.sessionStorage.getItem(activeTurnStorageKey);
    if (!storedTurnId) return;

    let cancelled = false;
    const recoverActiveTurn = async () => {
      try {
        const snapshot = await getAgentTurnSnapshot(chatId, storedTurnId);
        if (cancelled) return;
        ensureTurnGroups(snapshot.turn, true);
        seedFromSnapshot(snapshot);
        if (!isTerminalStatus(snapshot.turn.status)) {
          updateActiveTurn(snapshot.turn.turn_id);
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
    ensureTurnGroups,
    seedFromSnapshot,
    updateActiveTurn,
  ]);

  return (
    <div className="flex flex-col gap-6 pb-70">
      {messages.map((parts, index) => {
        const isAI = parts.some((part) => part.role === 'ai');
        const turnId = isAI ? getAiGroupId(parts) : undefined;
        const liveTurn = turnId ? liveTurns[turnId] : undefined;

        return (
          <div key={`${turnId || parts[0]?.timestamp || index}-${index}`}>
            {isAI ? (
              liveTurn ? (
                <AgentTurnStreamCard
                  chatId={chatId || ''}
                  liveTurn={liveTurn}
                  feedback={turnId ? feedbackByTurnId[turnId] : undefined}
                  onFeedback={handleMessageFeedback}
                  onTerminal={handleStreamTerminal}
                />
              ) : (
                <MessagePartsAi
                  pending={false}
                  loading={false}
                  parts={parts}
                  feedback={turnId ? feedbackByTurnId[turnId] : undefined}
                  onFeedback={handleMessageFeedback}
                />
              )
            ) : (
              <MessagePartsUser parts={parts} />
            )}
          </div>
        );
      })}
      <ChatInput
        chat={chat}
        welcome={_.isEmpty(messages)}
        onSubmit={handleSendMessage}
        disabled={false}
        loading={loading}
        onCancel={handleCancel}
      />
    </div>
  );
};

function buildStreamUrl(chatId: string | undefined, turnId: string): string | null {
  if (!chatId) return null;
  const base = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/agent`;
  return `${base}/chats/${chatId}/turns/${turnId}/events`;
}

// ---------------------------------------------------------------------------
// AgentTurnStreamCard — child component that owns one live
// `useAgentTurnStream` hook per turn and feeds the result straight into
// the new parts renderer. The hook seam is the contract; #78 plugs
// interactive consent / elicitation slots in via the renderer's
// `ConsentSlot` / `ElicitationSlot` props.
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
    baselineSnapshot?.turn.timeline_cursor || envelope.timeline_cursor || 0;

  const stream = useAgentTurnStream({
    chatId,
    turnId: envelope.turn_id,
    streamUrl,
    initialSequence,
  });

  // dongdong msg=97336fb9 — when the hook is dormant for a terminal
  // historical turn (`streamUrl: null`, no live frames), synthesize
  // parts + status from the snapshot's legacy artifacts so the
  // renderer shows the completed answer + references instead of an
  // empty `idle` state. Read-only fallback; deletes when the BE
  // snapshot endpoint returns UIMessages.
  const useFallback =
    streamUrl == null && stream.parts.length === 0 && Boolean(baselineSnapshot);
  const fallbackParts = useMemo<AgentMessagePart[]>(
    () =>
      useFallback && baselineSnapshot
        ? synthesizePartsFromSnapshot(baselineSnapshot)
        : [],
    [useFallback, baselineSnapshot],
  );
  const displayParts = useFallback ? fallbackParts : stream.parts;
  const displayStatus: AgentStreamStatus = useFallback
    ? mapBackendTurnStatus(envelope.status)
    : stream.status;
  const fallbackErrorText = useMemo(
    () =>
      useFallback && baselineSnapshot
        ? extractErrorTextFromSnapshot(baselineSnapshot)
        : null,
    [useFallback, baselineSnapshot],
  );
  const displayErrorText =
    displayStatus === 'failed'
      ? (stream.errorText ??
        fallbackErrorText ??
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

  return (
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
  );
}
