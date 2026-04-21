'use client';

import { ChatDetails, ChatMessage, Feedback } from '@/api';
import { useBotContext } from '@/components/providers/bot-provider';
import { apiClient } from '@/lib/api/client';
import _ from 'lodash';
import { useParams } from 'next/navigation';
import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { animateScroll as scroll } from 'react-scroll';
import { toast } from 'sonner';
import {
  AgentArtifactEnvelope,
  AgentTimelineEventEnvelope,
  AgentTurnCard,
  AgentTurnEnvelope,
  AgentTurnSnapshot,
} from './agent-turn-card';
import { ChatInput, ChatInputSubmitParams } from './chat-input';
import { MessagePartsAi } from './message-parts-ai';
import { MessagePartsUser } from './message-parts-user';

const AGENT_RUNTIME_BASE_PATH = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/agent`;
const ACTIVE_TURN_STORAGE_PREFIX = 'agent-runtime-v3:active-turn:';
const LEGACY_WEBSOCKET_FALLBACK =
  process.env.NEXT_PUBLIC_AGENT_LEGACY_WEBSOCKET_FALLBACK === '1';

const TURN_EVENT_TYPES = [
  'turn.started',
  'agent.state.changed',
  'tool.started',
  'tool.progress',
  'tool.finished',
  'external_action.started',
  'external_action.finished',
  'text.delta',
  'artifact.created',
  'turn.completed',
  'turn.failed',
  'turn.cancelled',
  'heartbeat',
] as const;

type AgentTurnCreateResponse = {
  turn: AgentTurnEnvelope;
  stream_url: string;
};

type AgentTurnState = {
  snapshot: AgentTurnSnapshot;
  streamingAnswer: string;
  pending: boolean;
};

function isTerminalStatus(status?: string) {
  return (
    status === 'COMPLETED' || status === 'FAILED' || status === 'CANCELLED'
  );
}

function getActiveTurnStorageKey(chatId?: string) {
  return `${ACTIVE_TURN_STORAGE_PREFIX}${chatId || 'unknown'}`;
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

function getErrorMessage(detail: unknown, fallback: string) {
  if (typeof detail === 'string') return detail;
  if (
    typeof detail === 'object' &&
    detail &&
    'detail' in detail &&
    typeof (detail as { detail?: unknown }).detail === 'string'
  ) {
    return (detail as { detail: string }).detail;
  }
  return fallback;
}

function cloneMessages(messages: ChatMessage[][]) {
  return messages.map((parts) => [...parts]);
}

function hasAnswerArtifact(snapshot: AgentTurnSnapshot) {
  return snapshot.artifacts.some((artifact) => artifact.artifact_type === 'answer');
}

function getStreamingAnswerFromSnapshot(snapshot: AgentTurnSnapshot) {
  if (hasAnswerArtifact(snapshot)) return '';
  return snapshot.timeline
    .filter(
      (event) =>
        event.type === 'text.delta' && typeof event.data.delta === 'string',
    )
    .map((event) => event.data.delta as string)
    .join('');
}

export const ChatMessages = ({ chat }: { chat: ChatDetails }) => {
  const { chatRename } = useBotContext();
  const { botId, chatId } = useParams<{ botId: string; chatId: string }>();
  const [messages, setMessages] = useState<ChatMessage[][]>(chat.history || []);
  const [turnStates, setTurnStates] = useState<Record<string, AgentTurnState>>(
    {},
  );
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [legacyFallbackEnabled, setLegacyFallbackEnabled] = useState(false);

  const streamsRef = useRef<Record<string, EventSource>>({});
  const reconnectTimersRef = useRef<
    Record<string, ReturnType<typeof setTimeout>>
  >({});

  const loading = useMemo(() => {
    return Object.values(turnStates).some((turnState) => turnState.pending);
  }, [turnStates]);

  const activeTurnStorageKey = useMemo(
    () => getActiveTurnStorageKey(chat.id),
    [chat.id],
  );

  const setTurnState = useCallback(
    (
      turnId: string,
      updater: (
        previous: AgentTurnState | undefined,
      ) => AgentTurnState | undefined,
    ) => {
      setTurnStates((previous) => {
        const next = updater(previous[turnId]);
        if (!next) {
          if (!(turnId in previous)) return previous;
          const rest = { ...previous };
          delete rest[turnId];
          return rest;
        }
        return {
          ...previous,
          [turnId]: next,
        };
      });
    },
    [],
  );

  const closeStream = useCallback((turnId: string) => {
    const stream = streamsRef.current[turnId];
    if (stream) {
      stream.close();
      delete streamsRef.current[turnId];
    }
    const timer = reconnectTimersRef.current[turnId];
    if (timer) {
      clearTimeout(timer);
      delete reconnectTimersRef.current[turnId];
    }
  }, []);

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

  const fetchJson = useCallback(
    async <T,>(url: string, init?: RequestInit): Promise<T> => {
      const response = await fetch(url, {
        credentials: 'same-origin',
        ...init,
        headers: {
          'Content-Type': 'application/json',
          ...(init?.headers || {}),
        },
      });

      if (!response.ok) {
        let body: unknown = undefined;
        try {
          body = await response.json();
        } catch {
          body = undefined;
        }
        throw new Error(
          getErrorMessage(
            body,
            `Request failed with status ${response.status}`,
          ),
        );
      }

      return (await response.json()) as T;
    },
    [],
  );

  const fetchArtifact = useCallback(
    async (artifactId: string) => {
      return fetchJson<AgentArtifactEnvelope>(
        `${AGENT_RUNTIME_BASE_PATH}/artifacts/${artifactId}`,
        {
          method: 'GET',
        },
      );
    },
    [fetchJson],
  );

  const fetchSnapshot = useCallback(
    async (turnId: string) => {
      return fetchJson<AgentTurnSnapshot>(
        `${AGENT_RUNTIME_BASE_PATH}/chats/${chatId}/turns/${turnId}`,
        {
          method: 'GET',
        },
      );
    },
    [chatId, fetchJson],
  );

  const mergeSnapshot = useCallback(
    (snapshot: AgentTurnSnapshot) => {
      const streamingAnswer = getStreamingAnswerFromSnapshot(snapshot);
      setTurnState(snapshot.turn.turn_id, () => ({
        snapshot: {
          turn: snapshot.turn,
          timeline: snapshot.timeline,
          artifacts: snapshot.artifacts,
        },
        streamingAnswer,
        pending: !isTerminalStatus(snapshot.turn.status),
      }));
      ensureTurnGroups(snapshot.turn, false);
      if (isTerminalStatus(snapshot.turn.status)) {
        closeStream(snapshot.turn.turn_id);
        if (activeTurnId === snapshot.turn.turn_id) {
          updateActiveTurn(null);
        }
      }
    },
    [
      activeTurnId,
      closeStream,
      ensureTurnGroups,
      setTurnState,
      updateActiveTurn,
    ],
  );

  const fetchAndMergeSnapshot = useCallback(
    async (turnId: string) => {
      const snapshot = await fetchSnapshot(turnId);
      mergeSnapshot(snapshot);
      return snapshot;
    },
    [fetchSnapshot, mergeSnapshot],
  );

  const onStreamEvent = useCallback(
    async (turnId: string, event: AgentTimelineEventEnvelope) => {
      setTurnState(turnId, (previous) => {
        if (!previous) return previous;
        const timeline = previous.snapshot.timeline.some(
          (item) => item.sequence === event.sequence,
        )
          ? previous.snapshot.timeline
          : [...previous.snapshot.timeline, event];
        const nextSnapshot: AgentTurnSnapshot = {
          turn: {
            ...previous.snapshot.turn,
            timeline_cursor: Math.max(
              previous.snapshot.turn.timeline_cursor || 0,
              event.sequence,
            ),
          },
          timeline,
          artifacts: previous.snapshot.artifacts,
        };

        let nextStreamingAnswer = previous.streamingAnswer;
        let nextPending = previous.pending;

        if (
          event.type === 'text.delta' &&
          typeof event.data.delta === 'string'
        ) {
          nextStreamingAnswer += event.data.delta;
        }

        if (event.type === 'turn.completed') {
          nextSnapshot.turn = { ...nextSnapshot.turn, status: 'COMPLETED' };
          nextPending = false;
        } else if (event.type === 'turn.failed') {
          nextSnapshot.turn = {
            ...nextSnapshot.turn,
            status: 'FAILED',
            error_message:
              typeof event.data.error === 'string'
                ? event.data.error
                : previous.snapshot.turn.error_message,
          };
          nextPending = false;
        } else if (event.type === 'turn.cancelled') {
          nextSnapshot.turn = { ...nextSnapshot.turn, status: 'CANCELLED' };
          nextPending = false;
        }

        return {
          snapshot: nextSnapshot,
          streamingAnswer: nextStreamingAnswer,
          pending: nextPending,
        };
      });

      if (event.type === 'artifact.created') {
        const artifactId = String(event.data.artifact_id || '');
        if (artifactId) {
          try {
            const artifact = await fetchArtifact(artifactId);
            setTurnState(turnId, (previous) => {
              if (!previous) return previous;
              const artifacts = previous.snapshot.artifacts.some(
                (item) => item.artifact_id === artifact.artifact_id,
              )
                ? previous.snapshot.artifacts.map((item) =>
                    item.artifact_id === artifact.artifact_id ? artifact : item,
                  )
                : [...previous.snapshot.artifacts, artifact];

              const turn = {
                ...previous.snapshot.turn,
                answer_artifact_id:
                  artifact.artifact_type === 'answer'
                    ? artifact.artifact_id
                    : previous.snapshot.turn.answer_artifact_id,
                reference_bundle_artifact_id:
                  artifact.artifact_type === 'reference_bundle'
                    ? artifact.artifact_id
                    : previous.snapshot.turn.reference_bundle_artifact_id,
              };

              return {
                ...previous,
                snapshot: {
                  ...previous.snapshot,
                  turn,
                  artifacts,
                },
              };
            });
          } catch (error) {
            console.error('Failed to fetch artifact for turn', turnId, error);
          }
        }
      }

      if (
        event.type === 'turn.completed' ||
        event.type === 'turn.failed' ||
        event.type === 'turn.cancelled'
      ) {
        try {
          await fetchAndMergeSnapshot(turnId);
        } catch (error) {
          console.error(
            'Failed to refresh terminal turn snapshot',
            turnId,
            error,
          );
        }
        if (event.type === 'turn.completed' && chatRename && chat.id) {
          chatRename(chat);
        }
      }
    },
    [chat, chatRename, fetchAndMergeSnapshot, fetchArtifact, setTurnState],
  );

  const connectTurnStream = useCallback(
    (turnId: string, afterSequence: number, streamUrl?: string) => {
      closeStream(turnId);

      const url = new URL(
        streamUrl ||
          `${AGENT_RUNTIME_BASE_PATH}/chats/${chatId}/turns/${turnId}/events`,
        window.location.origin,
      );
      url.searchParams.set('after_sequence', String(afterSequence));

      const stream = new EventSource(url.toString(), { withCredentials: true });
      streamsRef.current[turnId] = stream;

      const handleReconnect = async () => {
        closeStream(turnId);
        try {
          const snapshot = await fetchSnapshot(turnId);
          mergeSnapshot(snapshot);
          if (!isTerminalStatus(snapshot.turn.status)) {
            reconnectTimersRef.current[turnId] = setTimeout(() => {
              connectTurnStream(turnId, snapshot.turn.timeline_cursor || 0);
            }, 1000);
          }
        } catch (error) {
          console.error('Failed to reconnect turn stream', turnId, error);
        }
      };

      TURN_EVENT_TYPES.forEach((eventType) => {
        stream.addEventListener(eventType, (rawEvent) => {
          if (eventType === 'heartbeat') return;
          const messageEvent = rawEvent as MessageEvent<string>;
          try {
            const payload = JSON.parse(
              messageEvent.data,
            ) as AgentTimelineEventEnvelope;
            startTransition(() => {
              void onStreamEvent(turnId, payload);
            });
          } catch (error) {
            console.error('Failed to parse turn event', eventType, error);
          }
        });
      });

      stream.onerror = () => {
        void handleReconnect();
      };
    },
    [chatId, closeStream, fetchSnapshot, mergeSnapshot, onStreamEvent],
  );

  const handleLegacyFallback = useCallback(
    async (params: ChatInputSubmitParams) => {
      setLegacyFallbackEnabled(true);
      toast.info('Falling back to legacy websocket chat path.');

      const protocol =
        window.location.protocol === 'http:' ? 'ws://' : 'wss://';
      const host = window.location.host;
      const socket = new WebSocket(
        `${protocol}${host}${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v1/bots/${botId}/chats/${chatId}/connect`,
      );

      socket.onmessage = (message) => {
        const fragment = JSON.parse(message.data) as ChatMessage;
        if (fragment.type === 'stop' && chatRename && chat) {
          chatRename(chat);
        }
        setMessages((previous) => {
          const next = cloneMessages(previous);
          const partsIndex = next.findLastIndex((parts) =>
            Boolean(
              parts.find(
                (part) =>
                  part.id !== 'error' &&
                  part.id === fragment.id &&
                  part.role === 'ai',
              ),
            ),
          );
          const parts = partsIndex > -1 ? next[partsIndex] : undefined;

          if (parts) {
            if (fragment.type === 'stop') {
              parts.push({
                id: fragment.id,
                type: 'references',
                references: Array.isArray(fragment.data) ? fragment.data : [],
                data: '',
                role: 'ai',
              });
            }
            if (fragment.type === 'start') {
              parts.push({
                ...fragment,
                type: 'start',
                data: '',
              });
            } else if (fragment.type === 'message') {
              const part = parts.find((item) => item.type === 'message');
              if (part) {
                part.data = (part.data || '') + (fragment.data || '');
              } else {
                parts.push(fragment);
              }
            } else {
              const part = parts.find(
                (item) => fragment.part_id && fragment.part_id === item.part_id,
              );
              if (part) {
                part.data = (part.data || '') + (fragment.data || '');
              } else {
                parts.push(fragment);
              }
            }
            next[partsIndex] = [...parts];
          } else {
            next.push([{ ...fragment, role: 'ai' }]);
          }

          return next;
        });
      };

      socket.onopen = () => {
        socket.send(JSON.stringify(params));
      };

      socket.onerror = () => {
        toast.error('Legacy websocket fallback failed.');
        socket.close();
      };
    },
    [botId, chat, chatId, chatRename],
  );

  const handleSendMessage = useCallback(
    async (params: ChatInputSubmitParams) => {
      if (!chatId) return;

      setMessages((previous) => [
        ...cloneMessages(previous),
        createUserParts(params.query),
      ]);

      try {
        const response = await fetchJson<AgentTurnCreateResponse>(
          `${AGENT_RUNTIME_BASE_PATH}/chats/${chatId}/turns`,
          {
            method: 'POST',
            body: JSON.stringify({
              ...params,
              client_idempotency_key: crypto.randomUUID(),
            }),
          },
        );

        const turnId = response.turn.turn_id;
        ensureTurnGroups(response.turn, false);
        setTurnState(turnId, () => ({
          snapshot: {
            turn: response.turn,
            timeline: [],
            artifacts: [],
          },
          streamingAnswer: '',
          pending: !isTerminalStatus(response.turn.status),
        }));
        updateActiveTurn(turnId);
        connectTurnStream(turnId, 0, response.stream_url);
      } catch (error) {
        console.error('Failed to create agent turn', error);
        toast.error(
          error instanceof Error
            ? error.message
            : 'Failed to create a new agent turn.',
        );

        if (LEGACY_WEBSOCKET_FALLBACK) {
          await handleLegacyFallback(params);
        }
      }
    },
    [
      chatId,
      connectTurnStream,
      ensureTurnGroups,
      fetchJson,
      handleLegacyFallback,
      setTurnState,
      updateActiveTurn,
    ],
  );

  const handleCancel = useCallback(async () => {
    if (!activeTurnId) return;

    try {
      await fetchJson(
        `${AGENT_RUNTIME_BASE_PATH}/chats/${chatId}/turns/${activeTurnId}/cancel`,
        {
          method: 'POST',
          body: JSON.stringify({}),
        },
      );
    } catch (error) {
      console.error('Failed to cancel turn', error);
      toast.error(
        error instanceof Error
          ? error.message
          : 'Failed to cancel the running turn.',
      );
    }
  }, [activeTurnId, chatId, fetchJson]);

  const handleMessageFeedback = useCallback(
    async (part: ChatMessage, feedback: Feedback) => {
      if (!botId || !chatId || !part.id) return;

      const response =
        await apiClient.defaultApi.botsBotIdChatsChatIdMessagesMessageIdPost({
          botId,
          chatId,
          messageId: part.id,
          feedback,
        });

      if (response.status === 200) {
        setMessages((previous) => {
          const next = cloneMessages(previous);
          const parts = next.find((items) =>
            items.find(
              (message) =>
                message.id === part.id && message.type === 'references',
            ),
          );
          const feedbackPart = parts?.find(
            (message) => message.type === 'references',
          );
          if (feedbackPart) {
            feedbackPart.feedback = feedback;
          }
          return next;
        });
      }
    },
    [botId, chatId],
  );

  useEffect(() => {
    if (!messages.length && !Object.keys(turnStates).length) return;
    scroll.scrollToBottom({ duration: 0 });
  }, [messages, turnStates]);

  useEffect(() => {
    scroll.scrollToBottom({ duration: 0 });
  }, []);

  useEffect(() => {
    setMessages(chat.history || []);
  }, [chat.history]);

  useEffect(() => {
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
        if (cancelled || turnStates[turnId]) continue;
        try {
          const snapshot = await fetchSnapshot(turnId);
          if (!cancelled) {
            mergeSnapshot(snapshot);
          }
        } catch {
          // Ignore legacy chat history groups that are not Agent Runtime V3 turns.
        }
      }
    };

    void loadSnapshots();

    return () => {
      cancelled = true;
    };
  }, [
    activeTurnStorageKey,
    chat.history,
    fetchSnapshot,
    mergeSnapshot,
    turnStates,
  ]);

  useEffect(() => {
    if (typeof window === 'undefined' || !chat.id) return;

    const storedTurnId = window.sessionStorage.getItem(activeTurnStorageKey);
    if (!storedTurnId) return;

    let cancelled = false;

    const recoverActiveTurn = async () => {
      try {
        const snapshot = await fetchSnapshot(storedTurnId);
        if (cancelled) return;
        ensureTurnGroups(snapshot.turn, true);
        mergeSnapshot(snapshot);
        if (!isTerminalStatus(snapshot.turn.status)) {
          updateActiveTurn(snapshot.turn.turn_id);
          connectTurnStream(
            snapshot.turn.turn_id,
            snapshot.turn.timeline_cursor || 0,
          );
        } else {
          updateActiveTurn(null);
        }
      } catch {
        if (!cancelled) {
          updateActiveTurn(null);
        }
      }
    };

    void recoverActiveTurn();

    return () => {
      cancelled = true;
    };
  }, [
    activeTurnStorageKey,
    chat.id,
    connectTurnStream,
    ensureTurnGroups,
    fetchSnapshot,
    mergeSnapshot,
    updateActiveTurn,
  ]);

  useEffect(() => {
    const activeStreams = streamsRef.current;
    return () => {
      Object.keys(activeStreams).forEach((turnId) => closeStream(turnId));
    };
  }, [closeStream]);

  return (
    <div className="flex flex-col gap-6 pb-70">
      {messages.map((parts, index) => {
        const isAI = parts.some((part) => part.role === 'ai');
        const turnId = isAI ? getAiGroupId(parts) : undefined;
        const turnState = turnId ? turnStates[turnId] : undefined;

        return (
          <div key={`${turnId || parts[0]?.timestamp || index}-${index}`}>
            {isAI ? (
              turnState ? (
                <AgentTurnCard
                  snapshot={turnState.snapshot}
                  pending={turnState.pending}
                  streamingAnswer={turnState.streamingAnswer}
                  fallbackParts={parts}
                  onFeedback={handleMessageFeedback}
                />
              ) : (
                <MessagePartsAi
                  pending={false}
                  loading={false}
                  parts={parts}
                  hanldeMessageFeedback={handleMessageFeedback}
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
      {legacyFallbackEnabled && (
        <div className="text-muted-foreground px-4 text-xs">
          Legacy websocket fallback is enabled for this session.
        </div>
      )}
    </div>
  );
};
