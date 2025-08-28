'use client';

import { ChatDetails, ChatMessage, Feedback, Reference } from '@/api';

import { useWebSocket } from 'ahooks';
import { animateScroll as scroll } from 'react-scroll';

import { useWorkspaceContext } from '@/components/providers/workspace-provider';
import { apiClient } from '@/lib/api/client';
import { ReadyState } from 'ahooks/lib/useWebSocket';
import { motion } from 'framer-motion';
import _ from 'lodash';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChatInput, ChatInputSubmitParams } from './chat-input';
import { MessagePartsAi } from './message-parts-ai';
import { MessagePartsUser } from './message-parts-user';

export const ChatMessages = ({ chat }: { chat?: ChatDetails }) => {
  const { chatRename } = useWorkspaceContext();
  const { botId, chatId } = useParams<{ botId: string; chatId: string }>();
  const [messages, setMessages] = useState<Array<Array<ChatMessage>>>(
    chat?.history || [],
  );
  // const [messagesLoading, setMessagesLoading] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(false);
  const { protocol, host } = useMemo(() => {
    if (typeof window !== 'undefined') {
      return {
        protocol: window.location.protocol === 'http:' ? 'ws://' : 'wss://',
        host: window.location.host,
      };
    } else {
      return {
        protocol: 'ws://',
        host: 'localhost:8000',
      };
    }
  }, []);

  const { sendMessage, readyState, disconnect, connect } = useWebSocket(
    `${protocol}${host}${process.env.NEXT_PUBLIC_BASE_PATH}${process.env.NEXT_PUBLIC_API_SERVER_BASE_PATH}/bots/${botId}/chats/${chatId}/connect`,
    {
      onMessage: (message) => {
        const fragment = JSON.parse(message.data) as ChatMessage;
        if (fragment.type === 'start') {
          setLoading(true);
        }
        if (fragment.type === 'stop') {
          setLoading(false);
          if (chatRename && chat) {
            chatRename(chat);
          }
        }
        setMessages((msgs) => {
          const partsIndex = msgs.findLastIndex((parts) => {
            return Boolean(
              parts.find(
                (part) =>
                  part.id !== 'error' &&
                  part.id === fragment.id &&
                  part.role === 'ai',
              ),
            );
          });
          const parts = partsIndex > -1 ? msgs[partsIndex] : undefined;

          if (parts) {
            if (fragment.type === 'stop') {
              parts.push({
                id: fragment.id,
                type: 'references',
                references: Array.isArray(fragment.data)
                  ? (fragment.data as Reference[])
                  : [],
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
              const part = parts.find((p) => p.type === 'message');
              if (part) {
                part.data = (part.data || '') + fragment.data;
              } else {
                parts.push(fragment);
              }
            } else {
              const part = parts.find(
                (p) => fragment.part_id && fragment.part_id === p.part_id,
              );
              if (part) {
                part.data = (part.data || '') + fragment.data;
              } else {
                parts.push(fragment);
              }
            }
            msgs[partsIndex] = [...parts];
          } else {
            msgs.push([
              {
                ...fragment,
                role: 'ai',
              },
            ]);
          }
          return [...msgs];
        });
      },
    },
  );

  const handleSendMessage = useCallback(
    (params: ChatInputSubmitParams) => {
      const timestamp = Math.floor(new Date().getTime() / 1000);
      const part: ChatMessage = {
        type: 'message',
        role: 'human',
        data: params.query,
        timestamp,
      };
      setMessages((msgs) => {
        msgs?.push([part]);
        return [...msgs];
      });

      sendMessage(JSON.stringify(params));
    },
    [sendMessage],
  );

  const hanldeMessageFeedback = useCallback(
    async (part: ChatMessage, feedback: Feedback) => {
      if (!botId || !chatId || !part.id) return;
      const res =
        await apiClient.defaultApi.botsBotIdChatsChatIdMessagesMessageIdPost({
          botId,
          chatId,
          messageId: part.id,
          feedback,
        });
      if (res.status === 200) {
        setMessages((msgs) => {
          const parts = msgs.find((items) =>
            items.find((p) => p.id === part.id && p.type === 'references'),
          );
          const feedbackPart = parts?.find((p) => p.type === 'references');
          if (feedbackPart) {
            feedbackPart.feedback = feedback;
          }
          return [...msgs];
        });
      }
    },
    [botId, chatId],
  );

  const handleCancel = useCallback(() => {
    disconnect();
    connect();
    setLoading(false);
  }, [connect, disconnect]);

  useEffect(() => {
    if (loading) {
      scroll.scrollToBottom({ duration: 0 });
    }
  }, [messages, chat, loading]);

  useEffect(() => {
    scroll.scrollToBottom({ duration: 0 });
  }, []);

  /**
   * render in server for the first time
   * should delete for production
   */
  // const loadMessages = useCallback(async () => {
  //   const res = await apiClient.defaultApi.botsBotIdChatsChatIdGet({
  //     botId,
  //     chatId,
  //   });
  //   setMessages(res.data.history || []);
  //   setMessagesLoading(true);
  // }, [botId, chatId]);

  // useEffect(() => {
  //   loadMessages();
  // }, [loadMessages]);

  return (
    <div className="flex flex-col gap-6 pb-70">
      {messages.map((parts, index) => {
        const isAI = parts.some((part) => part.role === 'ai');
        const isLoading = loading && index + 1 === messages.length;
        const isAIPending =
          isLoading &&
          parts.filter((p) => p.type !== 'start').length === 0 &&
          isAI;

        return (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.3,
              ease: 'easeIn',
            }}
          >
            {isAI ? (
              <MessagePartsAi
                pending={isAIPending}
                loading={isLoading}
                parts={parts}
                hanldeMessageFeedback={hanldeMessageFeedback}
              />
            ) : (
              <MessagePartsUser parts={parts} />
            )}
          </motion.div>
        );
      })}
      <ChatInput
        welcome={_.isEmpty(messages)}
        onSubmit={handleSendMessage}
        disabled={readyState !== ReadyState.Open}
        loading={loading}
        onCancel={handleCancel}
      />
    </div>
  );
};
