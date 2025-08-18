import { ChatMessage, Feedback, Reference } from '@/api';
import { PageContainer } from '@/components';
import { api } from '@/services';
import { useWebSocket } from 'ahooks';
import { ReadyState } from 'ahooks/lib/useWebSocket';
import { Result, theme } from 'antd';
import _ from 'lodash';
import { useCallback, useEffect, useState } from 'react';
import { animateScroll as scroll } from 'react-scroll';
import { UndrawFirmware } from 'react-undraw-illustrations';
import { FormattedMessage, useModel, useParams } from 'umi';
import { ChatInput, ChatInputProps } from './_chat_input';
import { ChatMessageItem } from '@/components/chat/ChatMessageItem';

export default () => {
  const { chat, getChat, setChat, bot } = useModel('bot');
  const { botId, chatId } = useParams();
  const [messages, setMessages] = useState<ChatMessage[][]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const { token } = theme.useToken();

  const isAgent = bot?.type === 'agent';
  const protocol = window.location.protocol === 'http:' ? 'ws://' : 'wss://';
  const host = window.location.host;

  const { sendMessage, readyState, disconnect, connect } = useWebSocket(
    `${protocol}${host}/api/v1/bots/${botId}/chats/${chatId}/connect`,
    {
      onMessage: (message) => {
        const fragment = JSON.parse(message.data) as ChatMessage;
        if (fragment.type === 'start') {
          setLoading(true);
        }
        if (fragment.type === 'stop') {
          setLoading(false);
        }

        setMessages((msgs) => {
          const parts = msgs.findLast((parts) => {
            return Boolean(
              parts.find(
                (part) => part.id === fragment.id && part.role === 'ai',
              ),
            );
          });

          if (parts) {
            if (fragment.type === 'stop' && Array.isArray(fragment.data)) {
              parts.push({
                type: 'references',
                references: fragment.data as Reference[],
                data: '',
                role: 'ai',
              });
            } else {
              const part = parts.find((p) => {
                if (fragment.type === 'message') {
                  return p.type === 'message';
                } else {
                  return fragment.part_id && fragment.part_id === p.part_id;
                }
              });
              if (part) {
                part.data = (part.data || '') + fragment.data;
              } else {
                parts.push(fragment);
              }
            }
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

  const onSubmit: ChatInputProps['onSubmit'] = useCallback(
    async (params) => {
      const timestamp = Math.floor(new Date().getTime() / 1000);
      const msg: ChatMessage = {
        type: 'message',
        role: 'human',
        data: params.query,
        timestamp,
      };
      setMessages((msgs) => {
        msgs?.push([msg]);
        return [...msgs];
      });

      if (isAgent) {
        sendMessage(JSON.stringify(params));
      } else {
        sendMessage(JSON.stringify(msg));
      }
    },
    [isAgent],
  );

  const handleCancel = useCallback(() => {
    disconnect();
    connect();
    setLoading(false);
  }, []);

  const onVote = async (item: ChatMessage, feedback: Feedback) => {
    if (!botId || !chatId || !item.id) return;

    const res = await api.botsBotIdChatsChatIdMessagesMessageIdPost({
      botId,
      chatId,
      messageId: item.id,
      feedback,
    });
    if (res.status === 200) {
      setMessages((msgs) => {
        const parts = msgs.find((parts) =>
          parts.find((p) => p.id === item.id && p.type === 'references'),
        );
        const referencePart = parts?.find((p) => p.type === 'references');
        if (referencePart) {
          referencePart.feedback = feedback;
        }
        return [...msgs];
      });
    }
  };

  useEffect(() => {
    if (chatId && botId) getChat(botId, chatId);
    return () => {
      setChat(undefined);
    };
  }, [chatId, botId]);

  useEffect(() => {
    setMessages(chat?.history || []);
  }, [chat]);

  useEffect(() => {
    scroll.scrollToBottom({ duration: 0 });
  }, [messages, chat]);

  return (
    <PageContainer style={{ paddingBottom: 140 }}>
      {!_.isEmpty(chat) && _.isEmpty(messages) ? (
        <Result
          icon={
            <UndrawFirmware primaryColor={token.colorPrimary} height="200px" />
          }
          subTitle={<FormattedMessage id="chat.empty_description" />}
        />
      ) : (
        messages?.map((parts, index) => {
          const isAi = Boolean(parts.find((p) => p.role === 'ai'));
          return (
            <ChatMessageItem
              onVote={onVote}
              isAi={isAi}
              parts={parts}
              key={index}
              loading={isAi && loading && index + 1 === messages.length}
            />
          );
        })
      )}
      <ChatInput
        disabled={readyState !== ReadyState.Open}
        loading={loading}
        onSubmit={onSubmit}
        onCancel={handleCancel}
      />
    </PageContainer>
  );
};
