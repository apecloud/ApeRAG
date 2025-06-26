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
import { ChatInput } from './_chat_input';
import { ChatMessageItem } from './_chat_message';

export default () => {
  const { chat, getChat, setChat } = useModel('bot');
  const { botId, chatId } = useParams();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const { token } = theme.useToken();
  const protocol = window.location.protocol === 'http:' ? 'ws://' : 'wss://';
  const host = window.location.host;

  const { sendMessage, readyState } = useWebSocket(
    `${protocol}${host}/api/v1/bots/${botId}/chats/${chatId}/connect`,
    {
      onMessage: (message) => {
        const fragment = JSON.parse(message.data) as ChatMessage;

        if (fragment.type === 'start') {
          setMessages((msgs) =>
            msgs.concat({
              ...fragment,
              role: 'ai',
            }),
          );
          setLoading(true);
        }

        if (fragment.type === 'message') {
          setMessages((msgs) => {
            const last = msgs.findLast((m) => m.id === fragment.id);
            if (last) {
              last.data = (last.data || '') + (fragment.data || '');
            }
            return [...msgs];
          });
        }

        if (fragment.type === 'stop') {
          const references = fragment.data as unknown as Reference[];
          if (references) {
            setMessages((msgs) => {
              const last = msgs.findLast((m) => m.id === fragment.id);
              if (last) {
                last.references = references;
              }
              return [...msgs];
            });
          }
          setLoading(false);
        }

        if (fragment.type === 'error') {
          setMessages((msgs) => {
            const last = msgs.findLast((m) => m.id === fragment.id);
            if (last) {
              last.data = fragment.data;
            }
            return [...msgs];
          });
          setLoading(false);
        }
      }
    },
  );

  const onSubmit = useCallback(async (data: string) => {
    const timestamp = new Date().getTime();
    const msg: ChatMessage = {
      type: 'message',
      role: 'human',
      data,
      timestamp,
    };
    setMessages((msgs) => msgs?.concat(msg));
    sendMessage(JSON.stringify(msg));
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
        const index = msgs.findIndex(
          (m) => m.id === item.id && m.role === 'ai',
        );
        if (index !== -1) msgs.splice(index, 1, { ...item, feedback });
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
    <PageContainer style={{ paddingBottom: 100 }}>
      {!_.isEmpty(chat) && _.isEmpty(messages) ? (
        <Result
          icon={
            <UndrawFirmware primaryColor={token.colorPrimary} height="200px" />
          }
          subTitle={<FormattedMessage id="chat.empty_description" />}
        />
      ) : (
        messages?.map((item, index) => {
          return (
            <ChatMessageItem
              onVote={onVote}
              loading={
                item.role === 'ai' &&
                _.size(messages) === index + 1 &&
                loading &&
                _.isEmpty(item.data)
              }
              item={item}
              key={index}
            />
          );
        })
      )}

      <ChatInput
        loading={readyState !== ReadyState.Open || loading}
        onSubmit={onSubmit}
      />
    </PageContainer>
  );
};
