'use client';

import { apiClient } from '@/lib/api/client';
import { useLocale } from 'next-intl';
import { useParams, useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

import { Bot, Chat, ChatDetails } from '@/api';
import { createContext, useContext } from 'react';

type WorkspaceContextProps = {
  bot: Bot;
  chats?: Chat[];
  chatDelete?: (chat: Chat) => void;
  chatCreate?: () => void;
  chatsReload?: () => void;
  chatRename?: (chat: Chat | ChatDetails) => void;
};

const WorkspaceContext = createContext<WorkspaceContextProps>({
  bot: {},
});

export const useWorkspaceContext = () => useContext(WorkspaceContext);

export const WorkspaceProvider = ({
  bot,
  chats: initChats,
  children,
}: {
  bot: Bot;
  chats?: Chat[];
  children?: React.ReactNode;
}) => {
  const [chats, setChats] = useState<Chat[]>(initChats || []);
  const params = useParams();
  const router = useRouter();
  const locale = useLocale();

  const chatsReload = useCallback(async () => {
    if (!bot?.id) return;
    const chatsRes = await apiClient.defaultApi.botsBotIdChatsGet({
      botId: bot.id,
    });
    //@ts-expect-error api define has a bug
    setChats(chatsRes.data.items || []);
  }, [bot?.id]);

  const chatDelete = useCallback(
    async (chat: Chat) => {
      if (!chat.bot_id || !chat.id) return;
      await apiClient.defaultApi.botsBotIdChatsChatIdDelete({
        botId: chat.bot_id,
        chatId: chat.id,
      });

      if (params.chatId === chat.id) {
        const item = chats?.find((c) => c.id !== chat.id);
        if (item) {
          router.push(`/workspace/agents/${item.bot_id}/chats/${item.id}`);
        } else {
          router.push('/workspace/collections');
        }
      }
      chatsReload();
    },
    [chats, chatsReload, params.chatId, router],
  );

  const chatRename = useCallback(
    async (chat: Chat) => {
      if (chat.title !== 'New Chat' || !chat.id || !chat.bot_id) return;
      const titleRes = await apiClient.defaultApi.botsBotIdChatsChatIdTitlePost(
        {
          chatId: chat.id,
          botId: chat.bot_id,
          titleGenerateRequest: {
            language: locale,
          },
        },
      );
      const title = titleRes.data.title;
      if (title) {
        await apiClient.defaultApi.botsBotIdChatsChatIdPut({
          chatId: chat.id,
          botId: chat.bot_id,
          chatUpdate: {
            title,
          },
        });
        chatsReload();
      }
    },
    [chatsReload, locale],
  );

  const chatCreate = useCallback(async () => {
    if (!bot?.id) return;
    const res = await apiClient.defaultApi.botsBotIdChatsPost({
      botId: bot.id,
      chatCreate: {
        title: '',
      },
    });

    if (res.data.id) {
      router.push(`/workspace/agents/${bot.id}/chats/${res.data.id}`);
      chatsReload();
    }
  }, [bot?.id, chatsReload, router]);

  return (
    <WorkspaceContext.Provider
      value={{
        bot,
        chats,
        chatDelete,
        chatCreate,
        chatsReload,
        chatRename,
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
};
