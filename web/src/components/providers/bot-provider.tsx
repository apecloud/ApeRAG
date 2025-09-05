'use client';

import { apiClient } from '@/lib/api/client';
import { useCallback, useEffect, useState } from 'react';

import { Bot, Chat, ChatDetails, Collection, ModelSpec } from '@/api';
import { useLocale } from 'next-intl';
import { useParams, useRouter } from 'next/navigation';
import { createContext, useContext } from 'react';

export type ProviderModels = {
  label?: string;
  name?: string;
  models?: ModelSpec[];
}[];

type BotContextProps = {
  workspace?: boolean;
  bot?: Bot;
  chats: Chat[];
  mention: boolean;
  collections: Collection[];
  providerModels: ProviderModels;
  botReload?: () => void;
  chatDelete?: (chat: Chat) => void;
  chatCreate?: () => void;
  chatsReload?: () => void;
  chatRename?: (chat: Chat | ChatDetails) => void;
};

const BotContext = createContext<BotContextProps>({
  chats: [],
  mention: true,
  collections: [],
  providerModels: [],
});

export const useBotContext = () => useContext(BotContext);

export const BotProvider = ({
  bot: initBot,
  chats: initChats,
  mention = true,
  workspace,
  children,
}: {
  workspace: boolean;
  bot?: Bot;
  chats: Chat[];
  mention?: boolean;
  children?: React.ReactNode;
}) => {
  const [bot, setBot] = useState<Bot | undefined>(initBot);
  const [chats, setChats] = useState<Chat[]>(initChats || []);
  const params = useParams();
  const router = useRouter();
  const locale = useLocale();

  const [collections, setCollections] = useState<Collection[]>([]);
  const [providerModels, setProviderModels] = useState<ProviderModels>([]);

  const loadData = useCallback(async () => {
    const [modelRes, collectionsRes] = await Promise.all([
      apiClient.defaultApi.availableModelsPost({
        tagFilterRequest: {
          tag_filters: [{ operation: 'AND', tags: ['enable_for_agent'] }],
        },
      }),
      apiClient.defaultApi.collectionsGet(),
    ]);

    const items = modelRes.data.items?.map((m) => {
      return {
        label: m.label,
        name: m.name,
        models: m.completion,
      };
    });
    setCollections(collectionsRes.data.items || []);
    setProviderModels(items || []);
  }, []);

  const botCreate = useCallback(async () => {
    const createRes = await apiClient.defaultApi.botsPost({
      botCreate: {
        title: 'Default Agent Bot',
        type: 'agent',
      },
    });
    if (createRes.data.id) {
      setBot(createRes.data);
    }
  }, []);

  const botReload = useCallback(async () => {
    if (!bot?.id) return;
    const botRes = await apiClient.defaultApi.botsBotIdGet({
      botId: bot.id,
    });
    if (botRes.data.id) {
      setBot(botRes.data);
    }
  }, [bot?.id]);

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
        let url = '';
        if (item) {
          url = `/bots/${bot?.id}/chats/${item.id}`;
        } else {
          url = `/bots/${bot?.id}/chats`;
        }
        if (workspace) {
          url = '/workspace' + url;
        }
        router.push(url);
      }
      chatsReload();
    },
    [bot?.id, chats, chatsReload, params.chatId, router, workspace],
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
      let url = `/bots/${bot.id}/chats/${res.data.id}`;
      if (workspace) {
        url = '/workspace' + url;
      }
      router.push(url);
      chatsReload();
    }
  }, [bot?.id, chatsReload, router, workspace]);

  useEffect(() => {
    if (chats.length === 0) {
      chatCreate();
    }
  }, [chatCreate, chats]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (!bot) {
      botCreate();
    }
  }, [bot, botCreate]);

  return (
    <BotContext.Provider
      value={{
        mention,
        workspace,
        bot,
        chats,
        collections,
        providerModels,
        botReload,
        chatDelete,
        chatCreate,
        chatsReload,
        chatRename,
      }}
    >
      {children}
    </BotContext.Provider>
  );
};
