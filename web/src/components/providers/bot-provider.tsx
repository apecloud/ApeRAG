'use client';

import { useCallback, useEffect, useState } from 'react';

import {
  createBot,
  createBotChat,
  deleteBotChat,
  generateBotChatTitle,
  listBotChats,
  updateBotChat,
} from '@/features/bot/client-api';
import type { Bot, Chat, ChatDetails } from '@/features/bot/types';
import { listCollections } from '@/features/collection/client-api';
import type { CollectionView } from '@/features/collection/types';
import { getAvailableModels } from '@/features/providers/client-api';
import type { ModelSpec } from '@/features/providers/types';
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
  collections: CollectionView[];
  providerModels: ProviderModels;
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

  const [collections, setCollections] = useState<CollectionView[]>([]);
  const [providerModels, setProviderModels] = useState<ProviderModels>([]);
  const botChatsBasePath = workspace ? '/workspace/bots' : '/bots';

  const loadData = useCallback(async () => {
    const [models, collectionsRes] = await Promise.all([
      getAvailableModels(['chat']),
      listCollections(),
    ]);

    const items: ProviderModels = models.map((m) => ({
      label: 'Chat Models',
      name: 'chat',
      models: [{ model_id: m.id, temperature: 0.1, tags: [] }],
    }));
    setCollections(collectionsRes?.items ?? []);
    setProviderModels(items);
  }, []);

  const botCreate = useCallback(async () => {
    const created = await createBot({
      title: 'Default Agent Bot',
      type: 'agent',
    });
    if (created?.id) {
      setBot(created);
    }
  }, []);

  const chatsReload = useCallback(async () => {
    if (!bot?.id) return;
    const chatsRes = await listBotChats(bot.id);
    setChats(chatsRes?.items ?? []);
  }, [bot?.id]);

  const chatDelete = useCallback(
    async (chat: Chat) => {
      if (!chat.bot_id || !chat.id) return;
      await deleteBotChat(chat.bot_id, chat.id);

      if (params.chatId === chat.id) {
        const item = chats?.find((c) => c.id !== chat.id);
        const url = item
          ? `${botChatsBasePath}/${bot?.id}/chats/${item.id}`
          : `${botChatsBasePath}/${bot?.id}/chats`;
        router.push(url);
      }
      chatsReload();
    },
    [bot?.id, botChatsBasePath, chats, chatsReload, params.chatId, router],
  );

  const chatRename = useCallback(
    async (chat: Chat) => {
      if (chat.title !== 'New Chat' || !chat.id || !chat.bot_id) return;
      const titleRes = await generateBotChatTitle(chat.bot_id, chat.id, {
        language: locale,
      });
      const title = titleRes?.title;
      if (title) {
        await updateBotChat(chat.bot_id, chat.id, { title });
        chatsReload();
      }
    },
    [chatsReload, locale],
  );

  const chatCreate = useCallback(async () => {
    if (!bot?.id) return;
    const created = await createBotChat(bot.id);

    if (created?.id) {
      const url = `${botChatsBasePath}/${bot.id}/chats/${created.id}`;
      router.push(url);
      chatsReload();
    }
  }, [bot?.id, botChatsBasePath, chatsReload, router]);

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
