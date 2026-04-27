import { createServerApiClient } from '@/lib/api/typed/server';

import type { Bot, BotList, ChatDetails, ChatList } from './types';

export async function listBots(): Promise<BotList> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v2/bots');
  return data ?? {};
}

export async function getBot(botId: string): Promise<Bot | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v2/bots/{bot_id}', {
    params: { path: { bot_id: botId } },
  });
  return data ?? null;
}

export async function listBotChats(
  botId: string,
  options?: { page?: number; pageSize?: number },
): Promise<ChatList> {
  const client = await createServerApiClient();
  const { data } = await client.GET('/api/v2/bots/{bot_id}/chats', {
    params: {
      path: { bot_id: botId },
      query: {
        page: options?.page ?? 1,
        page_size: options?.pageSize ?? 50,
      },
    },
  });
  return data ?? {};
}

export async function getBotChat(
  botId: string,
  chatId: string,
): Promise<ChatDetails | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/bots/{bot_id}/chats/{chat_id}',
    {
      params: { path: { bot_id: botId, chat_id: chatId } },
    },
  );
  return data ?? null;
}
