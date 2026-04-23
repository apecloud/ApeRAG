'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  Bot,
  BotCreate,
  BotList,
  BotUpdateRequest,
  Chat,
  ChatDetails,
  ChatList,
  ChatUpdate,
  TitleGenerateRequest,
  TitleGenerateResponse,
} from './types';

export async function createBot(input: BotCreate): Promise<Bot | undefined> {
  const { data } = await browserApiClient.POST('/api/v2/bots', {
    body: input,
  });
  return data;
}

export async function listBots(): Promise<BotList | undefined> {
  const { data } = await browserApiClient.GET('/api/v2/bots');
  return data;
}

export async function getBot(botId: string): Promise<Bot | undefined> {
  const { data } = await browserApiClient.GET('/api/v2/bots/{bot_id}', {
    params: { path: { bot_id: botId } },
  });
  return data;
}

export async function updateBot(
  botId: string,
  input: BotUpdateRequest,
): Promise<Bot | undefined> {
  const { data } = await browserApiClient.PUT('/api/v2/bots/{bot_id}', {
    params: { path: { bot_id: botId } },
    body: input,
  });
  return data;
}

export async function deleteBot(botId: string): Promise<void> {
  await browserApiClient.DELETE('/api/v2/bots/{bot_id}', {
    params: { path: { bot_id: botId } },
  });
}

export async function listBotChats(
  botId: string,
  options?: { page?: number; pageSize?: number },
): Promise<ChatList | undefined> {
  const { data } = await browserApiClient.GET('/api/v2/bots/{bot_id}/chats', {
    params: {
      path: { bot_id: botId },
      query: {
        page: options?.page ?? 1,
        page_size: options?.pageSize ?? 50,
      },
    },
  });
  return data;
}

export async function getBotChat(
  botId: string,
  chatId: string,
): Promise<ChatDetails | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/bots/{bot_id}/chats/{chat_id}',
    {
      params: { path: { bot_id: botId, chat_id: chatId } },
    },
  );
  return data;
}

export async function createBotChat(botId: string): Promise<Chat | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/bots/{bot_id}/chats',
    {
      params: { path: { bot_id: botId } },
    },
  );
  return data;
}

export async function updateBotChat(
  botId: string,
  chatId: string,
  input: ChatUpdate,
): Promise<Chat | undefined> {
  const { data } = await browserApiClient.PUT(
    '/api/v2/bots/{bot_id}/chats/{chat_id}',
    {
      params: { path: { bot_id: botId, chat_id: chatId } },
      body: input,
    },
  );
  return data;
}

export async function deleteBotChat(
  botId: string,
  chatId: string,
): Promise<void> {
  await browserApiClient.DELETE('/api/v2/bots/{bot_id}/chats/{chat_id}', {
    params: { path: { bot_id: botId, chat_id: chatId } },
  });
}

export async function generateBotChatTitle(
  botId: string,
  chatId: string,
  input: TitleGenerateRequest,
): Promise<TitleGenerateResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/bots/{bot_id}/chats/{chat_id}/title',
    {
      params: { path: { bot_id: botId, chat_id: chatId } },
      body: input,
    },
  );
  return data;
}
