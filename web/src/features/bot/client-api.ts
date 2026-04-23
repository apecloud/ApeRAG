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

type TitleLanguage = NonNullable<TitleGenerateRequest['language']>;

const SUPPORTED_TITLE_LANGUAGES: readonly TitleLanguage[] = [
  'zh-CN',
  'en-US',
  'ja-JP',
  'ko-KR',
] as const;

/**
 * Narrow an arbitrary `next-intl` locale / UI string down to the
 * `TitleGenerateRequest['language']` enum the backend accepts. Any input
 * outside the supported set falls back to `null`, which triggers the backend
 * default (`zh-CN` per v2 schema).
 */
export function toTitleLanguage(
  value: string | null | undefined,
): TitleGenerateRequest['language'] {
  if (value == null) {
    return null;
  }
  return (SUPPORTED_TITLE_LANGUAGES as readonly string[]).includes(value)
    ? (value as TitleLanguage)
    : null;
}

export type TitleGenerateInput = {
  language?: string | null;
  max_length?: number | null;
  turns?: number | null;
};

/**
 * Build a fully-typed `TitleGenerateRequest` body from a caller-friendly
 * input. Each required field is always set (defaulting to `null` so the
 * backend applies its own default) and `language` is normalised to the
 * generated schema's enum union.
 */
export function buildTitleGenerateRequest(
  input: TitleGenerateInput = {},
): TitleGenerateRequest {
  return {
    max_length: input.max_length ?? null,
    language: toTitleLanguage(input.language),
    turns: input.turns ?? null,
  };
}

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
  input: TitleGenerateInput = {},
): Promise<TitleGenerateResponse | undefined> {
  const body = buildTitleGenerateRequest(input);
  const { data } = await browserApiClient.POST(
    '/api/v2/bots/{bot_id}/chats/{chat_id}/title',
    {
      params: { path: { bot_id: botId, chat_id: chatId } },
      body,
    },
  );
  return data;
}
