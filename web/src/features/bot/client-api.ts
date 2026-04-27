'use client';

import { browserApiClient } from '@/lib/api/typed/browser';
import type { Document } from '@/features/document/types';

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

// Runtime-safe defaults for the `TitleGenerateRequest` body. The backend
// Pydantic schema advertises defaults (20 / "zh-CN" / 1), but
// `aperag/views/bots_v2.py` forwards the parsed values straight into
// `chat_title_service.generate_title()` where `max(1, turns)` /
// `min(max_length, 50)` would raise a TypeError on `null`. The FE adapter
// therefore must send concrete values rather than relying on backend
// fallback.
const DEFAULT_TITLE_MAX_LENGTH = 20;
const DEFAULT_TITLE_TURNS = 1;
const DEFAULT_TITLE_LANGUAGE: TitleLanguage = 'zh-CN';

/**
 * Narrow an arbitrary `next-intl` locale / UI string down to the
 * `TitleGenerateRequest['language']` enum the backend accepts. Any input
 * outside the supported set falls back to the product default language
 * (`zh-CN`) instead of `null`, because the backend service would forward a
 * `null` straight into `generate_title()` and crash.
 */
export function toTitleLanguage(
  value: string | null | undefined,
): TitleLanguage {
  if (value == null) {
    return DEFAULT_TITLE_LANGUAGE;
  }
  return (SUPPORTED_TITLE_LANGUAGES as readonly string[]).includes(value)
    ? (value as TitleLanguage)
    : DEFAULT_TITLE_LANGUAGE;
}

export type TitleGenerateInput = {
  language?: string | null;
  max_length?: number | null;
  turns?: number | null;
};

/**
 * Build a fully-typed `TitleGenerateRequest` body from a caller-friendly
 * input. Each required field is always filled with a runtime-safe concrete
 * default (`max_length: 20`, `turns: 1`, `language: "zh-CN"`) instead of
 * `null`, because the backend does not apply Pydantic defaults when the
 * FastAPI route has already parsed an explicit value.
 */
export function buildTitleGenerateRequest(
  input: TitleGenerateInput = {},
): TitleGenerateRequest {
  return {
    max_length: input.max_length ?? DEFAULT_TITLE_MAX_LENGTH,
    language: toTitleLanguage(input.language),
    turns: input.turns ?? DEFAULT_TITLE_TURNS,
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

// Chat document attachment (for chat-input.tsx). Thin typed wrap of
// `/api/v1/chats/{chat_id}/documents` (POST multipart upload +
// GET {document_id} status poll). Lives in bot feature because chats are
// owned by the bot domain; no separate features/chat/* surface per
// design-lock msg=5f0a370b decision 2d.

export async function uploadChatDocument(
  chatId: string,
  file: File,
): Promise<Document> {
  const { data } = await browserApiClient.POST(
    '/api/v1/chats/{chat_id}/documents',
    {
      params: { path: { chat_id: chatId } },
      body: { file: file as unknown as string },
      bodySerializer: (body) => {
        const fd = new FormData();
        fd.append('file', (body as unknown as { file: File }).file);
        return fd;
      },
    },
  );
  if (!data) {
    throw new Error('uploadChatDocument: empty response body');
  }
  return data;
}

export async function getChatDocument(
  chatId: string,
  documentId: string,
): Promise<Document> {
  const { data } = await browserApiClient.GET(
    '/api/v1/chats/{chat_id}/documents/{document_id}',
    {
      params: { path: { chat_id: chatId, document_id: documentId } },
    },
  );
  if (!data) {
    throw new Error('getChatDocument: empty response body');
  }
  return data;
}
