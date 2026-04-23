import type { components } from '@/api-v2/schema';

export type Bot = components['schemas']['Bot'];
export type BotList = components['schemas']['BotList'];
export type BotCreate = components['schemas']['BotCreate'];
export type BotUpdateRequest = components['schemas']['BotUpdateRequest'];

export type Chat = components['schemas']['Chat'];
export type ChatList = components['schemas']['ChatList'];
export type ChatDetails = components['schemas']['ChatDetails'];
export type ChatUpdate = components['schemas']['ChatUpdate'];

export type TitleGenerateRequest =
  components['schemas']['TitleGenerateRequest'];
export type TitleGenerateResponse =
  components['schemas']['TitleGenerateResponse'];
