import type { components } from '@/api-v2/schema';

export type Bot = components['schemas']['Bot'];
export type BotList = components['schemas']['BotList'];
export type BotCreate = components['schemas']['BotCreate'];
export type BotUpdateRequest = components['schemas']['BotUpdateRequest'];

export type Chat = components['schemas']['Chat'];
export type ChatList = components['schemas']['ChatList'];
export type ChatDetails = components['schemas']['ChatDetails'];
export type ChatUpdate = components['schemas']['ChatUpdate'];
export type ChatMessage = components['schemas']['ChatMessage'];

export type Feedback = components['schemas']['Feedback'];
export type FeedbackTag = NonNullable<Feedback['tag']>;
export type FeedbackType = NonNullable<Feedback['type']>;

export const FEEDBACK_TAGS = [
  'Harmful',
  'Unsafe',
  'Fake',
  'Unhelpful',
  'Other',
] as const satisfies readonly FeedbackTag[];

export type Reference = components['schemas']['Reference'];

export type TitleGenerateRequest =
  components['schemas']['TitleGenerateRequest'];
export type TitleGenerateResponse =
  components['schemas']['TitleGenerateResponse'];
export type TitleLanguage = NonNullable<TitleGenerateRequest['language']>;
