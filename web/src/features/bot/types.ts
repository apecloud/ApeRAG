import type { components } from '@/api-v2/schema';

export type Bot = components['schemas']['Bot'];
export type BotList = components['schemas']['BotList'];
export type BotCreate = components['schemas']['BotCreate'];
export type BotUpdateRequest = components['schemas']['BotUpdateRequest'];

export type Chat = components['schemas']['Chat'];
export type ChatList = components['schemas']['ChatList'];
export type ChatDetails = components['schemas']['ChatDetails'];
export type ChatUpdate = components['schemas']['ChatUpdate'];

// FE-local message-part shape. The backend ``ChatMessage`` schema was
// removed in Wave 8 D8.5 (history now ships ``AgentTurnSnapshot[]``);
// renderers (``message-parts-user`` / ``message-parts-ai`` /
// ``message-timestamp`` / ``chat-messages``) still use this hand-rolled
// shape for user-typed input parts and the renderer-side part list.
// Fields cover observed renderer access; additional fields fall through
// the index signature to permit drift absorption (TODO W9-3: migrate
// renderer to typed ``UIMessagePart`` envelope).
export type ChatMessage = {
  id?: string | null;
  type?: string | null;
  role?: string | null;
  data?: string | null;
  timestamp?: number | null;
  references?: Reference[] | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

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

// FE-local citation/reference shape consumed by ``message-reference`` /
// ``agent-turn-renderer``. Backend ``Reference`` schema was refactored
// into ``DataCitationPart`` / ``SourceDocumentPart`` / ``SourceUrlPart``
// in Wave 8 D8 — renderer still consumes the legacy envelope shape
// directly. Permissive optional fields cover all observed renderer
// access (text/score/title/uri + metadata.query/type); index signature
// permits drift absorption (TODO W9-3: migrate renderer to typed
// ``Citation``/``SourceDocument``/``SourceUrl`` union).
export type Reference = {
  text?: string | null;
  score?: number | null;
  title?: string | null;
  uri?: string | null;
  metadata?: {
    query?: string | null;
    type?: string | null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
  } | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
};

export type TitleGenerateRequest =
  components['schemas']['TitleGenerateRequest'];
export type TitleGenerateResponse =
  components['schemas']['TitleGenerateResponse'];
export type TitleLanguage = NonNullable<TitleGenerateRequest['language']>;
