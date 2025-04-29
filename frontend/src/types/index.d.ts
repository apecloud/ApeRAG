import { Bot, Collection, Document } from '@/api';

export * from './flow';

export type Merge<M, N> = Omit<M, Extract<keyof M, keyof N>> & N;

/**
 * bots
 */
export type BotConfig = {
  model?: string;
  model_service_provider?: string;
  model_name?: string;
  charactor?: string;
  llm?: {
    similarity_score_threshold?: number;
    similarity_topk?: number;
    context_window?: number;
    prompt_template?: string;
    temperature?: number;
    memory_prompt_template: string;
  };
  memory: boolean;
  use_related_question: boolean;
  feishu?: {
    encrypt_key?: string;
  };
};

export type ApeBot = Merge<Bot, { config?: BotConfig }>;

/**
 * collections
 */
export type CollectionConfigSource =
  | 'system'
  | 'local'
  | 's3'
  | 'oss'
  | 'ftp'
  | 'url'
  | 'git'
  | 'email'
  | 'feishu';

export type CollectionType = 'document' | 'database' | 'code';
export type CollectionEmailSource = 'gmail' | 'outlook' | 'qqmail' | 'others';
// export type CollectionSyncHistory = {
//   id: string;
//   total_documents: number;
//   status: "RUNNING" | "CANCELED" | "COMPLETED";
//   total_documents_to_sync: number;
//   new_documents: number;
//   deleted_documents: number;
//   processing_documents: number;
//   modified_documents: number;
//   failed_documents: number;
//   successful_documents: number;
//   start_time: string;
//   execution_time: string;
// };
export type CollectionConfig = {
  source?: CollectionConfigSource;

  embedding_model_name?: string;
  embedding_model_service_provider?: string;
  enable_lightrag?: boolean;

  sensitive_protect?: boolean;
  sensitive_protect_method?: 'nostore' | 'replace';

  crontab?: {
    enabled: boolean;
    minute: string;
    hour: string;
    day_of_month: string;
    month: string;
    day_of_week: string;
  };

  // local and ftp
  path?: string;

  // ftp
  host?: string;
  username?: string;
  password?: string;

  // s3 | oss
  region?: string;
  access_key_id?: string;
  secret_access_key?: string;
  bucket?: string;
  dir?: string;

  // email
  email_source?: CollectionEmailSource;
  pop_server?: string;
  port?: string;
  email_address?: string;
  email_password?: string;

  // feishu
  app_id?: string;
  app_secret?: string;
  space_id?: string;
};

export type ApeCollection = Merge<Collection, { config?: CollectionConfig }>;

/**
 * documents
 */
export type ApeDocumentConfig = {
  path?: string;
  labels?: {
    [key in string]: string;
  }[];
};

export type ApeDocument = Merge<Document, { config?: ApeDocumentConfig }>;
