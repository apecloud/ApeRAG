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

export type CollectionConfig = {
  source?: CollectionConfigSource;

  embedding_model_name?: string;
  embedding_model_service_provider?: string;
  enable_light_rag?: boolean;

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

export type CollectionSyncHistory = {
  id: string;
  total_documents: number;
  status: 'RUNNING' | 'CANCELED' | 'COMPLETED';
  total_documents_to_sync: number;
  new_documents: number;
  deleted_documents: number;
  processing_documents: number;
  modified_documents: number;
  failed_documents: number;
  successful_documents: number;
  start_time: string;
  execution_time: string;
};
