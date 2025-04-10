export type TypesModels = {
  label: string;
  value: string;
  enabled: boolean;
  endpoint: string;
  context_window?: number;
  prompt_template?: string;
  temperature?: number;
  memory_prompt_template: string;
  similarity_topk:number;
  similarity_score_threshold:number;
  family_name:string;
};

export type TypesCollectionConfigSource =
  | 'system'
  | 'local'
  | 's3'
  | 'oss'
  | 'ftp'
  | 'url'
  | 'github'
  | 'email'
  | 'feishu';

export type TypesCollectionConfig = {
  source?: TypesCollectionConfigSource;

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
  pop_server?: string;
  port?: string;
  email_address?: string;
  email_password?: string;

  // feishu
  app_id?: string;
  app_secret?: string;
  space_id?: string;
};

export type TypesCollectionConfigSourceOption = {
  label: string;
  value: TypesCollectionConfigSource;
  icon?: string;
  icon_on?: string;
};

export type TypesCollectionStatus = 'INACTIVE' | 'ACTIVE' | 'DELETED';

export type TypesCollectionType = 'document' | 'database' | 'code';

export type TypesCollection = {
  id?: string;
  title?: string;
  description?: string;
  bot_ids?: number[];
  system?: boolean;
  status?: TypesCollectionStatus;
  type: TypesCollectionType;
  config?: TypesCollectionConfig;
  created?: string;
  updated?: string;
  sensitive_protect?: boolean;
  sensitive_protect_method: string;
};

export type TypesEmailSource = 'gmail' | 'outlook' | 'qqmail' | 'others';

export type TypesCollectionSyncHistory = {
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
