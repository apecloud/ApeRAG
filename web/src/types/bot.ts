import { TypesCollection } from './collection';

export type TypesBotConfig = {
  model?: string;
  llm?: {
    endpoint?: string;
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

export type TypesBot = {
  id?: string;
  type: string;
  title?: string;
  description?: string;
  status?: string;
  collections?: TypesCollection[];
  collection_ids?: string[];
  created?: string;
  updated?: string;
  config?: TypesBotConfig;
};
