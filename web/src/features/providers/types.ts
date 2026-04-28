export type ModelCapability = 'chat' | 'completion' | 'embedding' | 'rerank';

export type ModelProvider = {
  id?: string | null;
  provider_type: string;
  display_name: string;
  description?: string | null;
  supported_capabilities: ModelCapability[];
  default_base_url?: string | null;
  enabled?: boolean;
};

export type ModelAccount = {
  id?: string | null;
  user_id?: string | null;
  provider_type: string;
  name: string;
  display_name: string;
  base_url: string;
  status: 'ACTIVE' | 'INACTIVE';
  validation_error?: string | null;
};

export type Model = {
  id?: string | null;
  account_id: string;
  provider_model_id: string;
  display_name: string;
  capability: ModelCapability;
  runner_type: string;
  context_window?: number | null;
  max_input_tokens?: number | null;
  max_output_tokens?: number | null;
  embedding_dimensions?: number | null;
  supports_vision?: boolean;
  supports_tool_calling?: boolean;
  status?: 'ACTIVE' | 'INACTIVE';
};

export type ModelUseScenario =
  | 'agent_chat'
  | 'collection_completion'
  | 'collection_embedding'
  | 'retrieval_rerank'
  | 'background_task';

export type ModelUse = {
  id?: string | null;
  scenario: ModelUseScenario;
  capability: ModelCapability;
  primary_model_id?: string | null;
  fallback_model_ids?: string[];
  enabled: boolean;
};

export type ModelAccountCreateInput = {
  provider_type: string;
  name: string;
  display_name: string;
  base_url: string;
  api_key: string;
};

export type ModelCreateInput = {
  account_id: string;
  provider_model_id: string;
  display_name: string;
  capability: ModelCapability;
  runner_type?: string | null;
  context_window?: number | null;
  max_input_tokens?: number | null;
  max_output_tokens?: number | null;
  embedding_dimensions?: number | null;
  supports_vision?: boolean;
  supports_tool_calling?: boolean;
};

export type ModelPlatformViewModel = {
  providers: ModelProvider[];
  accounts: ModelAccount[];
  models: Model[];
  uses: ModelUse[];
};

export type ModelSpec = {
  model_id?: string | null;
  /** UI label from model platform; form values stay `model_id`. */
  display_name?: string | null;
  temperature?: number | null;
  max_tokens?: number | null;
  max_completion_tokens?: number | null;
  timeout?: number | null;
  top_n?: number | null;
};
