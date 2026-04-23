import type { components } from '@/api-v2/schema';

export type ProviderModelApi = 'completion' | 'embedding' | 'rerank';
export type Provider = {
  name: string;
  user_id: string;
  label: string;
  completion_dialect: string;
  embedding_dialect: string;
  rerank_dialect: string;
  allow_custom_base_url: boolean;
  base_url: string;
  extra?: string | null;
  api_key?: string | null;
  created?: string | null;
  updated?: string | null;
};
export type ProviderModel = {
  provider_name: string;
  api: ProviderModelApi;
  model: string;
  custom_llm_provider: string;
  context_window?: number | null;
  max_input_tokens?: number | null;
  max_output_tokens?: number | null;
  tags?: string[] | null;
  created?: string | null;
  updated?: string | null;
};
export type ModelConfig = components['schemas']['ModelConfig'];
export type ModelSpec = components['schemas']['ModelSpec'];
export type DefaultModelConfig = components['schemas']['DefaultModelConfig'];
export type DefaultModelScenario = DefaultModelConfig['scenario'];
export type ProviderModelFormInput = {
  provider_name: string;
  api: ProviderModelApi;
  model: string;
  custom_llm_provider: string;
  context_window?: number | null;
  max_input_tokens?: number | null;
  max_output_tokens?: number | null;
  tags: string[];
};
export type ProviderModelUpdateInput = {
  custom_llm_provider?: string | null;
  context_window?: number | null;
  max_input_tokens?: number | null;
  max_output_tokens?: number | null;
  tags?: string[] | null;
};

export type ProviderFormInput = {
  label: string;
  base_url: string;
  completion_dialect: string;
  embedding_dialect: string;
  rerank_dialect: string;
  allow_custom_base_url?: boolean;
  api_key?: string;
  extra?: string;
  status?: 'enable' | 'disable';
};

export type ProviderCatalogViewModel = {
  providers: Provider[];
  models: ProviderModel[];
};

export type ScenarioModelGroup = {
  label?: string | null;
  name?: string | null;
  models?: ModelSpec[] | null;
};

export type ScenarioModelsViewModel = Record<
  DefaultModelScenario,
  ScenarioModelGroup[]
>;
