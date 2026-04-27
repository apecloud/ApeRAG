import type { components } from '@/api-v2/schema';

export type Provider = components['schemas']['LlmProvider'];
export type ProviderModel = components['schemas']['LlmProviderModel'];
export type ProviderModelApi = ProviderModel['api'];
export type ModelConfig = components['schemas']['ModelConfig'];
export type ModelSpec = components['schemas']['ModelSpec'];
export type DefaultModelConfig = components['schemas']['DefaultModelConfig'];
export type DefaultModelScenario = DefaultModelConfig['scenario'];
export type ProviderModelFormInput =
  components['schemas']['LlmProviderModelCreateRequest'];
export type ProviderModelUpdateInput =
  components['schemas']['LlmProviderModelUpdate'];

export type ProviderFormInput = Omit<
  components['schemas']['LlmProviderCreateWithApiKey'],
  'allow_custom_base_url'
> & {
  allow_custom_base_url?: boolean | null;
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
