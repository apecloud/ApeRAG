import type { components } from '@/api-v2/schema';

export type PromptsPayload = components['schemas']['PromptsPayload'];

export type UpdateUserPromptsRequest =
  components['schemas']['UpdateUserPromptsRequest'];

// The current `/api/v2/prompts/user` GET response is typed as
// `{ [key: string]: unknown }` in the generated schema, and the
// `/api/v2/prompts/user/{prompt_type}` DELETE path param is a plain string.
// Phase 8 #49 G3 carved the route into the `model_platform` domain and
// hard-cut `/api/v1/prompts*` → `/api/v2/prompts*` per D7 canonical;
// the OpenAPI shape itself was not touched. These domain-boundary
// types mirror the current runtime shape so the api-key-style adapter
// pattern (cast once at the adapter boundary, typed from there) can
// still hold. Tracked under the `model_platform` breaking-changes
// table: upgrade `GET /prompts/user` to a concrete
// `UserPromptsResponse` component and enum `prompt_type` path param.
export type PromptType = keyof PromptsPayload;

export interface PromptDetail {
  content?: string;
  source?: 'user' | 'system' | 'hardcoded' | string;
  customized?: boolean;
}

export type UserPromptsResponse = Partial<Record<PromptType, PromptDetail>>;
