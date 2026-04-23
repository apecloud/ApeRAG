import type { components } from '@/api-v2/schema';

export type PromptsPayload = components['schemas']['PromptsPayload'];

export type UpdateUserPromptsRequest =
  components['schemas']['UpdateUserPromptsRequest'];

// The current `/api/v1/prompts/user` GET response is typed as
// `{ [key: string]: unknown }` in the generated schema, and the
// `/api/v1/prompts/user/{prompt_type}` DELETE path param is a plain string.
// Phase 1b does not change the OpenAPI shape (the v1→v2 hard-cut is the
// job of the later `model_platform` API phase). These domain-boundary
// types mirror the current runtime shape so the api-key-style adapter
// pattern (cast once at the adapter boundary, typed from there) can
// still hold. Track under the `model_platform` breaking-changes table:
// upgrade `GET /prompts/user` to a concrete `UserPromptsResponse`
// component and enum `prompt_type` path param.
export type PromptType = keyof PromptsPayload;

export interface PromptDetail {
  content?: string;
  source?: 'user' | 'system' | 'hardcoded' | string;
  customized?: boolean;
}

export type UserPromptsResponse = Partial<Record<PromptType, PromptDetail>>;
