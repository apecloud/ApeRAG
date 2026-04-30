import type { components } from '@/api-v2/schema';

export type GraphLabelsResponse = NonNullable<
  components['schemas']['GraphLabelsResponse']
>;
export type KnowledgeGraph = NonNullable<
  components['schemas']['KnowledgeGraph']
>;
export type GraphNode = NonNullable<components['schemas']['GraphNode']>;
export type GraphEdge = NonNullable<components['schemas']['GraphEdge']>;
export type GraphNodeProperties = NonNullable<
  components['schemas']['GraphNodeProperties']
>;
export type GraphEdgeProperties = NonNullable<
  components['schemas']['GraphEdgeProperties']
>;
export type GraphSearchEntity = NonNullable<
  components['schemas']['GraphSearchEntity']
>;
export type GraphEvidenceResponse = NonNullable<
  components['schemas']['GraphEvidenceResponse']
>;
export type GraphEvidenceRef = NonNullable<
  components['schemas']['GraphEvidenceRef']
>;
export type GraphRelationEvidenceRequest = NonNullable<
  components['schemas']['GraphRelationEvidenceRequest']
>;
export type GraphEmbeddingMapResponse = NonNullable<
  components['schemas']['GraphEmbeddingMapResponse']
>;
export type GraphHybridResponse = NonNullable<
  components['schemas']['GraphHybridResponse']
>;
export type GraphHybridNode = NonNullable<
  components['schemas']['GraphHybridNode']
>;
export type GraphEmbeddingPoint = NonNullable<
  components['schemas']['GraphEmbeddingPoint']
>;

export type MergeSuggestionsRequest = NonNullable<
  components['schemas']['MergeSuggestionsRequest']
>;

// The OpenAPI schema currently models this as just `{ action }`, but the
// backend accepts an optional `target_entity_data` override when the
// caller wants to change the resolved name/type before an "accept" run.
// Mirror that here so callers don't need to cast.
type RawSuggestionActionRequest = NonNullable<
  components['schemas']['SuggestionActionRequest']
>;

export type SuggestionActionRequest = RawSuggestionActionRequest & {
  target_entity_data?: MergeSuggestionTargetEntity;
};

export type SuggestionAction = NonNullable<
  RawSuggestionActionRequest['action']
>;

export const SUGGESTION_ACTIONS = [
  'accept',
  'reject',
] as const satisfies readonly SuggestionAction[];

// The merge-suggestions endpoints return `{[key: string]: unknown}` in the
// OpenAPI schema today — the canonical response shape is maintained on the
// Python side. The FE keeps a typed mirror of that shape so route callers
// can consume it without falling back to the legacy `@/api` SDK. Keep this
// list in lockstep with `aperag/schema/view_models.py::MergeSuggestions*`.
export type MergeSuggestionStatus =
  | 'PENDING'
  | 'APPLY_PENDING'
  | 'APPLYING'
  | 'APPLIED'
  | 'APPLY_FAILED'
  | 'ACCEPTED'
  | 'REJECTED'
  | 'DISMISSED'
  | 'EXPIRED'
  | 'SUPERSEDED';

export interface MergeSuggestionTargetEntity {
  entity_name: string;
  entity_type: string;
}

export interface MergeSuggestionItem {
  id: string;
  collection_id: string;
  suggestion_batch_id: string;
  entity_ids: string[];
  confidence_score: number;
  merge_reason: string;
  reason?: string;
  evidence?: Record<string, unknown> | null;
  evidence_refs?: GraphEvidenceRef[];
  suggested_target_entity: MergeSuggestionTargetEntity;
  target_entity_id?: string;
  status: MergeSuggestionStatus;
  created: string;
  updated?: string;
  resolution_note?: string | null;
  operated_at?: string;
}

export interface MergeSuggestionsResponse {
  suggestions: MergeSuggestionItem[];
  total_analyzed_nodes: number;
  processing_time_seconds: number;
  from_cache: boolean;
  generated_at: string;
  total_suggestions: number;
  pending_count: number;
  accepted_count: number;
  rejected_count: number;
  // Some legacy paths also surface a `run` summary — keep it optional so
  // the typed FE can opt in without breaking historical callers.
  run?: unknown;
}

export type SuggestionActionStatus = 'success' | 'error';

export interface SuggestionActionResponse {
  status: SuggestionActionStatus;
  message: string;
  suggestion_id: string;
  action: SuggestionAction;
  merge_result?: unknown;
}
