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

export type MergeSuggestionsRequest = NonNullable<
  components['schemas']['MergeSuggestionsRequest']
>;
export type MergeSuggestionsResponse = NonNullable<
  components['schemas']['MergeSuggestionsResponse']
>;
export type MergeSuggestionsRunResponse = NonNullable<
  components['schemas']['MergeSuggestionsRunResponse']
>;
export type GraphCurationRunSummary = NonNullable<
  components['schemas']['GraphCurationRunSummary']
>;
export type GraphMergeSuggestionItem = NonNullable<
  components['schemas']['GraphMergeSuggestionItem']
>;
export type GraphMergeSuggestionEntity = NonNullable<
  components['schemas']['GraphMergeSuggestionEntity']
>;

export type SuggestionActionRequest = NonNullable<
  components['schemas']['SuggestionActionRequest']
>;
export type SuggestionActionResponse = NonNullable<
  components['schemas']['SuggestionActionResponse']
>;
export type SuggestionActionMergeResult = NonNullable<
  components['schemas']['SuggestionActionMergeResult']
>;

export type SuggestionAction = NonNullable<SuggestionActionRequest['action']>;

export const SUGGESTION_ACTIONS = [
  'accept',
  'reject',
] as const satisfies readonly SuggestionAction[];
