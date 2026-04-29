'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  GraphEmbeddingMapResponse,
  GraphHybridResponse,
  GraphLabelsResponse,
  GraphSearchEntity,
  KnowledgeGraph,
  MergeSuggestionItem,
  MergeSuggestionsResponse,
  SuggestionActionRequest,
  SuggestionActionResponse,
} from './types';

const createEmptyMergeSuggestionsResponse = (): MergeSuggestionsResponse => ({
  suggestions: [],
  total_analyzed_nodes: 0,
  processing_time_seconds: 0,
  from_cache: false,
  generated_at: '',
  total_suggestions: 0,
  pending_count: 0,
  accepted_count: 0,
  rejected_count: 0,
});

const countSuggestionsByStatus = (
  suggestions: MergeSuggestionItem[],
  status: MergeSuggestionItem['status'],
) => suggestions.filter((suggestion) => suggestion.status === status).length;

const normalizeMergeSuggestionsResponse = (
  data: unknown,
): MergeSuggestionsResponse | undefined => {
  if (!data || typeof data !== 'object') {
    return undefined;
  }

  const raw = data as Partial<MergeSuggestionsResponse> &
    Record<string, unknown>;
  const suggestions = Array.isArray(raw.suggestions)
    ? (raw.suggestions as MergeSuggestionItem[])
    : [];

  return {
    ...createEmptyMergeSuggestionsResponse(),
    ...raw,
    suggestions,
    total_suggestions:
      typeof raw.total_suggestions === 'number'
        ? raw.total_suggestions
        : suggestions.length,
    pending_count:
      typeof raw.pending_count === 'number'
        ? raw.pending_count
        : countSuggestionsByStatus(suggestions, 'PENDING'),
    accepted_count:
      typeof raw.accepted_count === 'number'
        ? raw.accepted_count
        : countSuggestionsByStatus(suggestions, 'ACCEPTED'),
    rejected_count:
      typeof raw.rejected_count === 'number'
        ? raw.rejected_count
        : countSuggestionsByStatus(suggestions, 'REJECTED'),
  };
};

export async function getGraphLabels(
  collectionId: string,
): Promise<GraphLabelsResponse | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/graphs/labels',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return data;
}

export async function getKnowledgeGraph(
  collectionId: string,
  options?: {
    label?: string;
    maxNodes?: number;
    maxDepth?: number;
    signal?: AbortSignal;
  },
): Promise<KnowledgeGraph | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/graphs',
    {
      params: {
        path: { collection_id: collectionId },
        query: {
          label: options?.label ?? '*',
          max_nodes: options?.maxNodes ?? 1000,
          max_depth: options?.maxDepth ?? 3,
        },
      },
      signal: options?.signal,
    },
  );
  return data;
}

export async function getGraphEmbeddingMap(
  collectionId: string,
  maxEntities = 1000,
): Promise<GraphEmbeddingMapResponse | null> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/graphs/embedding-map',
    {
      params: {
        path: { collection_id: collectionId },
        query: { max_entities: maxEntities },
      },
    },
  );
  return data ?? null;
}

export async function getGraphHybrid(
  collectionId: string,
  maxEntities = 1000,
): Promise<GraphHybridResponse | null> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/graphs/hybrid',
    {
      params: {
        path: { collection_id: collectionId },
        query: { max_entities: maxEntities },
      },
    },
  );
  return data ?? null;
}

export async function searchGraphEntities(
  collectionId: string,
  q: string,
  topK = 8,
): Promise<GraphSearchEntity[]> {
  if (!q.trim()) return [];
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/graphs/entities/search',
    {
      params: {
        path: { collection_id: collectionId },
        query: { q, top_k: topK },
      },
    },
  );
  return data?.entities ?? [];
}

export async function getMarketplaceKnowledgeGraph(
  collectionId: string,
  options?: {
    label?: string;
    maxNodes?: number;
    maxDepth?: number;
    signal?: AbortSignal;
  },
): Promise<KnowledgeGraph | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/marketplace/collections/{collection_id}/graph',
    {
      params: {
        path: { collection_id: collectionId },
        query: {
          label: options?.label,
          max_nodes: options?.maxNodes,
          max_depth: options?.maxDepth,
        },
      },
      signal: options?.signal,
    },
  );
  return data;
}

export async function getMarketplaceGraphEmbeddingMap(
  collectionId: string,
  maxEntities = 1000,
): Promise<GraphEmbeddingMapResponse | null> {
  const { data } = await browserApiClient.GET(
    '/api/v2/marketplace/collections/{collection_id}/graph/embedding-map',
    {
      params: {
        path: { collection_id: collectionId },
        query: { max_entities: maxEntities },
      },
    },
  );
  return data ?? null;
}

export async function getMarketplaceGraphHybrid(
  collectionId: string,
  maxEntities = 1000,
): Promise<GraphHybridResponse | null> {
  const { data } = await browserApiClient.GET(
    '/api/v2/marketplace/collections/{collection_id}/graph/hybrid',
    {
      params: {
        path: { collection_id: collectionId },
        query: { max_entities: maxEntities },
      },
    },
  );
  return data ?? null;
}

export async function searchMarketplaceGraphEntities(
  collectionId: string,
  q: string,
  topK = 8,
): Promise<GraphSearchEntity[]> {
  if (!q.trim()) return [];
  const { data } = await browserApiClient.GET(
    '/api/v2/marketplace/collections/{collection_id}/graph/entities/search',
    {
      params: {
        path: { collection_id: collectionId },
        query: { q, top_k: topK },
      },
    },
  );
  return data?.entities ?? [];
}

export async function getMergeSuggestions(
  collectionId: string,
): Promise<MergeSuggestionsResponse | undefined> {
  const { data } = await browserApiClient.GET(
    '/api/v2/collections/{collection_id}/graphs/merge-suggestions',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  return normalizeMergeSuggestionsResponse(data);
}

export async function runMergeSuggestions(
  collectionId: string,
): Promise<MergeSuggestionsResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/graphs/merge-suggestions',
    {
      params: { path: { collection_id: collectionId } },
      body: {},
    },
  );
  return normalizeMergeSuggestionsResponse(data);
}

export async function handleSuggestionAction(
  collectionId: string,
  suggestionId: string,
  input: SuggestionActionRequest,
): Promise<SuggestionActionResponse | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/graphs/merge-suggestions/{suggestion_id}/action',
    {
      params: {
        path: {
          collection_id: collectionId,
          suggestion_id: suggestionId,
        },
      },
      body: input,
    },
  );
  return data as SuggestionActionResponse | undefined;
}
