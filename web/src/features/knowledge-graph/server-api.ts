import { createServerApiClient } from '@/lib/api/typed/server';

import type {
  GraphLabelsResponse,
  KnowledgeGraph,
  MergeSuggestionsResponse,
} from './types';

export async function getGraphLabels(
  collectionId: string,
): Promise<GraphLabelsResponse> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/graphs/labels',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  // Lesson 9a: empty labels list is a legitimate not-yet-indexed
  // answer, not a 404 or a null response.
  return data ?? { labels: [] };
}

export async function getKnowledgeGraph(
  collectionId: string,
  options?: { label?: string; maxNodes?: number; maxDepth?: number },
): Promise<KnowledgeGraph | null> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
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
    },
  );
  return data ?? null;
}

export async function getMergeSuggestions(
  collectionId: string,
): Promise<MergeSuggestionsResponse> {
  const client = await createServerApiClient();
  const { data } = await client.GET(
    '/api/v2/collections/{collection_id}/graphs/merge-suggestions',
    {
      params: { path: { collection_id: collectionId } },
    },
  );
  // Lesson 9a: typed empty shape when nothing persisted yet.
  return (
    (data as MergeSuggestionsResponse | undefined) ?? {
      suggestions: [],
      total_analyzed_nodes: 0,
      processing_time_seconds: 0,
      from_cache: false,
      generated_at: '',
      total_suggestions: 0,
      pending_count: 0,
      accepted_count: 0,
      rejected_count: 0,
    }
  );
}
