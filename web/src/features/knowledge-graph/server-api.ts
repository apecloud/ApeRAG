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
  return (data as MergeSuggestionsResponse | undefined) ?? {
    run: null,
    suggestions: [],
  };
}
