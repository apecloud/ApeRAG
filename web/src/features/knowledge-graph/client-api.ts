'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  GraphLabelsResponse,
  KnowledgeGraph,
  MergeSuggestionsResponse,
  SuggestionActionRequest,
  SuggestionActionResponse,
} from './types';

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
    '/api/v1/marketplace/collections/{collection_id}/graph',
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

export async function mergeGraphNodes(
  collectionId: string,
  input: { entity_ids: string[]; target_entity_id?: string | null },
): Promise<Record<string, unknown> | undefined> {
  const { data } = await browserApiClient.POST(
    '/api/v2/collections/{collection_id}/graphs/nodes/merge',
    {
      params: { path: { collection_id: collectionId } },
      body: input as never,
    },
  );
  return data as Record<string, unknown> | undefined;
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
  return data as MergeSuggestionsResponse | undefined;
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
  return data as MergeSuggestionsResponse | undefined;
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
