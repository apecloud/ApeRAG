'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  EvaluationDatasetCreate,
  EvaluationDatasetItemCreate,
  EvaluationRunCreate,
} from './types';

/**
 * Draft item returned by the AI auto-generate preview endpoint
 * (`POST .../{dataset_id}/items/generate-preview`). The BE writes
 * `reference_context` from the source chunk text so Phase 3
 * LLM-as-judge has the ground-truth retrieval context ready without
 * the user having to type it. Shape is intentionally aligned with
 * `EvaluationDatasetItemCreate` so the bulk-create endpoint can
 * consume the same items unchanged.
 */
export type EvaluationDatasetItemDraft = {
  question: string;
  expected_answer: string;
  reference_context: string;
};

export type GenerateEvaluationDatasetItemsPreviewRequest = {
  collection_id: string;
  count?: number;
  language?: string;
  prompt_template?: string;
};

export type GenerateEvaluationDatasetItemsPreviewResponse = {
  items: EvaluationDatasetItemDraft[];
  requested_count?: number;
  delivered_count?: number;
  language?: string;
};

/**
 * Generate AI draft questions from collection content. The endpoint
 * does not write to the dataset — it only returns drafts; the caller
 * lets the user prune/edit and then bulk-creates via
 * `appendEvaluationDatasetItems`.
 *
 * Uses raw fetch (not the typed browserApiClient) because the BE
 * endpoint is currently being added in a parallel PR; this wrapper
 * lets the FE land the UI in parallel and switches to the typed
 * client once the OpenAPI schema regenerates.
 */
export async function generateEvaluationDatasetItemsPreview(
  datasetId: string,
  body: GenerateEvaluationDatasetItemsPreviewRequest,
): Promise<GenerateEvaluationDatasetItemsPreviewResponse> {
  const url = `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/v2/evaluation-datasets/${encodeURIComponent(
    datasetId,
  )}/items/generate-preview`;
  const resp = await fetch(url, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '');
    throw new Error(
      `Generate preview failed: ${resp.status} ${resp.statusText}${
        detail ? ` — ${detail}` : ''
      }`,
    );
  }
  return (await resp.json()) as GenerateEvaluationDatasetItemsPreviewResponse;
}

export async function createEvaluationDataset(input: EvaluationDatasetCreate) {
  const { data } = await browserApiClient.POST('/api/v2/evaluation-datasets', {
    body: input,
  });
  return data;
}

export async function deleteEvaluationDataset(datasetId: string) {
  await browserApiClient.DELETE('/api/v2/evaluation-datasets/{dataset_id}', {
    params: {
      path: { dataset_id: datasetId },
    },
  });
}

export async function appendEvaluationDatasetItems(
  datasetId: string,
  items: EvaluationDatasetItemCreate[],
) {
  const { data } = await browserApiClient.POST(
    '/api/v2/evaluation-datasets/{dataset_id}/items',
    {
      params: {
        path: { dataset_id: datasetId },
      },
      body: { items },
    },
  );
  return data;
}

export async function deleteEvaluationDatasetItem(
  datasetId: string,
  itemId: string,
) {
  await browserApiClient.DELETE(
    '/api/v2/evaluation-datasets/{dataset_id}/items/{item_id}',
    {
      params: {
        path: { dataset_id: datasetId, item_id: itemId },
      },
    },
  );
}

export async function createEvaluationRun(input: EvaluationRunCreate) {
  const { data } = await browserApiClient.POST('/api/v2/evaluation-runs', {
    body: input,
  });
  return data;
}

export async function cancelEvaluationRun(runId: string) {
  const { data } = await browserApiClient.POST(
    '/api/v2/evaluation-runs/{run_id}/cancel',
    {
      params: {
        path: { run_id: runId },
      },
    },
  );
  return data;
}

export async function retryEvaluationRunItem(runId: string, itemId: string) {
  const { data } = await browserApiClient.POST(
    '/api/v2/evaluation-runs/{run_id}/items/{item_id}/retry',
    {
      params: {
        path: { run_id: runId, item_id: itemId },
      },
    },
  );
  return data;
}
