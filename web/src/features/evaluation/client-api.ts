'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  EvaluationDatasetCreate,
  EvaluationDatasetGeneratePreviewRequest,
  EvaluationDatasetGeneratePreviewResponse,
  EvaluationDatasetItemCreate,
  EvaluationRunCreate,
} from './types';

// Re-export the canonical names so existing callers that imported via
// `@/features/evaluation/client-api` keep working. Domain consumers
// should still prefer importing from `@/features/evaluation/types`.
export type {
  EvaluationDatasetGeneratePreviewRequest as GenerateEvaluationDatasetItemsPreviewRequest,
  EvaluationDatasetGeneratePreviewResponse as GenerateEvaluationDatasetItemsPreviewResponse,
  EvaluationDatasetItemDraft,
} from './types';

/**
 * Generate AI draft questions from collection content. The endpoint
 * does not write to the dataset — it only returns drafts; the caller
 * lets the user prune/edit and then bulk-creates via
 * `appendEvaluationDatasetItems`.
 */
export async function generateEvaluationDatasetItemsPreview(
  datasetId: string,
  body: EvaluationDatasetGeneratePreviewRequest,
): Promise<EvaluationDatasetGeneratePreviewResponse> {
  const { data } = await browserApiClient.POST(
    '/api/v2/evaluation-datasets/{dataset_id}/items/generate-preview',
    {
      params: {
        path: { dataset_id: datasetId },
      },
      body,
    },
  );
  if (!data) {
    throw new Error('Generate preview returned an empty response');
  }
  return data;
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
