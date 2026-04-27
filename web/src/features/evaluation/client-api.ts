'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  EvaluationDatasetCreate,
  EvaluationDatasetItemCreate,
  EvaluationDatasetItemUpdate,
  EvaluationDatasetUpdate,
  EvaluationRunCreate,
} from './types';

export async function createEvaluationDataset(input: EvaluationDatasetCreate) {
  const { data } = await browserApiClient.POST('/api/v2/evaluation-datasets', {
    body: input,
  });
  return data;
}

export async function updateEvaluationDataset(
  datasetId: string,
  input: EvaluationDatasetUpdate,
) {
  const { data } = await browserApiClient.PUT(
    '/api/v2/evaluation-datasets/{dataset_id}',
    {
      params: {
        path: { dataset_id: datasetId },
      },
      body: input,
    },
  );
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

export async function updateEvaluationDatasetItem(
  datasetId: string,
  itemId: string,
  input: EvaluationDatasetItemUpdate,
) {
  const { data } = await browserApiClient.PUT(
    '/api/v2/evaluation-datasets/{dataset_id}/items/{item_id}',
    {
      params: {
        path: { dataset_id: datasetId, item_id: itemId },
      },
      body: input,
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
