'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type {
  BenchmarkDatasetCreate,
  BenchmarkDatasetVersionCreate,
  EvaluationRunCreate,
} from './types';

export async function createBenchmarkDataset(input: BenchmarkDatasetCreate) {
  const { data } = await browserApiClient.POST('/api/v2/benchmark-datasets', {
    body: input,
  });
  return data;
}

export async function createBenchmarkDatasetVersion(
  datasetId: string,
  input: BenchmarkDatasetVersionCreate,
) {
  const { data } = await browserApiClient.POST(
    '/api/v2/benchmark-datasets/{dataset_id}/versions',
    {
      params: {
        path: {
          dataset_id: datasetId,
        },
      },
      body: input,
    },
  );
  return data;
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
        path: {
          run_id: runId,
        },
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
        path: {
          run_id: runId,
          item_id: itemId,
        },
      },
    },
  );
  return data;
}
