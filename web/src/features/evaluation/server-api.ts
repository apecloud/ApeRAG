import { createServerApiClient } from '@/lib/api/typed/server';

import type {
  EvaluationDataset,
  EvaluationDatasetItem,
  EvaluationPagination,
  EvaluationRun,
  EvaluationRunDetailResponse,
  EvaluationRunItem,
  EvaluationRunItemAttempt,
  FetchState,
  ListState,
} from './types';

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : 'Network request failed';

const fetchState = async <T>(
  loader: () => Promise<T>,
): Promise<FetchState<T>> => {
  try {
    return {
      payload: await loader(),
      unavailable: false,
    };
  } catch (error) {
    return {
      payload: null,
      unavailable: false,
      error: errorMessage(error),
    };
  }
};

const pageFrom = (
  page?: EvaluationPagination | null,
): EvaluationPagination | undefined => page ?? undefined;

export async function listEvaluationDatasets(
  collectionId: string,
): Promise<ListState<EvaluationDataset>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET('/api/v2/evaluation-datasets', {
      params: {
        query: { collection_id: collectionId },
      },
    });
    return data;
  });

  return {
    ...result,
    items: result.payload?.items ?? [],
    page: pageFrom(result.payload?.pagination),
  };
}

export async function getEvaluationDataset(
  datasetId: string,
): Promise<FetchState<EvaluationDataset>> {
  const client = await createServerApiClient();
  return fetchState(async () => {
    const { data } = await client.GET(
      '/api/v2/evaluation-datasets/{dataset_id}',
      {
        params: {
          path: { dataset_id: datasetId },
        },
      },
    );
    return data as EvaluationDataset;
  });
}

export async function listEvaluationDatasetItems(
  datasetId: string,
): Promise<ListState<EvaluationDatasetItem>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET(
      '/api/v2/evaluation-datasets/{dataset_id}/items',
      {
        params: {
          path: { dataset_id: datasetId },
        },
      },
    );
    return data;
  });

  return {
    ...result,
    items: result.payload?.items ?? [],
    page: pageFrom(result.payload?.pagination),
  };
}

export async function listEvaluationRuns(
  filter: { collectionId?: string; botId?: string; datasetId?: string },
): Promise<ListState<EvaluationRun>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET('/api/v2/evaluation-runs', {
      params: {
        query: {
          collection_id: filter.collectionId,
          bot_id: filter.botId,
          dataset_id: filter.datasetId,
        },
      },
    });
    return data;
  });

  return {
    ...result,
    items: result.payload?.items ?? [],
    page: pageFrom(result.payload?.pagination),
  };
}

export async function getEvaluationRunDetail(
  runId: string,
): Promise<FetchState<EvaluationRunDetailResponse>> {
  const client = await createServerApiClient();
  return fetchState(async () => {
    const { data } = await client.GET('/api/v2/evaluation-runs/{run_id}', {
      params: {
        path: { run_id: runId },
      },
    });
    return data as EvaluationRunDetailResponse;
  });
}

export async function listEvaluationRunItems(
  runId: string,
): Promise<ListState<EvaluationRunItem>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET(
      '/api/v2/evaluation-runs/{run_id}/items',
      {
        params: {
          path: { run_id: runId },
        },
      },
    );
    return data;
  });

  return {
    ...result,
    items: result.payload?.items ?? [],
    page: pageFrom(result.payload?.pagination),
  };
}

export async function listEvaluationRunItemAttempts(
  runId: string,
  itemId: string,
): Promise<ListState<EvaluationRunItemAttempt>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET(
      '/api/v2/evaluation-runs/{run_id}/items/{item_id}/attempts',
      {
        params: {
          path: { run_id: runId, item_id: itemId },
        },
      },
    );
    return data;
  });

  return {
    ...result,
    items: result.payload?.items ?? [],
    page: undefined,
  };
}
