import { createServerApiClient } from '@/lib/api/typed/server';

import type {
  BenchmarkDataset,
  EvaluationPagination,
  EvaluationRun,
  EvaluationRunDetailResponse,
  EvaluationRunItem,
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

export async function listBenchmarkDatasets(
  collectionId: string,
): Promise<ListState<BenchmarkDataset>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET('/api/v2/benchmark-datasets', {
      params: {
        query: {
          collection_id: collectionId,
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

export async function listEvaluationRuns(
  botId: string,
): Promise<ListState<EvaluationRun>> {
  const client = await createServerApiClient();
  const result = await fetchState(async () => {
    const { data } = await client.GET('/api/v2/evaluation-runs', {
      params: {
        query: {
          bot_id: botId,
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
        path: {
          run_id: runId,
        },
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
          path: {
            run_id: runId,
          },
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
