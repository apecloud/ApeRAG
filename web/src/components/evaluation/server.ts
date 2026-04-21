import 'server-only';

import { cookies } from 'next/headers';

import { getLocale } from '@/services/cookies';

import type {
  BenchmarkDataset,
  EvaluationPagination,
  EvaluationRun,
  EvaluationRunDetailResponse,
  EvaluationRunItem,
} from './types';

type FetchState<T> = {
  payload: T | null;
  unavailable: boolean;
  error?: string;
};

type ListState<T> = FetchState<unknown> & {
  items: T[];
  page?: EvaluationPagination;
};

const API_SERVER_ORIGIN =
  process.env.API_SERVER_ENDPOINT || 'http://localhost:8000';
const API_SERVER_BASE_PATH = process.env.API_SERVER_BASE_PATH || '/api/v1';
const API_ROOT_BASE_PATH = API_SERVER_BASE_PATH.replace(
  /\/api\/v1\/?$/,
  '',
).replace(/\/$/, '');
const API_V2_BASE_URL = `${API_SERVER_ORIGIN}${API_ROOT_BASE_PATH}/api/v2`;

const buildUrl = (
  path: string,
  query?: Record<string, string | number | undefined>,
) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const url = new URL(`${API_V2_BASE_URL}${normalizedPath}`);

  Object.entries(query || {}).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    url.searchParams.set(key, String(value));
  });

  return url.toString();
};

const extractErrorMessage = (payload: unknown) => {
  if (
    payload &&
    typeof payload === 'object' &&
    'detail' in payload &&
    typeof payload.detail === 'string'
  ) {
    return payload.detail;
  }

  if (
    payload &&
    typeof payload === 'object' &&
    'message' in payload &&
    typeof payload.message === 'string'
  ) {
    return payload.message;
  }

  return undefined;
};

const getCookieHeader = async () => {
  const cookieStore = await cookies();
  return cookieStore
    .getAll()
    .map((cookie) => `${cookie.name}=${cookie.value}`)
    .join('; ');
};

const fetchEvaluationV2 = async <T>(
  path: string,
  query?: Record<string, string | number | undefined>,
): Promise<FetchState<T>> => {
  const lang = await getLocale();
  const cookieHeader = await getCookieHeader();

  try {
    const response = await fetch(buildUrl(path, query), {
      method: 'GET',
      cache: 'no-store',
      headers: {
        Lang: lang,
        Cookie: cookieHeader,
      },
    });

    if (
      response.status === 404 ||
      response.status === 405 ||
      response.status === 501
    ) {
      return {
        payload: null,
        unavailable: true,
      };
    }

    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json')
      ? await response.json()
      : null;

    if (!response.ok) {
      return {
        payload: null,
        unavailable: false,
        error:
          extractErrorMessage(payload) ||
          `Request failed with status ${response.status}`,
      };
    }

    return {
      payload: payload as T,
      unavailable: false,
    };
  } catch (error) {
    return {
      payload: null,
      unavailable: true,
      error: error instanceof Error ? error.message : 'Network request failed',
    };
  }
};

const extractItems = <T>(payload: unknown, fallbackKeys: string[] = []) => {
  if (Array.isArray(payload)) {
    return payload as T[];
  }

  if (!payload || typeof payload !== 'object') {
    return [];
  }

  const candidateKeys = ['items', ...fallbackKeys];
  for (const key of candidateKeys) {
    const candidate = (payload as Record<string, unknown>)[key];
    if (Array.isArray(candidate)) {
      return candidate as T[];
    }
  }

  return [];
};

const extractPage = (payload: unknown): EvaluationPagination | undefined => {
  if (!payload || typeof payload !== 'object') return undefined;
  const candidate =
    (payload as Record<string, unknown>).page ||
    (payload as Record<string, unknown>).pagination;

  if (!candidate || typeof candidate !== 'object') return undefined;
  return candidate as EvaluationPagination;
};

export const listBenchmarkDatasets = async (
  collectionId: string,
): Promise<ListState<BenchmarkDataset>> => {
  const result = await fetchEvaluationV2<unknown>('/benchmark-datasets', {
    collection_id: collectionId,
  });

  return {
    ...result,
    items: extractItems<BenchmarkDataset>(result.payload, ['datasets']),
    page: extractPage(result.payload),
  };
};

export const listEvaluationRuns = async (
  botId: string,
): Promise<ListState<EvaluationRun>> => {
  const result = await fetchEvaluationV2<unknown>('/evaluation-runs', {
    bot_id: botId,
  });

  return {
    ...result,
    items: extractItems<EvaluationRun>(result.payload, ['runs']),
    page: extractPage(result.payload),
  };
};

export const getEvaluationRunDetail = async (
  runId: string,
): Promise<FetchState<EvaluationRunDetailResponse>> => {
  return fetchEvaluationV2<EvaluationRunDetailResponse>(
    `/evaluation-runs/${runId}`,
  );
};

export const listEvaluationRunItems = async (
  runId: string,
): Promise<ListState<EvaluationRunItem>> => {
  const result = await fetchEvaluationV2<unknown>(
    `/evaluation-runs/${runId}/items`,
  );

  return {
    ...result,
    items: extractItems<EvaluationRunItem>(result.payload, ['run_items']),
    page: extractPage(result.payload),
  };
};
