'use client';

import type { paths } from '@/api-v2/schema';
import {
  ApiClientError,
  apiErrorCode,
  apiErrorMessage,
  type ApiErrorBody,
} from '@/lib/api/typed/errors';
import createClient, { type Middleware } from 'openapi-fetch';
import { toast } from 'sonner';

const apiBaseUrl = process.env.NEXT_PUBLIC_BASE_PATH || '';

const errorMiddleware: Middleware = {
  async onResponse({ response }) {
    if (response.ok || response.status === 204) {
      return undefined;
    }

    const body = (await response
      .clone()
      .json()
      .catch(() => null)) as ApiErrorBody | null;
    const message = apiErrorMessage(body, response.status);

    if (message) {
      toast.error(message);
    }

    throw new ApiClientError(
      response.status,
      apiErrorCode(body),
      message,
      body,
    );
  },
};

export function createBrowserApiClient() {
  const client = createClient<paths>({
    baseUrl: apiBaseUrl,
    credentials: 'include',
  });
  client.use(errorMiddleware);
  return client;
}

export const browserApiClient = createBrowserApiClient();
