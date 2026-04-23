'use server';

import type { paths } from '@/api-v2/schema';
import { getLocale } from '@/services/cookies';
import { cookies } from 'next/headers';
import createClient from 'openapi-fetch';

function mergeHeaders(
  initHeaders: HeadersInit | undefined,
  injected: Record<string, string>,
) {
  const headers = new Headers(initHeaders);
  for (const [key, value] of Object.entries(injected)) {
    if (value) {
      headers.set(key, value);
    }
  }
  return headers;
}

async function serverFetch(request: Request, cookieHeader: string, lang: string) {
  const response = await fetch(
    new Request(request, {
      cache: request.cache || 'no-store',
      headers: mergeHeaders(request.headers, {
        Cookie: cookieHeader,
        Lang: lang,
      }),
    }),
  );

  if (!response.ok) {
    throw new Error(
      `Server API request failed: ${response.status} ${response.statusText}`,
    );
  }

  return response;
}

export async function createServerApiClient() {
  const lang = await getLocale();
  const cookieHeader = (await cookies())
    .getAll()
    .map((item) => `${item.name}=${item.value}`)
    .join('; ');

  return createClient<paths>({
    baseUrl: process.env.API_SERVER_ENDPOINT || 'http://localhost:8000',
    fetch: (request: Request) => serverFetch(request, cookieHeader, lang),
  });
}
