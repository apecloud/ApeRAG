'use client';

import { browserApiClient } from '@/lib/api/typed/browser';
import type { User } from '@/features/identity/types';

import type { Login, OAuthAuthorizeResponse, OAuthProvider, Register } from './types';

export async function login(input: Login): Promise<User> {
  const { data } = await browserApiClient.POST('/api/v1/login', {
    body: input,
  });
  if (!data) {
    throw new Error('login: empty response body');
  }
  return data;
}

export async function register(input: Register): Promise<User> {
  const { data } = await browserApiClient.POST('/api/v1/register', {
    body: input,
  });
  if (!data) {
    throw new Error('register: empty response body');
  }
  return data;
}

export async function logout(): Promise<void> {
  await browserApiClient.POST('/api/v1/logout', {});
}

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || '';

// Raw fetch: `/api/v1/auth/{provider}/authorize` returns a 302 redirect to the
// external OAuth provider; the redirect URL itself is not a typed API data
// contract (it's an upstream-controlled URL), so there is no value in wrapping
// it through the typed openapi-fetch client. Kept inside the adapter boundary
// per Phase 1 design-lock (msg=5f0a370b decision 2b / OAuth authorize raw fetch).
export async function oauthAuthorize(
  provider: OAuthProvider,
): Promise<OAuthAuthorizeResponse> {
  const response = await fetch(
    `${BASE_PATH}/api/v1/auth/${provider}/authorize`,
  );
  const data = (await response.json()) as OAuthAuthorizeResponse;
  return data;
}
