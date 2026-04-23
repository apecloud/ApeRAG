import { cookies } from 'next/headers';

import { getLocale } from '@/services/cookies';
import { createServerApiClient } from '@/lib/api/typed/server';
import type { User } from '@/features/identity/types';

// `/api/v1/user` is the authenticated "me" endpoint. Used by root / workspace
// / admin layouts to resolve the current user and redirect to signin when
// unauthenticated. `createServerApiClient` throws on non-2xx, so we swallow
// the error and return `null` to preserve the layout's redirect-on-unauth
// control flow.
export async function getCurrentUser(): Promise<User | null> {
  const client = await createServerApiClient();
  try {
    const { data } = await client.GET('/api/v1/user', {});
    return data ?? null;
  } catch {
    return null;
  }
}

// Raw fetch: `/api/v1/config` is hidden from public OpenAPI per Phase 4
// governance (see `aperag/openapi_spec.py::HIDDEN_FROM_PUBLIC_PATH_PREFIXES`
// — the same exception as audit). Phase 4 will promote to typed contract.
// Used by `auth/signin/page.tsx` to fetch the server-side login_methods
// config (local / github / google).

const API_SERVER_ENDPOINT =
  process.env.API_SERVER_ENDPOINT || 'http://localhost:8000';

type AuthConfig = {
  login_methods?: string[];
};

export async function getAuthConfig(): Promise<AuthConfig> {
  const lang = await getLocale();
  const cookieHeader = (await cookies())
    .getAll()
    .map((item) => `${item.name}=${item.value}`)
    .join('; ');

  try {
    const response = await fetch(`${API_SERVER_ENDPOINT}/api/v1/config`, {
      cache: 'no-store',
      headers: {
        Cookie: cookieHeader,
        Lang: lang,
      },
    });
    if (!response.ok) {
      return { login_methods: ['local'] };
    }
    return (await response.json()) as AuthConfig;
  } catch {
    return { login_methods: ['local'] };
  }
}
