import type { components } from '@/api-v2/schema';

export type Login = components['schemas']['Login'];
export type Register = components['schemas']['Register'];

export type OAuthProvider = 'google' | 'github';

export type OAuthAuthorizeResponse = {
  /** Backend returns a redirect URL under `authorization_url`. */
  authorization_url?: string | null;
};
