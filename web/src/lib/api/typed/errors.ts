export type ApiErrorBody = {
  detail?: string | { message?: string };
  error?: { code?: string; message?: string; status?: number };
  message?: string;
};

export class ApiClientError extends Error {
  status: number;
  code: string;
  body: unknown;

  constructor(status: number, code: string, message: string, body: unknown) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.code = code;
    this.body = body;
  }
}

export function apiErrorMessage(body: ApiErrorBody | null, status: number) {
  if (typeof body?.detail === 'string') {
    return body.detail;
  }
  if (typeof body?.detail?.message === 'string') {
    return body.detail.message;
  }
  if (typeof body?.error?.message === 'string') {
    return body.error.message;
  }
  if (typeof body?.message === 'string') {
    return body.message;
  }
  return `Request failed: ${status}`;
}

export function apiErrorCode(body: ApiErrorBody | null) {
  return body?.error?.code ?? 'unknown';
}
