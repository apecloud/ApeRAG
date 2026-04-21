'use client';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

export const extractEvaluationErrorMessage = (payload: unknown) => {
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

export const postEvaluationAction = async <T>(
  path: string,
  body?: unknown,
): Promise<T> => {
  const response = await fetch(`${basePath}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const payload = await response.json().catch(() => undefined);

  if (!response.ok) {
    throw new Error(
      extractEvaluationErrorMessage(payload) ||
        `Request failed with status ${response.status}`,
    );
  }

  return payload as T;
};
