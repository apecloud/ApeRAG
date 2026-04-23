'use client';

import { browserApiClient } from '@/lib/api/typed/browser';

import type { PromptType, UpdateUserPromptsRequest } from './types';

export async function updateUserPrompts(input: UpdateUserPromptsRequest) {
  const { data } = await browserApiClient.PUT('/api/v1/prompts/user', {
    body: input,
  });
  return data;
}

export async function deleteUserPrompt(promptType: PromptType) {
  await browserApiClient.DELETE('/api/v1/prompts/user/{prompt_type}', {
    params: {
      path: { prompt_type: promptType },
    },
  });
}
