import type { components } from '@/api-v2/schema';

export type EvaluationRunStatus = components['schemas']['EvaluationRunStatus'];

export type EvaluationRunItemStatus =
  components['schemas']['EvaluationRunItemStatus'];

export type EvaluationAttemptStatus =
  components['schemas']['EvaluationRunItemAttemptStatus'];

export type EvaluationDataset =
  components['schemas']['EvaluationDatasetEnvelope'];

export type EvaluationDatasetCreate =
  components['schemas']['EvaluationDatasetCreate'];

export type EvaluationDatasetUpdate =
  components['schemas']['EvaluationDatasetUpdate'];

export type EvaluationDatasetItem =
  components['schemas']['EvaluationDatasetItemEnvelope'];

export type EvaluationDatasetItemCreate =
  components['schemas']['EvaluationDatasetItemCreate'];

export type EvaluationDatasetItemUpdate =
  components['schemas']['EvaluationDatasetItemUpdate'];

export type EvaluationDatasetItemsAppendRequest =
  components['schemas']['EvaluationDatasetItemsAppendRequest'];

/**
 * AI auto-generate preview surface (PR #1838 BE + #1839 FE).
 * Re-export here so domain consumers go through `@/features/evaluation/types`
 * per the typed-API-boundary contract; raw `@/api-v2/schema` access is
 * restricted to typed adapters in `client-api.ts` / `server-api.ts`.
 */
export type EvaluationDatasetItemDraft = NonNullable<
  components['schemas']['GeneratedDatasetItem']
>;

export type EvaluationDatasetGeneratePreviewRequest = NonNullable<
  components['schemas']['EvaluationDatasetGeneratePreviewRequest']
>;

export type EvaluationDatasetGeneratePreviewResponse = NonNullable<
  components['schemas']['EvaluationDatasetGeneratePreviewResponse']
>;

export type EvaluationRun = components['schemas']['EvaluationRunEnvelope'];

export type EvaluationRunCreate = components['schemas']['EvaluationRunCreate'];

export type EvaluationRunDetailResponse =
  components['schemas']['EvaluationRunDetailResponse'];

export type EvaluationRunItem =
  components['schemas']['EvaluationRunItemEnvelope'];

export type EvaluationRunItemAttempt =
  components['schemas']['EvaluationRunItemAttemptEnvelope'];

export type EvaluationRunSummary =
  components['schemas']['EvaluationRunSummary'];

export type EvaluationRunProgress =
  components['schemas']['EvaluationRunProgress'];

export type EvaluationPagination =
  components['schemas']['EvaluationPagination'];

export type FetchState<T> = {
  payload: T | null;
  unavailable: boolean;
  error?: string;
};

export type ListState<T> = FetchState<unknown> & {
  items: T[];
  page?: EvaluationPagination;
};
