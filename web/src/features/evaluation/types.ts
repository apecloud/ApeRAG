import type { components } from '@/api-v2/schema';

export type DatasetVersionStatus =
  components['schemas']['BenchmarkDatasetVersionStatus'];

export type EvaluationRunStatus = components['schemas']['EvaluationRunStatus'];

export type EvaluationRunItemStatus =
  components['schemas']['EvaluationRunItemStatus'];

export type EvaluationAttemptStatus =
  components['schemas']['EvaluationRunItemAttemptStatus'];

export type BenchmarkDatasetSourceType =
  components['schemas']['BenchmarkDatasetSourceType'];

export type BenchmarkDataset =
  components['schemas']['BenchmarkDatasetEnvelope'];

export type BenchmarkDatasetCreate =
  components['schemas']['BenchmarkDatasetCreate'];

export type BenchmarkDatasetVersion =
  components['schemas']['BenchmarkDatasetVersionEnvelope'];

export type BenchmarkDatasetVersionCreate =
  components['schemas']['BenchmarkDatasetVersionCreate'];

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
