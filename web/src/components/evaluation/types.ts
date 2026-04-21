export type DatasetVersionStatus = 'draft' | 'published' | 'archived';

export type EvaluationRunStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type EvaluationRunItemStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type EvaluationAttemptStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type BenchmarkDatasetSourceType =
  | 'manual'
  | 'import'
  | 'migrated_from_question_set'
  | string;

export interface BenchmarkDatasetVersion {
  id?: string;
  dataset_id?: string;
  version?: string;
  version_name?: string;
  status?: DatasetVersionStatus;
  case_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface BenchmarkDataset {
  id?: string;
  collection_id?: string;
  name?: string;
  description?: string;
  source_type?: BenchmarkDatasetSourceType;
  latest_version?: BenchmarkDatasetVersion;
  version_count?: number;
  case_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface EvaluationRunConfig {
  model?: string;
  judge_mode?: 'none' | 'exact_match' | 'llm_as_judge' | string;
  concurrency?: number;
  max_attempts?: number;
}

export interface EvaluationRunSummary {
  total?: number;
  pending?: number;
  running?: number;
  completed?: number;
  failed?: number;
  cancelled?: number;
  avg_score?: number;
}

export interface EvaluationRunProgress {
  percent?: number;
  eta_ms?: number;
}

export interface EvaluationRun {
  id?: string;
  bot_id?: string;
  dataset_version_id?: string;
  status?: EvaluationRunStatus;
  config?: EvaluationRunConfig;
  summary?: EvaluationRunSummary;
  progress?: EvaluationRunProgress;
  created_at?: string;
  updated_at?: string;
  started_at?: string;
  finished_at?: string;
}

export interface EvaluationRunItemAttempt {
  id?: string;
  run_item_id?: string;
  attempt_no?: number;
  status?: EvaluationAttemptStatus;
  agent_turn_id?: string;
  agent_chat_id?: string;
  score?: number;
  error?: string;
  started_at?: string;
  finished_at?: string;
}

export interface EvaluationRunItem {
  id?: string;
  run_id?: string;
  case_id?: string;
  case_key?: string;
  status?: EvaluationRunItemStatus;
  best_score?: number;
  latest_attempt?: EvaluationRunItemAttempt;
  error?: string;
  created_at?: string;
  updated_at?: string;
}

export interface EvaluationRunDetailResponse {
  run?: EvaluationRun;
  summary?: EvaluationRunSummary;
  progress?: EvaluationRunProgress;
}

export interface EvaluationPagination {
  total?: number;
  offset?: number;
  limit?: number;
}
