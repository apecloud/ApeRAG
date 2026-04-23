'use client';

import type { Feedback, Reference } from '@/features/bot/types';
import { CopyToClipboard } from '@/components/copy-to-clipboard';
import { Markdown } from '@/components/markdown';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Card,
  CardContent,
} from '@/components/ui/card';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { cn } from '@/lib/utils';
import {
  AlertTriangle,
  Bot,
  BookOpen,
  Brain,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  LoaderCircle,
  PencilLine,
  Search,
} from 'lucide-react';
import { useFormatter, useTranslations } from 'next-intl';
import { useMemo } from 'react';
import { MessageCollapseContent } from './message-collapse-content';
import { MessageFeedback } from './message-feedback';

export type AgentArtifactEnvelope = {
  schema_version: string;
  artifact_id: string;
  turn_id: string;
  artifact_type: string;
  summary?: string | null;
  payload: Record<string, unknown>;
  storage_ref?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentTurnEnvelope = {
  schema_version: string;
  turn_id: string;
  chat_id: string;
  user_id: string;
  bot_id: string;
  request_id: string;
  client_idempotency_key: string;
  status: string;
  input_text: string;
  model_profile: Record<string, unknown>;
  error_code?: string | null;
  error_message?: string | null;
  answer_artifact_id?: string | null;
  reference_bundle_artifact_id?: string | null;
  timeline_cursor: number;
  started_at?: string | null;
  finished_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AgentTimelineEventEnvelope = {
  schema_version: string;
  event_id: string;
  turn_id: string;
  sequence: number;
  timestamp: string;
  type: string;
  technical_type?: string | null;
  label?: string | null;
  status?: string | null;
  actor: 'agent' | 'tool' | 'system';
  data: Record<string, unknown>;
  user_activity?: UserActivityEnvelope | null;
};

export type AgentTurnSnapshot = {
  turn: AgentTurnEnvelope;
  timeline: AgentTimelineEventEnvelope[];
  artifacts: AgentArtifactEnvelope[];
};

export type ReferenceBundleItem = {
  source_type: string;
  source_id?: string | null;
  title?: string | null;
  snippet?: string | null;
  score?: number | null;
  uri?: string | null;
  metadata?: Record<string, unknown>;
};

export type UserActivityIntent =
  | 'thinking'
  | 'searching_knowledge'
  | 'reading_source'
  | 'comparing_results'
  | 'writing_answer'
  | 'waiting'
  | 'completed'
  | 'error';

export type UserActivityContext = {
  source_name?: string | null;
  keyword?: string | null;
  count?: number | null;
  target_type?: 'knowledge_base' | 'document' | 'web' | null;
  scope_label?: string | null;
};

export type UserActivityEnvelope = {
  intent: UserActivityIntent;
  title_key: string;
  subtitle_key: string;
  detail_key?: string | null;
  context?: UserActivityContext | null;
};

type OrderedTimelineItem = {
  key: string;
  status: string;
  rawType: string;
  technicalType?: string | null;
  userActivity: UserActivityEnvelope;
  timestamp?: string | null;
  argsPreview?: string;
  resultPreview?: string;
  occurrences?: number;
};

const terminalStatuses = new Set(['COMPLETED', 'FAILED', 'CANCELLED']);
const terminalTimelineItemStatuses = new Set(['completed', 'failed']);
const knowledgeSearchTools = new Set(['list_collections', 'search_collection']);
const webSearchTools = new Set(['search_web', 'web_search']);
const readingTools = new Set(['read_document', 'web_read']);

function mapReferenceItem(item: ReferenceBundleItem): Reference {
  return {
    score: item.score ?? undefined,
    text: item.snippet || '',
    metadata: {
      ...(item.metadata || {}),
      title: item.title,
      source_type: item.source_type,
      source_id: item.source_id,
      uri: item.uri,
    },
  };
}

function extractAnswerText(
  snapshot: AgentTurnSnapshot,
  streamingAnswer: string,
) {
  const answerArtifact = snapshot.artifacts.find(
    (artifact) => artifact.artifact_type === 'answer',
  );
  const artifactText =
    typeof answerArtifact?.payload?.text === 'string'
      ? answerArtifact.payload.text
      : typeof answerArtifact?.payload?.content === 'string'
      ? answerArtifact.payload.content
      : '';
  if (artifactText) return artifactText;
  if (streamingAnswer) return streamingAnswer;
  return '';
}

function extractReferences(snapshot: AgentTurnSnapshot): Reference[] {
  const referenceArtifact = snapshot.artifacts.find(
    (artifact) => artifact.artifact_type === 'reference_bundle',
  );
  const items = Array.isArray(referenceArtifact?.payload?.items)
    ? (referenceArtifact.payload.items as ReferenceBundleItem[])
    : [];
  if (items.length > 0) {
    return items.map(mapReferenceItem);
  }
  return [];
}

function compactPreview(value: unknown, maxLength = 220) {
  if (value == null) return undefined;

  const raw =
    typeof value === 'string'
      ? value
      : (() => {
          try {
            return JSON.stringify(value, null, 2);
          } catch {
            return String(value);
          }
        })();

  const normalized = raw.trim();
  if (!normalized) return undefined;
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

function normalizeActivityText(value: unknown, maxLength = 160) {
  if (typeof value !== 'string') return undefined;
  const normalized = value.trim().replace(/\s+/g, ' ');
  if (!normalized) return undefined;
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 3).trimEnd()}...`;
}

function iterActivityPayloads(
  data: Record<string, unknown>,
): Record<string, unknown>[] {
  const payloads: Record<string, unknown>[] = [data];

  for (const key of ['args', 'result']) {
    const nested = data[key];
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      payloads.push(nested as Record<string, unknown>);
    }
  }

  return payloads;
}

function extractActivityString(
  data: Record<string, unknown>,
  keys: string[],
  maxLength = 160,
) {
  for (const payload of iterActivityPayloads(data)) {
    for (const key of keys) {
      const value = normalizeActivityText(payload[key], maxLength);
      if (value) return value;
    }
  }
  return undefined;
}

function extractActivityCount(data: Record<string, unknown>) {
  for (const payload of iterActivityPayloads(data)) {
    for (const key of ['count', 'total', 'total_count', 'result_count']) {
      const value = payload[key];
      if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
        return value;
      }
    }

    for (const key of ['items', 'results']) {
      const items = payload[key];
      if (Array.isArray(items)) {
        return items.length;
      }
    }
  }

  return undefined;
}

function inferToolName(event: AgentTimelineEventEnvelope) {
  return (
    extractActivityString(event.data, ['tool_name']) ||
    normalizeActivityText(event.label)
  );
}

function inferTargetType(
  toolName?: string,
): UserActivityContext['target_type'] | undefined {
  if (!toolName) return undefined;
  if (knowledgeSearchTools.has(toolName)) return 'knowledge_base';
  if (webSearchTools.has(toolName)) return 'web';
  if (readingTools.has(toolName)) {
    return toolName === 'read_document' ? 'document' : 'web';
  }
  return undefined;
}

function buildActivityContext(
  data: Record<string, unknown>,
  toolName?: string,
): UserActivityContext | undefined {
  const keyword = extractActivityString(data, [
    'query',
    'keyword',
    'keywords',
    'search_query',
  ]);
  const sourceName = extractActivityString(
    data,
    [
      'source_name',
      'collection_name',
      'collection_title',
      'document_title',
      'title',
      'name',
    ],
    120,
  );
  const count = extractActivityCount(data);
  const targetType = inferTargetType(toolName);
  const scopeLabel = extractActivityString(
    data,
    ['collection_id', 'document_id', 'url'],
    120,
  );

  if (
    keyword == null &&
    sourceName == null &&
    count == null &&
    targetType == null &&
    scopeLabel == null
  ) {
    return undefined;
  }

  return {
    keyword,
    source_name: sourceName,
    count,
    target_type: targetType,
    scope_label:
      scopeLabel && scopeLabel !== sourceName ? scopeLabel : undefined,
  };
}

function getActivityDetailKey(
  intent: UserActivityIntent,
  context?: UserActivityContext,
) {
  if (!context) return undefined;
  if (intent === 'searching_knowledge') {
    if (context.keyword) return 'activity.searching_knowledge.detail.keyword';
    if (context.source_name) {
      return 'activity.searching_knowledge.detail.source_name';
    }
    if (context.count != null) {
      return 'activity.searching_knowledge.detail.count';
    }
  }
  if (intent === 'reading_source' && context.source_name) {
    return 'activity.reading_source.detail.source_name';
  }
  if (intent === 'comparing_results' && context.count != null) {
    return 'activity.comparing_results.detail.count';
  }
  return undefined;
}

function createUserActivity(
  intent: UserActivityIntent,
  context?: UserActivityContext,
): UserActivityEnvelope {
  return {
    intent,
    title_key: `activity.${intent}.title`,
    subtitle_key: `activity.${intent}.subtitle`,
    detail_key: getActivityDetailKey(intent, context),
    context,
  };
}

function inferActivityIntentFromTool(toolName?: string): UserActivityIntent {
  if (!toolName) return 'waiting';
  if (knowledgeSearchTools.has(toolName) || webSearchTools.has(toolName)) {
    return 'searching_knowledge';
  }
  if (readingTools.has(toolName)) {
    return 'reading_source';
  }
  return 'waiting';
}

function inferUserActivity(
  event: AgentTimelineEventEnvelope,
): UserActivityEnvelope | undefined {
  if (event.user_activity) return event.user_activity;

  const technicalType = event.technical_type || event.type;
  const normalizedStatus = String(event.status || '').toLowerCase();
  const toolName = inferToolName(event);
  const context = buildActivityContext(event.data, toolName);

  if (technicalType === 'agent.state.changed') {
    if (normalizedStatus === 'thinking') {
      return createUserActivity('thinking');
    }
    if (normalizedStatus === 'searching') {
      return createUserActivity('searching_knowledge', context);
    }
    if (normalizedStatus === 'calling_tool') {
      return createUserActivity(inferActivityIntentFromTool(toolName), context);
    }
    if (normalizedStatus === 'reading_result') {
      return createUserActivity('comparing_results', context);
    }
    if (normalizedStatus === 'composing' || normalizedStatus === 'streaming') {
      return createUserActivity('writing_answer');
    }
    if (normalizedStatus === 'done') {
      return createUserActivity('completed');
    }
    if (normalizedStatus === 'failed' || normalizedStatus === 'error') {
      return createUserActivity('error');
    }
    return createUserActivity('waiting');
  }

  if (
    technicalType === 'tool.started' ||
    technicalType === 'external_action.started'
  ) {
    return createUserActivity(inferActivityIntentFromTool(toolName), context);
  }

  if (
    technicalType === 'tool.finished' ||
    technicalType === 'external_action.finished'
  ) {
    if (normalizedStatus === 'failed' || normalizedStatus === 'error') {
      return createUserActivity('error');
    }
    const intent = inferActivityIntentFromTool(toolName);
    return createUserActivity(
      intent === 'waiting' ? 'comparing_results' : intent,
      context,
    );
  }

  if (technicalType === 'text.delta') {
    return createUserActivity('writing_answer');
  }

  if (technicalType === 'turn.started') {
    return createUserActivity('thinking');
  }
  if (technicalType === 'turn.completed') {
    return createUserActivity('completed');
  }
  if (
    technicalType === 'turn.failed' ||
    technicalType === 'turn.cancelled'
  ) {
    return createUserActivity('error');
  }

  return createUserActivity('waiting');
}

function normalizeTimelineStatus(event: AgentTimelineEventEnvelope) {
  const technicalType = event.technical_type || event.type;
  const normalized = String(event.status || '').toLowerCase();

  if (technicalType === 'agent.state.changed') {
    if (normalized === 'failed' || normalized === 'error') return 'failed';
    if (normalized === 'done' || normalized === 'completed') return 'completed';
    return 'running';
  }

  if (technicalType === 'turn.started') return 'running';
  if (
    technicalType === 'tool.started' ||
    technicalType === 'external_action.started'
  ) {
    return 'running';
  }
  if (
    technicalType === 'tool.finished' ||
    technicalType === 'external_action.finished'
  ) {
    if (normalized === 'failed' || normalized === 'error') return 'failed';
    return 'completed';
  }
  if (technicalType === 'text.delta') return 'running';
  if (technicalType === 'turn.completed') return 'completed';
  if (
    technicalType === 'turn.failed' ||
    technicalType === 'turn.cancelled'
  ) {
    return 'failed';
  }
  return normalized || 'waiting';
}

function serializeDisplayedUserActivity(activity: UserActivityEnvelope) {
  return JSON.stringify({
    intent: activity.intent,
    title_key: activity.title_key,
    subtitle_key: activity.subtitle_key,
    context: {
      keyword: activity.context?.keyword || null,
      source_name: activity.context?.source_name || null,
      target_type: activity.context?.target_type || null,
      scope_label: activity.context?.scope_label || null,
    },
  });
}

function mergeUserActivityEnvelope(
  previous: UserActivityEnvelope,
  next: UserActivityEnvelope,
): UserActivityEnvelope {
  return {
    ...previous,
    ...next,
    detail_key: next.detail_key ?? previous.detail_key,
    context: {
      ...(previous.context || {}),
      ...(next.context || {}),
    },
  };
}

function normalizeDisplayedStepStatus(previousStatus: string, nextStatus: string) {
  if (nextStatus === 'failed') return 'failed';
  if (nextStatus === 'completed') {
    return previousStatus === 'failed' ? 'failed' : 'completed';
  }
  if (nextStatus === 'running') {
    if (terminalTimelineItemStatuses.has(previousStatus)) {
      return previousStatus;
    }
    return 'running';
  }
  if (nextStatus === 'waiting') {
    return previousStatus === 'running' ? 'running' : previousStatus || 'waiting';
  }
  return nextStatus;
}

function shouldDisplayUserActivityInTimeline(activity: UserActivityEnvelope) {
  return activity.intent !== 'waiting';
}

function findDisplayedStepIndex(
  entries: OrderedTimelineItem[],
  status: string,
  activity: UserActivityEnvelope,
) {
  const lastIndex = entries.length - 1;
  if (lastIndex < 0) return -1;

  const previous = entries[lastIndex];
  if (
    serializeDisplayedUserActivity(previous.userActivity) !==
    serializeDisplayedUserActivity(activity)
  ) {
    return -1;
  }

  if (
    terminalTimelineItemStatuses.has(previous.status) &&
    previous.status !== status
  ) {
    return -1;
  }

  if (
    terminalTimelineItemStatuses.has(previous.status) &&
    previous.status === status
  ) {
    return lastIndex;
  }

  if (
    !terminalTimelineItemStatuses.has(previous.status) &&
    ['waiting', 'running', 'completed', 'failed'].includes(status)
  ) {
    return lastIndex;
  }

  return -1;
}

function closeTimelineItems(
  entries: OrderedTimelineItem[],
  turnStatus: string,
): OrderedTimelineItem[] {
  const normalizedTurnStatus = turnStatus.toUpperCase();

  if (
    normalizedTurnStatus !== 'COMPLETED' &&
    normalizedTurnStatus !== 'FAILED' &&
    normalizedTurnStatus !== 'CANCELLED'
  ) {
    return entries;
  }

  const lastOpenIndex = entries.findLastIndex(
    (entry) =>
      shouldDisplayUserActivityInTimeline(entry.userActivity) &&
      (entry.status === 'waiting' || entry.status === 'running'),
  );

  const closedEntries = entries
    .map((entry, index) => {
      if (!shouldDisplayUserActivityInTimeline(entry.userActivity)) return null;
      if (entry.status !== 'waiting' && entry.status !== 'running') {
        return entry;
      }

      if (normalizedTurnStatus === 'COMPLETED') {
        return {
          ...entry,
          status: 'completed',
        };
      }

      return {
        ...entry,
        status: index === lastOpenIndex ? 'failed' : 'completed',
      };
    })
    .filter((entry): entry is OrderedTimelineItem => entry != null);

  return closedEntries;
}

function buildDisplayedTimelineItems(
  timeline: AgentTimelineEventEnvelope[],
  turnStatus: string,
): OrderedTimelineItem[] {
  const orderedEvents = [...timeline].sort((left, right) => {
    if (left.sequence !== right.sequence) {
      return left.sequence - right.sequence;
    }
    return new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime();
  });
  const entries: OrderedTimelineItem[] = [];

  for (const event of orderedEvents) {
    const userActivity = inferUserActivity(event);
    if (!userActivity) continue;
    if (!shouldDisplayUserActivityInTimeline(userActivity)) continue;

    const status = normalizeTimelineStatus(event);
    const argsPreview = compactPreview(event.data.args, 180);
    const resultPreview = compactPreview(event.data.result, 280);
    const mergeableIndex = findDisplayedStepIndex(
      entries,
      status,
      userActivity,
    );

    if (mergeableIndex >= 0) {
      const previous = entries[mergeableIndex];
      entries[mergeableIndex] = {
        ...previous,
        userActivity: mergeUserActivityEnvelope(
          previous.userActivity,
          userActivity,
        ),
        status: normalizeDisplayedStepStatus(previous.status, status),
        timestamp: event.timestamp,
        occurrences: (previous.occurrences || 1) + 1,
        argsPreview: previous.argsPreview || argsPreview,
        resultPreview: resultPreview || previous.resultPreview,
        rawType: event.type,
        technicalType: event.technical_type || event.type,
      };
      continue;
    }

    entries.push({
      key: `${event.sequence}-${event.type}`,
      status,
      rawType: event.type,
      technicalType: event.technical_type || event.type,
      userActivity,
      timestamp: event.timestamp,
      argsPreview,
      resultPreview,
      occurrences: 1,
    });
  }

  return closeTimelineItems(entries, turnStatus);
}

function getAnswerSectionTitle(status: string, hasAnswerText: boolean) {
  if (status === 'FAILED') {
    return hasAnswerText
      ? 'page_chat.answer_section.failure_details'
      : 'page_chat.answer_section.run_failed';
  }
  if (status === 'CANCELLED') {
    return hasAnswerText
      ? 'page_chat.answer_section.cancelled_output'
      : 'page_chat.answer_section.run_cancelled';
  }
  if (status === 'COMPLETED') {
    return 'page_chat.answer_section.final_answer';
  }
  return hasAnswerText
    ? 'page_chat.answer_section.draft_answer'
    : 'page_chat.answer_section.answer';
}

function describeEmptyAnswerState(status: string) {
  if (status === 'FAILED') {
    return 'page_chat.answer_section.failed_empty';
  }
  if (status === 'CANCELLED') {
    return 'page_chat.answer_section.cancelled_empty';
  }
  return 'page_chat.answer_section.pending_empty';
}

function getTimelineItemStyles(item: OrderedTimelineItem) {
  if (item.userActivity.intent === 'error' || item.status === 'failed') {
    return {
      iconWrapper: 'border-destructive/25 bg-destructive/10 text-destructive',
      card: 'border-destructive/20 bg-destructive/5 shadow-sm',
      badge:
        'border-destructive/20 bg-destructive/10 text-destructive',
    };
  }

  if (item.userActivity.intent === 'completed' || item.status === 'completed') {
    return {
      iconWrapper:
        'border-emerald-500/15 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400',
      card: 'border-border/50 bg-background/70 opacity-85',
      badge:
        'border-emerald-500/15 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400',
    };
  }

  if (item.status === 'running') {
    return {
      iconWrapper:
        'border-emerald-500/25 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
      card:
        'border-emerald-500/20 bg-emerald-500/8 shadow-sm shadow-emerald-500/10 ring-1 ring-emerald-500/10',
      badge:
        'border-emerald-500/20 bg-emerald-500/12 text-emerald-700 dark:text-emerald-300',
    };
  }

  return {
    iconWrapper: 'border-border/70 bg-background text-foreground',
    card: 'border-border/60 bg-background/70',
    badge: 'border-border/60 bg-background text-muted-foreground',
  };
}

function getTimelineItemIcon(item: OrderedTimelineItem) {
  switch (item.userActivity.intent) {
    case 'thinking':
      return BrainCircuit;
    case 'searching_knowledge':
      return Search;
    case 'reading_source':
      return BookOpen;
    case 'comparing_results':
      return Brain;
    case 'writing_answer':
      return PencilLine;
    case 'completed':
      return CheckCircle2;
    case 'error':
      return AlertTriangle;
    case 'waiting':
    default:
      return Clock3;
  }
}

function getActivityTranslationValues(
  context?: UserActivityContext | null,
  targetTypeLabel?: string,
) {
  if (!context) return undefined;
  return {
    sourceName: context.source_name || undefined,
    keyword: context.keyword || undefined,
    count: context.count ?? undefined,
    targetType: targetTypeLabel || undefined,
    scopeLabel: context.scope_label || undefined,
  };
}

function getStatusTone(status: string): 'default' | 'secondary' | 'destructive' {
  if (status === 'COMPLETED') return 'default';
  if (status === 'FAILED' || status === 'CANCELLED') return 'destructive';
  return 'secondary';
}

export const AgentTurnCard = ({
  snapshot,
  pending,
  streamingAnswer,
  feedback,
  onFeedback,
}: {
  snapshot: AgentTurnSnapshot;
  pending: boolean;
  streamingAnswer: string;
  feedback?: Feedback;
  onFeedback: (turnId: string, feedback: Feedback) => void;
}) => {
  const t = useTranslations();
  const pageChat = useTranslations('page_chat');
  const format = useFormatter();

  const translateActivityText = (
    key: string | null | undefined,
    values?: Record<string, string | number | undefined>,
  ) => {
    if (!key) return undefined;
    const message = t(
      key as never,
      values as never,
    );
    return message === key ? undefined : message;
  };

  const translatePageChat = (
    key: string,
    values?: Record<string, string | number | undefined>,
  ) => {
    const message = pageChat(
      key as never,
      values as never,
    );
    return message === key || message === `page_chat.${key}`
      ? undefined
      : message;
  };

  const getItemStatusLabel = (status: string) => {
    return (
      translatePageChat(`activity_stream.item_status.${status}`) ||
      {
        waiting: 'Waiting',
        running: 'In progress',
        completed: 'Completed',
        failed: 'Issue',
      }[status] ||
      'Waiting'
    );
  };

  const getTurnStatusLabel = (statusKey: string) => {
    return (
      translatePageChat(`activity_stream.status.${statusKey}`) ||
      {
        queued: 'Queued',
        running: 'Running',
        completed: 'Completed',
        failed: 'Failed',
        cancelled: 'Cancelled',
      }[statusKey] ||
      statusKey
    );
  };

  const answerText = useMemo(
    () => extractAnswerText(snapshot, streamingAnswer),
    [snapshot, streamingAnswer],
  );
  const timelineItems = (() => {
    const items = buildDisplayedTimelineItems(
      snapshot.timeline,
      snapshot.turn.status,
    );
    if (!answerText) return items;

    return items.map((item) => {
      if (
        item.status === 'failed' &&
        translateActivityText(
          item.userActivity.detail_key || item.userActivity.subtitle_key,
          getActivityTranslationValues(item.userActivity.context),
        )?.trim() === answerText.trim()
      ) {
        return {
          ...item,
          userActivity: {
            ...item.userActivity,
            detail_key: null,
          },
        };
      }
      return item;
    });
  })();
  const references = useMemo(
    () => extractReferences(snapshot),
    [snapshot],
  );

  const timestamp = snapshot.turn.finished_at || snapshot.turn.started_at;
  const displayStatus = terminalStatuses.has(snapshot.turn.status)
    ? snapshot.turn.status
    : pending
      ? 'RUNNING'
      : snapshot.turn.status;
  const displayStatusKey = displayStatus.toLowerCase();
  const showAnswerSection = Boolean(answerText) || terminalStatuses.has(displayStatus);
  const showReferencesTrigger =
    references.length > 0 &&
    !(displayStatus === 'COMPLETED' && Boolean(answerText));

  return (
    <div className="flex w-full flex-row gap-3">
      <div>
        <div className="bg-muted text-muted-foreground relative flex size-10 flex-col justify-center rounded-full">
          {pending && (
            <LoaderCircle className="absolute -left-1 size-12 animate-spin opacity-20" />
          )}
          <Bot className="size-5 self-center" />
        </div>
      </div>
      <div className="flex min-w-0 max-w-sm flex-1 flex-col gap-2.5 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        <div className="flex flex-row items-center gap-2">
          <Badge
            variant={getStatusTone(displayStatus)}
            className="h-5 px-2 text-[10px]"
          >
            {getTurnStatusLabel(displayStatusKey)}
          </Badge>
          {timestamp && (
            <div className="text-muted-foreground text-xs">
              {format.dateTime(new Date(timestamp), 'medium')}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-3">
          <Collapsible
            defaultOpen={pending}
            className="group/activity-stream"
          >
            <div className="pl-3">
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground mb-2 flex w-full items-center gap-2 text-left transition-colors"
                >
                  <ChevronRight className="size-3.5 transition-transform group-data-[state=open]/activity-stream:rotate-90" />
                  <span className="text-[11px] font-medium tracking-[0.12em] uppercase">
                    {pageChat('activity_stream.label')}
                  </span>
                  <span className="text-[11px]">
                    {timelineItems.length}
                  </span>
                </button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="flex flex-col gap-0">
                  {timelineItems.length === 0 ? (
                    <div className="text-muted-foreground rounded-lg border border-dashed px-3 py-2 text-sm">
                      {pageChat('activity_stream.empty')}
                    </div>
                  ) : (
                    timelineItems.map((item, index) => {
                      const Icon = getTimelineItemIcon(item);
                      const styles = getTimelineItemStyles(item);
                      const targetTypeLabel =
                        item.userActivity.context?.target_type
                          ? pageChat(
                              `activity_stream.target_type.${item.userActivity.context.target_type}`,
                            )
                          : undefined;
                      const translationValues = getActivityTranslationValues(
                        item.userActivity.context,
                        targetTypeLabel,
                      );
                      const title = translateActivityText(
                        item.userActivity.title_key,
                        translationValues,
                      );
                      const subtitle = translateActivityText(
                        item.userActivity.subtitle_key,
                        translationValues,
                      );
                      const detail = translateActivityText(
                        item.userActivity.detail_key,
                        translationValues,
                      );
                      const hasDebugContent =
                        !!item.argsPreview ||
                        !!item.resultPreview ||
                        !!item.technicalType;

                      const cardBody = (
                        <>
                          <div className="flex flex-wrap items-center gap-2">
                            <div className="text-sm font-medium leading-5">
                              {title}
                            </div>
                            <Badge
                              variant="outline"
                              className={cn(
                                'h-5 rounded-full px-1.5 text-[9px]',
                                styles.badge,
                              )}
                            >
                              {getItemStatusLabel(item.status)}
                            </Badge>
                            {item.timestamp && (
                              <div className="text-muted-foreground text-[10px]">
                                {format.dateTime(
                                  new Date(item.timestamp),
                                  'short',
                                )}
                              </div>
                            )}
                          </div>
                          {subtitle && (
                            <div className="text-muted-foreground mt-1 text-xs">
                              {subtitle}
                            </div>
                          )}
                          {detail && (
                            <div className="text-foreground/85 mt-1.5 text-xs">
                              {detail}
                            </div>
                          )}
                        </>
                      );

                      return (
                        <div key={item.key} className="flex gap-2.5">
                          <div className="flex flex-col items-center">
                            <div
                              className={cn(
                                'mt-0.5 flex size-6 items-center justify-center rounded-full border',
                                styles.iconWrapper,
                              )}
                            >
                              <Icon className="size-3.5" />
                            </div>
                            {index + 1 < timelineItems.length && (
                              <div className="bg-border mt-1.5 min-h-6 w-px flex-1" />
                            )}
                          </div>

                          <div className="min-w-0 flex-1 pb-3">
                            <div className="space-y-2">
                              <div
                                className={cn(
                                  'rounded-xl border px-3 py-2.5 transition-colors',
                                  styles.card,
                                )}
                              >
                                {cardBody}
                              </div>

                              {hasDebugContent && (
                                <Collapsible className="group/timeline-debug">
                                  <CollapsibleTrigger asChild>
                                    <button
                                      type="button"
                                      className="text-muted-foreground hover:text-foreground flex items-center gap-1.5 text-[11px] transition-colors"
                                    >
                                      <ChevronRight className="size-3 transition-transform group-data-[state=open]/timeline-debug:rotate-90" />
                                      <span>
                                        {pageChat(
                                          'activity_stream.debug.title',
                                        )}
                                      </span>
                                    </button>
                                  </CollapsibleTrigger>
                                  <CollapsibleContent className="pt-1">
                                    <div className="bg-muted/20 grid gap-2 rounded-xl border border-dashed px-3 py-2 text-xs">
                                      {item.technicalType && (
                                        <div className="grid gap-1">
                                          <div className="text-muted-foreground">
                                            {pageChat(
                                              'activity_stream.debug.technical_type',
                                            )}
                                          </div>
                                          <div className="font-mono break-all">
                                            {item.technicalType}
                                          </div>
                                        </div>
                                      )}
                                      {item.argsPreview && (
                                        <div className="grid gap-1">
                                          <div className="text-muted-foreground">
                                            {pageChat(
                                              'activity_stream.debug.command_input',
                                            )}
                                          </div>
                                          <pre className="bg-background overflow-x-auto rounded-md border p-2 whitespace-pre-wrap break-all">
                                            {item.argsPreview}
                                          </pre>
                                        </div>
                                      )}
                                      {item.resultPreview && (
                                        <div className="grid gap-1">
                                          <div className="text-muted-foreground">
                                            {pageChat(
                                              'activity_stream.debug.result_summary',
                                            )}
                                          </div>
                                          <pre className="bg-background overflow-x-auto rounded-md border p-2 whitespace-pre-wrap break-all">
                                            {item.resultPreview}
                                          </pre>
                                        </div>
                                      )}
                                    </div>
                                  </CollapsibleContent>
                                </Collapsible>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </CollapsibleContent>
            </div>
          </Collapsible>

          {showAnswerSection && (
            <Card
              className={cn(
                'gap-0 overflow-hidden py-0',
                displayStatus === 'COMPLETED'
                  ? 'border-primary/20 bg-background shadow-sm'
                  : displayStatus === 'FAILED'
                    ? 'border-border/50 bg-muted/20 shadow-none'
                    : 'border-border/60 bg-background/80',
              )}
            >
              <CardContent className="px-4 py-4 text-sm">
                <div className="text-muted-foreground mb-2 text-[11px] font-medium tracking-[0.12em] uppercase">
                  {t(getAnswerSectionTitle(displayStatus, Boolean(answerText)))}
                </div>
                {answerText ? (
                  <Markdown>{answerText}</Markdown>
                ) : pending ? (
                  <div className="space-y-2">
                    <div className="text-muted-foreground text-sm">
                      {t(describeEmptyAnswerState(displayStatus))}
                    </div>
                    <div className="flex flex-row gap-2 py-1">
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0" />
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200" />
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400" />
                    </div>
                  </div>
                ) : (
                  <div className="text-muted-foreground text-sm">
                    {t(describeEmptyAnswerState(displayStatus))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          <Collapsible className="group/details pl-4">
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="text-muted-foreground hover:text-foreground flex items-center gap-2 text-left text-xs transition-colors"
              >
                <ChevronRight className="size-3 transition-transform group-data-[state=open]/details:rotate-90" />
                <span>{pageChat('activity_stream.debug.title')}</span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2">
              <div className="bg-muted/20 grid gap-3 rounded-xl border border-dashed px-4 py-3 text-xs">
                <div className="grid gap-1">
                  <div className="text-muted-foreground">
                    {pageChat('activity_stream.debug.turn_id')}
                  </div>
                  <div className="font-mono break-all">
                    {snapshot.turn.turn_id}
                  </div>
                </div>
                <div className="grid gap-1">
                  <div className="text-muted-foreground">
                    {pageChat('activity_stream.debug.request_id')}
                  </div>
                  <div className="font-mono break-all">
                    {snapshot.turn.request_id}
                  </div>
                </div>
                <div className="grid gap-1">
                  <div className="text-muted-foreground">
                    {pageChat('activity_stream.debug.status')}
                  </div>
                  <div>{getTurnStatusLabel(displayStatusKey)}</div>
                </div>
                {snapshot.turn.error_code && (
                  <div className="grid gap-1">
                    <div className="text-muted-foreground">
                      {pageChat('activity_stream.debug.error_code')}
                    </div>
                    <div className="font-mono">{snapshot.turn.error_code}</div>
                  </div>
                )}
                {snapshot.turn.error_message && (
                  <div className="grid gap-1">
                    <div className="text-muted-foreground">
                      {pageChat('activity_stream.debug.error_message')}
                    </div>
                    <div className="break-all">
                      {snapshot.turn.error_message}
                    </div>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>

        <div className="flex flex-row items-center gap-2">
          {showReferencesTrigger && (
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="sm" className="cursor-pointer">
                  <Badge
                    className="mr-2 h-5 min-w-5 rounded-full px-1 font-mono tabular-nums"
                    variant="destructive"
                  >
                    {references.length}
                  </Badge>
                  {pageChat('references')}
                </Button>
              </SheetTrigger>
              <SheetContent
                side="right"
                className="overflow-y-auto sm:min-w-xl"
              >
                <SheetHeader>
                  <SheetTitle>{pageChat('references')}</SheetTitle>
                </SheetHeader>
                <div className="mt-6 flex flex-col gap-3 px-4 pb-6">
                  {references.map((reference, index) => (
                    <MessageCollapseContent
                      key={`${snapshot.turn.turn_id}-ref-${index}`}
                      defaultOpen={index < 3}
                      title={
                        <div className="flex items-center justify-between gap-2">
                          <div className="truncate">
                            {index + 1}.{' '}
                            {String(
                              reference.metadata?.title ||
                                reference.metadata?.source_id ||
                                reference.text ||
                                'Reference',
                            )}
                          </div>
                          {typeof reference.score === 'number' && (
                            <div className="text-muted-foreground text-xs">
                              {reference.score.toFixed(2)}
                            </div>
                          )}
                        </div>
                      }
                    >
                      <Markdown>{reference.text || ''}</Markdown>
                    </MessageCollapseContent>
                  ))}
                </div>
              </SheetContent>
            </Sheet>
          )}

          <MessageFeedback
            turnId={snapshot.turn.turn_id}
            feedback={feedback}
            onFeedback={onFeedback}
          />

          {answerText && (
            <CopyToClipboard
              variant="ghost"
              className="text-muted-foreground"
              text={answerText}
            />
          )}
        </div>
      </div>
    </div>
  );
};
