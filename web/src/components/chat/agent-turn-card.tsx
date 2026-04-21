'use client';

import { ChatMessage, Feedback, Reference } from '@/api';
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
  Bot,
  BrainCircuit,
  ChevronRight,
  LoaderCircle,
  TerminalSquare,
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
  label?: string | null;
  status?: string | null;
  actor: 'agent' | 'tool' | 'system';
  data: Record<string, unknown>;
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

type TimelineEntry = {
  key: string;
  title: string;
  subtitle?: string;
  timestamp?: string | null;
};

type OrderedTimelineItem = TimelineEntry & {
  kind: 'activity' | 'command';
  status: string;
  commandLabel?: string;
  argsPreview?: string;
  resultPreview?: string;
  occurrences?: number;
};

const terminalStatuses = new Set(['COMPLETED', 'FAILED', 'CANCELLED']);

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
  fallbackParts: ChatMessage[],
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
  return fallbackParts
    .filter((part) => part.type === 'message')
    .map((part) => part.data || '')
    .join('');
}

function extractReferences(
  snapshot: AgentTurnSnapshot,
  fallbackParts: ChatMessage[],
): Reference[] {
  const referenceArtifact = snapshot.artifacts.find(
    (artifact) => artifact.artifact_type === 'reference_bundle',
  );
  const items = Array.isArray(referenceArtifact?.payload?.items)
    ? (referenceArtifact.payload.items as ReferenceBundleItem[])
    : [];
  if (items.length > 0) {
    return items.map(mapReferenceItem);
  }
  return (
    fallbackParts.findLast((part) => Array.isArray(part.references))
      ?.references || []
  );
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

function humanizeToolName(toolName: string) {
  return toolName.replace(/[_-]+/g, ' ').trim();
}

function describeActivityStep(event: AgentTimelineEventEnvelope) {
  const status = String(event.status || '').toLowerCase();
  const toolName =
    typeof event.data.tool_name === 'string'
      ? humanizeToolName(event.data.tool_name)
      : undefined;

  switch (status) {
    case 'thinking':
      return {
        title: 'Planning the next step',
        subtitle: 'Reviewing the request and deciding what to do next.',
      };
    case 'searching':
      return {
        title: 'Looking for supporting context',
        subtitle: 'Searching for information that can support the reply.',
      };
    case 'calling_tool':
      return null;
    case 'reading_result':
      return {
        title: toolName
          ? `Reviewing results from ${toolName}`
          : 'Reviewing the latest results',
        subtitle: 'Using the returned context to decide the next move.',
      };
    case 'composing':
      return {
        title: 'Writing the answer',
        subtitle: 'Turning the gathered context into a response.',
      };
    default:
      return null;
  }
}

function summarizeCollectionResult(result: unknown) {
  if (!result || typeof result !== 'object') return undefined;

  const items = (result as { items?: unknown }).items;
  if (!Array.isArray(items)) return undefined;

  if (items.length === 0) {
    return 'No knowledge bases were available for this step.';
  }

  return `Found ${items.length} knowledge base${items.length === 1 ? '' : 's'}.`;
}

function summarizeToolResult(label: string, result: unknown, status: string) {
  if (status === 'failed') {
    if (typeof result === 'string' && result.trim()) {
      return result.trim();
    }
    return 'This step ended before returning a usable result.';
  }

  switch (label) {
    case 'list_collections':
      return summarizeCollectionResult(result);
    default: {
      const preview = compactPreview(result, 180);
      return preview
        ? preview.replace(/\s+/g, ' ').trim()
        : 'Step completed and returned context for the answer.';
    }
  }
}

function describeCommandStep(label: string, status: string, result: unknown) {
  switch (label) {
    case 'list_collections':
      return status === 'running'
        ? {
            title: 'Checking available knowledge bases',
            subtitle: 'Finding which knowledge bases can be used in this reply.',
          }
        : {
            title:
              status === 'failed'
                ? 'Could not check available knowledge bases'
                : 'Checked available knowledge bases',
            subtitle: summarizeToolResult(label, result, status),
          };
    case 'search_web':
    case 'web_search':
      return status === 'running'
        ? {
            title: 'Searching the web for supporting context',
            subtitle: 'Looking for external information that can support the reply.',
          }
        : {
            title:
              status === 'failed'
                ? 'Could not search the web'
                : 'Searched the web for supporting context',
            subtitle: summarizeToolResult(label, result, status),
          };
    case 'external_action':
      return status === 'running'
        ? {
            title: 'Calling an external action',
            subtitle: 'Waiting for the external action to return more context.',
          }
        : {
            title:
              status === 'failed'
                ? 'External action did not complete'
                : 'Completed an external action',
            subtitle: summarizeToolResult(label, result, status),
          };
    default: {
      const readableLabel = humanizeToolName(label);
      return status === 'running'
        ? {
            title: `Using ${readableLabel}`,
            subtitle: 'Gathering context that can support the reply.',
          }
        : {
            title:
              status === 'failed'
                ? `${readableLabel} did not complete`
                : `Used ${readableLabel}`,
            subtitle: summarizeToolResult(label, result, status),
          };
    }
  }
}

function findOpenCommandIndex(
  entries: OrderedTimelineItem[],
  commandLabel: string,
) {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const item = entries[index];
    if (
      item.kind === 'command' &&
      item.commandLabel === commandLabel &&
      item.status === 'running'
    ) {
      return index;
    }
  }
  return -1;
}

function buildOrderedTimelineItems(
  timeline: AgentTimelineEventEnvelope[],
): OrderedTimelineItem[] {
  const orderedEvents = [...timeline].sort((left, right) => {
    if (left.sequence !== right.sequence) {
      return left.sequence - right.sequence;
    }
    return new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime();
  });
  const entries: OrderedTimelineItem[] = [];

  for (const event of orderedEvents) {
    if (event.type === 'agent.state.changed') {
      const status = String(event.status || '').toLowerCase();
      const description = describeActivityStep(event);
      if (!description) continue;
      const previous = entries[entries.length - 1];
      if (
        previous &&
        previous.kind === 'activity' &&
        previous.status === status &&
        previous.title === description.title &&
        previous.subtitle === description.subtitle
      ) {
        previous.timestamp = event.timestamp;
        previous.occurrences = (previous.occurrences || 1) + 1;
        continue;
      }

      entries.push({
        key: `${event.sequence}-${event.type}`,
        kind: 'activity',
        title: description.title,
        subtitle: description.subtitle,
        timestamp: event.timestamp,
        status,
        occurrences: 1,
      });
      continue;
    }

    if (
      event.type === 'tool.started' ||
      event.type === 'tool.finished' ||
      event.type === 'external_action.started' ||
      event.type === 'external_action.finished'
    ) {
      const fallbackLabel =
        event.type.startsWith('external_action') ? 'external_action' : 'tool';
      const rawLabel =
        typeof event.data.tool_name === 'string'
          ? event.data.tool_name
          : event.label || fallbackLabel;
      const normalizedStatus =
        event.type.endsWith('.started')
          ? 'running'
          : String(event.status || 'finished').toLowerCase();
      const description = describeCommandStep(
        rawLabel,
        normalizedStatus,
        event.data.result,
      );
      const nextArgsPreview = compactPreview(event.data.args, 180);
      const nextResultPreview = compactPreview(event.data.result, 280);

      if (event.type.endsWith('.started')) {
        entries.push({
          key: `${event.sequence}-${event.type}`,
          kind: 'command',
          title: description.title,
          subtitle: description.subtitle,
          timestamp: event.timestamp,
          status: normalizedStatus,
          commandLabel: rawLabel,
          argsPreview: nextArgsPreview,
          resultPreview: nextResultPreview,
        });
        continue;
      }

      const openCommandIndex = findOpenCommandIndex(entries, rawLabel);
      if (openCommandIndex >= 0) {
        const previous = entries[openCommandIndex];
        entries[openCommandIndex] = {
          ...previous,
          title: description.title,
          subtitle: description.subtitle,
          timestamp: event.timestamp,
          status: normalizedStatus,
          argsPreview: previous.argsPreview || nextArgsPreview,
          resultPreview: nextResultPreview || previous.resultPreview,
        };
        continue;
      }

      entries.push({
        key: `${event.sequence}-${event.type}`,
        kind: 'command',
        title: description.title,
        subtitle: description.subtitle,
        timestamp: event.timestamp,
        status: normalizedStatus,
        commandLabel: rawLabel,
        argsPreview: nextArgsPreview,
        resultPreview: nextResultPreview,
      });
      continue;
    }

    if (event.type === 'turn.failed' || event.type === 'turn.cancelled') {
      entries.push({
        key: `${event.sequence}-${event.type}`,
        kind: 'activity',
        title: event.type === 'turn.failed' ? 'Run failed' : 'Run cancelled',
        subtitle:
          typeof event.data.error === 'string' ? event.data.error : undefined,
        timestamp: event.timestamp,
        status: event.type === 'turn.failed' ? 'failed' : 'cancelled',
      });
    }
  }

  return entries;
}

function getAnswerSectionTitle(status: string, hasAnswerText: boolean) {
  if (status === 'FAILED') return hasAnswerText ? 'Failure details' : 'Run failed';
  if (status === 'CANCELLED') return hasAnswerText ? 'Cancelled output' : 'Run cancelled';
  if (status === 'COMPLETED') return 'Final answer';
  return hasAnswerText ? 'Draft answer' : 'Answer';
}

function describeEmptyAnswerState(status: string) {
  if (status === 'FAILED') {
    return 'This run ended before a final answer was produced.';
  }
  if (status === 'CANCELLED') {
    return 'This run was cancelled before a final answer was produced.';
  }
  return 'The answer will appear here once the activity stream finishes.';
}

function getTimelineItemBadgeLabel(item: OrderedTimelineItem) {
  if (item.kind === 'activity') return 'step';
  if (item.status === 'running') return 'running';
  if (item.status === 'success') return 'done';
  if (item.status === 'failed') return 'failed';
  return 'command';
}

function getTimelineItemIcon(item: OrderedTimelineItem) {
  return item.kind === 'activity' ? BrainCircuit : TerminalSquare;
}

function getTimelineItemStyles(item: OrderedTimelineItem) {
  if (item.kind === 'activity') {
    return {
      iconWrapper: 'border-primary/25 bg-primary/10 text-primary',
      card: 'border-border/60 bg-muted/25',
    };
  }

  if (item.status === 'failed') {
    return {
      iconWrapper: 'border-destructive/25 bg-destructive/10 text-destructive',
      card: 'border-destructive/20 bg-destructive/5',
    };
  }

  if (item.status === 'running') {
    return {
      iconWrapper:
        'border-emerald-500/25 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
      card: 'border-emerald-500/15 bg-emerald-500/5',
    };
  }

  return {
    iconWrapper: 'border-border/70 bg-background text-foreground',
    card: 'border-border/60 bg-background/70',
  };
}

export const AgentTurnCard = ({
  snapshot,
  pending,
  streamingAnswer,
  fallbackParts,
  onFeedback,
}: {
  snapshot: AgentTurnSnapshot;
  pending: boolean;
  streamingAnswer: string;
  fallbackParts: ChatMessage[];
  onFeedback: (part: ChatMessage, feedback: Feedback) => void;
}) => {
  const pageChat = useTranslations('page_chat');
  const format = useFormatter();

  const answerText = useMemo(
    () => extractAnswerText(snapshot, streamingAnswer, fallbackParts),
    [fallbackParts, snapshot, streamingAnswer],
  );
  const timelineItems = useMemo(() => {
    const items = buildOrderedTimelineItems(snapshot.timeline);
    if (!answerText) return items;

    return items.map((item) => {
      if (
        item.kind === 'activity' &&
        item.status === 'failed' &&
        item.subtitle?.trim() === answerText.trim()
      ) {
        return {
          ...item,
          subtitle: undefined,
        };
      }
      return item;
    });
  }, [answerText, snapshot.timeline]);
  const references = useMemo(
    () => extractReferences(snapshot, fallbackParts),
    [fallbackParts, snapshot],
  );

  const feedbackParts = useMemo<ChatMessage[]>(() => {
    const feedbackPart = fallbackParts.findLast((part) => part.references);
    if (feedbackPart) return fallbackParts;
    if (references.length === 0) return fallbackParts;
    return [
      {
        id: snapshot.turn.turn_id,
        type: 'references',
        role: 'ai',
        data: '',
        references,
      },
    ];
  }, [fallbackParts, references, snapshot.turn.turn_id]);

  const timestamp = snapshot.turn.finished_at || snapshot.turn.started_at;
  const displayStatus = terminalStatuses.has(snapshot.turn.status)
    ? snapshot.turn.status
    : pending
      ? 'RUNNING'
      : snapshot.turn.status;
  const showAnswerSection = Boolean(answerText) || terminalStatuses.has(displayStatus);

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
            variant={displayStatus === 'COMPLETED' ? 'default' : 'secondary'}
            className="h-5 px-2 text-[10px]"
          >
            {displayStatus}
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
                    Activity stream
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
                      Waiting for activity events...
                    </div>
                  ) : (
                    timelineItems.map((item, index) => {
                      const Icon = getTimelineItemIcon(item);
                      const styles = getTimelineItemStyles(item);
                      const hasExpandableContent =
                        !!item.argsPreview || !!item.resultPreview;

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
                            {hasExpandableContent ? (
                              <Collapsible
                                defaultOpen={pending && item.status === 'running'}
                                className="group/timeline-item"
                              >
                                <div
                                  className={cn(
                                    'rounded-xl border px-3 py-2.5',
                                    styles.card,
                                  )}
                                >
                                  <CollapsibleTrigger asChild>
                                    <button
                                      type="button"
                                      className="flex w-full items-start gap-2 text-left"
                                    >
                                      <ChevronRight className="text-muted-foreground mt-0.5 size-3.5 shrink-0 transition-transform group-data-[state=open]/timeline-item:rotate-90" />
                                      <div className="min-w-0 flex-1">
                                        <div className="flex flex-wrap items-center gap-1.5">
                                          <div className="text-sm font-medium">
                                            {item.title}
                                          </div>
                                          <Badge
                                            variant="outline"
                                            className="h-5 rounded-full px-1.5 text-[9px] uppercase"
                                          >
                                            {getTimelineItemBadgeLabel(item)}
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
                                        {item.subtitle && (
                                          <div className="text-muted-foreground mt-1 text-xs">
                                            {item.subtitle}
                                          </div>
                                        )}
                                        {item.kind === 'activity' &&
                                          (item.occurrences || 1) > 1 && (
                                            <div className="text-muted-foreground mt-1.5 text-[11px]">
                                              Repeated {item.occurrences} times
                                              while this step stayed active.
                                            </div>
                                          )}
                                      </div>
                                    </button>
                                  </CollapsibleTrigger>
                                  <CollapsibleContent className="mt-2 border-t pt-2">
                                    <div className="grid gap-2 text-xs">
                                      {item.argsPreview && (
                                        <div className="grid gap-1">
                                          <div className="text-muted-foreground">
                                            Command input
                                          </div>
                                          <pre className="bg-background overflow-x-auto rounded-md border p-2 whitespace-pre-wrap break-all">
                                            {item.argsPreview}
                                          </pre>
                                        </div>
                                      )}
                                      {item.resultPreview && (
                                        <div className="grid gap-1">
                                          <div className="text-muted-foreground">
                                            Result summary
                                          </div>
                                          <pre className="bg-background overflow-x-auto rounded-md border p-2 whitespace-pre-wrap break-all">
                                            {item.resultPreview}
                                          </pre>
                                        </div>
                                      )}
                                    </div>
                                  </CollapsibleContent>
                                </div>
                              </Collapsible>
                            ) : (
                              <div
                                className={cn(
                                  'rounded-xl border px-3 py-2.5',
                                  styles.card,
                                )}
                              >
                                <div className="flex flex-wrap items-center gap-1.5">
                                  <div className="text-sm font-medium">
                                    {item.title}
                                  </div>
                                  <Badge
                                    variant="outline"
                                    className="h-5 rounded-full px-1.5 text-[9px] uppercase"
                                  >
                                    {getTimelineItemBadgeLabel(item)}
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
                                {item.subtitle && (
                                  <div className="text-muted-foreground mt-1 text-xs">
                                    {item.subtitle}
                                  </div>
                                )}
                                {item.kind === 'activity' &&
                                  (item.occurrences || 1) > 1 && (
                                    <div className="text-muted-foreground mt-1.5 text-[11px]">
                                      Repeated {item.occurrences} times while
                                      this step stayed active.
                                    </div>
                                  )}
                              </div>
                            )}
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
                  {getAnswerSectionTitle(displayStatus, Boolean(answerText))}
                </div>
                {answerText ? (
                  <Markdown>{answerText}</Markdown>
                ) : pending ? (
                  <div className="space-y-2">
                    <div className="text-muted-foreground text-sm">
                      {describeEmptyAnswerState(displayStatus)}
                    </div>
                    <div className="flex flex-row gap-2 py-1">
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0" />
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200" />
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400" />
                    </div>
                  </div>
                ) : (
                  <div className="text-muted-foreground text-sm">
                    {describeEmptyAnswerState(displayStatus)}
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
                <span>Details</span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2">
              <div className="bg-muted/20 grid gap-3 rounded-xl border border-dashed px-4 py-3 text-xs">
                <div className="grid gap-1">
                  <div className="text-muted-foreground">Turn ID</div>
                  <div className="font-mono break-all">
                    {snapshot.turn.turn_id}
                  </div>
                </div>
                <div className="grid gap-1">
                  <div className="text-muted-foreground">Request ID</div>
                  <div className="font-mono break-all">
                    {snapshot.turn.request_id}
                  </div>
                </div>
                <div className="grid gap-1">
                  <div className="text-muted-foreground">Status</div>
                  <div>{snapshot.turn.status}</div>
                </div>
                {snapshot.turn.error_code && (
                  <div className="grid gap-1">
                    <div className="text-muted-foreground">Error Code</div>
                    <div className="font-mono">{snapshot.turn.error_code}</div>
                  </div>
                )}
                {snapshot.turn.error_message && (
                  <div className="grid gap-1">
                    <div className="text-muted-foreground">Error Message</div>
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
          {references.length > 0 && (
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
            parts={feedbackParts}
            hanldeMessageFeedback={onFeedback}
          />

          <CopyToClipboard
            variant="ghost"
            className={cn('text-muted-foreground', !answerText && 'opacity-50')}
            text={answerText}
          />
        </div>
      </div>
    </div>
  );
};
