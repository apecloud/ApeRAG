'use client';

import { ChatMessage, Feedback, Reference } from '@/api';
import { CopyToClipboard } from '@/components/copy-to-clipboard';
import { Markdown } from '@/components/markdown';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { cn } from '@/lib/utils';
import { Bot, LoaderCircle } from 'lucide-react';
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
  detail?: string;
  timestamp?: string | null;
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

function buildTimelineEntries(
  timeline: AgentTimelineEventEnvelope[],
): TimelineEntry[] {
  const entries: TimelineEntry[] = [];
  for (const event of timeline) {
    let title: string | undefined;
    let detail: string | undefined;

    switch (event.type) {
      case 'agent.state.changed':
        title = event.label || event.status || 'Thinking';
        if (typeof event.data.tool_name === 'string') {
          detail = event.data.tool_name;
        }
        break;
      case 'external_action.started':
        title = 'Searching';
        detail =
          typeof event.data.tool_name === 'string'
            ? event.data.tool_name
            : event.label || undefined;
        break;
      case 'tool.started':
        title = 'Calling Tool';
        detail =
          typeof event.data.tool_name === 'string'
            ? event.data.tool_name
            : event.label || undefined;
        break;
      case 'tool.finished':
        title = 'Reading Result';
        detail =
          typeof event.data.tool_name === 'string'
            ? event.data.tool_name
            : event.label || undefined;
        break;
      case 'turn.completed':
        title = 'Completed';
        break;
      case 'turn.failed':
        title = 'Failed';
        detail =
          typeof event.data.error === 'string' ? event.data.error : undefined;
        break;
      case 'turn.cancelled':
        title = 'Failed';
        detail = 'Cancelled';
        break;
      default:
        break;
    }

    if (!title) continue;

    const previous = entries[entries.length - 1];
    if (previous && previous.title === title && previous.detail === detail) {
      previous.timestamp = event.timestamp;
      continue;
    }

    entries.push({
      key: `${event.sequence}-${event.type}`,
      title,
      detail,
      timestamp: event.timestamp,
    });
  }
  return entries;
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

  const timelineEntries = useMemo(
    () => buildTimelineEntries(snapshot.timeline),
    [snapshot.timeline],
  );
  const answerText = useMemo(
    () => extractAnswerText(snapshot, streamingAnswer, fallbackParts),
    [fallbackParts, snapshot, streamingAnswer],
  );
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

  return (
    <div className="flex w-max flex-row gap-4">
      <div>
        <div className="bg-muted text-muted-foreground relative flex size-12 flex-col justify-center rounded-full">
          {pending && (
            <LoaderCircle className="absolute -left-1 size-14 animate-spin opacity-20" />
          )}
          <Bot className="size-6 self-center" />
        </div>
      </div>
      <div className="flex max-w-sm flex-col gap-2 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        <div className="flex flex-row items-center gap-2">
          <Badge
            variant={displayStatus === 'COMPLETED' ? 'default' : 'secondary'}
          >
            {displayStatus}
          </Badge>
          {timestamp && (
            <div className="text-muted-foreground text-xs">
              {format.dateTime(new Date(timestamp), 'medium')}
            </div>
          )}
        </div>

        <Card className="dark:border-card/0 block gap-0 px-4 py-4 text-sm">
          <MessageCollapseContent
            title="Timeline"
            defaultOpen={pending || timelineEntries.length <= 4}
            animate={pending}
          >
            <div className="flex flex-col gap-3">
              {timelineEntries.length === 0 ? (
                <div className="text-muted-foreground text-sm">
                  Waiting for events...
                </div>
              ) : (
                timelineEntries.map((entry, index) => (
                  <div key={entry.key} className="flex gap-3">
                    <div className="flex flex-col items-center">
                      <div className="bg-primary mt-1 size-2 rounded-full" />
                      {index + 1 < timelineEntries.length && (
                        <div className="bg-border mt-1 h-full w-px flex-1" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1 pb-2">
                      <div className="font-medium">{entry.title}</div>
                      {entry.detail && (
                        <div className="text-muted-foreground mt-1 text-xs break-all">
                          {entry.detail}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </MessageCollapseContent>

          <div className="mt-4">
            {answerText ? (
              <Markdown>{answerText}</Markdown>
            ) : (
              <div className="flex flex-row gap-2 py-2">
                <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0" />
                <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200" />
                <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400" />
              </div>
            )}
          </div>

          <div className="mt-4">
            <MessageCollapseContent title="Diagnostics">
              <div className="grid gap-3 text-xs">
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
            </MessageCollapseContent>
          </div>
        </Card>

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
