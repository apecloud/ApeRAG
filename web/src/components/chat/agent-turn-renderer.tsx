'use client';

// Phase 8 D8.4b — message-parts renderer.
//
// Consumes the new `useAgentTurnStream` seam (D8.4a, merge commit
// 63a9d522) directly and renders each `AgentMessagePart` by type. The
// previous `AgentTurnCard` + `legacy-snapshot-shim.ts` projection is
// retired here.
//
// Rendering layout (preserved from L1 design + agent-turn-card visual
// baseline so non-technical users see the same affordance):
//
//   ┌─ avatar ────┐  ┌─ status badge + timestamp ────────────────┐
//   │             │  ├─ Activity stream collapsible (top) ───────┤
//   │             │  │   tool calls + transient activity badge   │
//   │             │  │   inline ConsentSlot / ElicitationSlot    │
//   │             │  │     (interactive bodies are #78 territory)│
//   │             │  ├─ Answer section (markdown text parts) ────┤
//   │             │  ├─ Debug collapsible (turn id / req id) ────┤
//   │             │  └─ References + feedback + copy (bottom) ───┘
//
// Slot props (the only seam crossing into #78 territory):
//   * `ConsentSlot` consumes one `AgentToolConsentPart` and the
//     `chatId` / `turnId` so it can call `decideToolConsent(...)`
//     from `features/agent-runtime/api.ts`.
//   * `ElicitationSlot` consumes one `AgentElicitationPart` similarly.
//
// Both slots are optional: when omitted, the renderer falls back to a
// minimal "awaiting decision" placeholder so the parts stream is
// never silently dropped. #78 will provide concrete slot
// implementations.

import { CopyToClipboard } from '@/components/copy-to-clipboard';
import { Markdown } from '@/components/markdown';
import {
  type AgentCitationPart,
  type AgentElicitationPart,
  type AgentMessagePart,
  type AgentSourceDocumentPart,
  type AgentSourceUrlPart,
  type AgentStreamStatus,
  type AgentTextPart,
  type AgentToolConsentPart,
  type AgentToolPart,
  type AgentTurnEnvelope,
  type ActivityData,
  type CitationLocation,
} from '@/features/agent-runtime';
import type { Feedback, Reference } from '@/features/bot/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
} from '@/components/ui/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
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
  BookOpen,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock3,
  CornerDownRight,
  HandCoins,
  HelpCircle,
  LoaderCircle,
  Sparkles,
  Wrench,
  XCircle,
} from 'lucide-react';
import { useFormatter, useTranslations } from 'next-intl';
import { useMemo } from 'react';
import { MessageCollapseContent } from './message-collapse-content';
import { MessageFeedback } from './message-feedback';

// ---------------------------------------------------------------------------
// Public seam props (consumed by chat-messages.tsx + #78 chenyexuan slot
// implementations).

export type ConsentSlotProps = {
  chatId: string;
  turnId: string;
  part: AgentToolConsentPart;
};

export type ElicitationSlotProps = {
  chatId: string;
  turnId: string;
  part: AgentElicitationPart;
};

export type AgentTurnRendererProps = {
  chatId: string;
  turn: AgentTurnEnvelope;
  parts: AgentMessagePart[];
  transientActivity: ActivityData | null;
  status: AgentStreamStatus;
  errorText: string | null;
  feedback?: Feedback;
  onFeedback: (turnId: string, feedback: Feedback) => Promise<void> | void;
  /** Interactive consent UI; provided by #78 chenyexuan. */
  ConsentSlot?: React.ComponentType<ConsentSlotProps>;
  /** Interactive elicitation UI; provided by #78 chenyexuan. */
  ElicitationSlot?: React.ComponentType<ElicitationSlotProps>;
};

// ---------------------------------------------------------------------------

const TERMINAL_STATUSES: ReadonlySet<AgentStreamStatus> = new Set([
  'completed',
  'failed',
  'cancelled',
  'aborted',
]);

const STATUS_LABEL_KEY: Record<AgentStreamStatus, string> = {
  idle: 'queued',
  connecting: 'running',
  streaming: 'running',
  completed: 'completed',
  failed: 'failed',
  cancelled: 'cancelled',
  aborted: 'cancelled',
};

const STATUS_BADGE_TONE: Record<
  AgentStreamStatus,
  'default' | 'secondary' | 'destructive'
> = {
  idle: 'secondary',
  connecting: 'secondary',
  streaming: 'default',
  completed: 'default',
  failed: 'destructive',
  cancelled: 'destructive',
  aborted: 'destructive',
};

// ---------------------------------------------------------------------------

function partitionParts(parts: AgentMessagePart[]) {
  const text: AgentTextPart[] = [];
  const tool: AgentToolPart[] = [];
  const sourceUrl: AgentSourceUrlPart[] = [];
  const sourceDoc: AgentSourceDocumentPart[] = [];
  const citation: AgentCitationPart[] = [];
  const consent: AgentToolConsentPart[] = [];
  const elicitation: AgentElicitationPart[] = [];
  for (const part of parts) {
    if (part.type === 'text') text.push(part);
    else if (part.type === 'source-url') sourceUrl.push(part);
    else if (part.type === 'source-document') sourceDoc.push(part);
    else if (part.type === 'data-citation') citation.push(part);
    else if (part.type === 'data-tool-consent') consent.push(part);
    else if (part.type === 'data-elicitation') elicitation.push(part);
    else if (part.type.startsWith('tool-')) tool.push(part as AgentToolPart);
  }
  return { text, tool, sourceUrl, sourceDoc, citation, consent, elicitation };
}

function joinTextParts(parts: AgentTextPart[]): string {
  return parts.map((p) => p.text).filter(Boolean).join('\n\n');
}

function citationToReference(part: AgentCitationPart): Reference {
  const loc = part.data.location;
  return {
    text: part.data.cited_text,
    metadata: {
      title: locationTitle(loc),
      uri: 'url' in loc ? (loc.url ?? undefined) : undefined,
      source_type: loc.type,
    },
  };
}

function locationTitle(loc: CitationLocation): string | undefined {
  if ('title' in loc && loc.title) return loc.title;
  if ('doc_title' in loc && loc.doc_title) return loc.doc_title;
  if ('url' in loc && loc.url) return loc.url;
  return undefined;
}

function sourceUrlToReference(part: AgentSourceUrlPart): Reference {
  return {
    text: part.title || part.url,
    metadata: {
      title: part.title || undefined,
      uri: part.url,
      source_type: 'source-url',
    },
  };
}

function sourceDocToReference(part: AgentSourceDocumentPart): Reference {
  return {
    text: part.title,
    metadata: {
      title: part.title,
      source_type: 'source-document',
      uri: part.mediaType,
    },
  };
}

function toolDisplayName(part: AgentToolPart): string {
  return part.toolName || part.type.replace(/^tool-/, '') || 'tool';
}

// ---------------------------------------------------------------------------

function ToolActivityItem({ part }: { part: AgentToolPart }) {
  const pageChat = useTranslations('page_chat');
  const tone = toolStateTone(part.state);
  const Icon = toolStateIcon(part.state);
  const name = toolDisplayName(part);
  const inputPreview = previewJson(part.input, 220);
  const outputPreview = part.errorText
    ? part.errorText
    : previewJson(part.output, 280);
  const hasDebug =
    part.state === 'output-error' &&
    (inputPreview != null || outputPreview != null);

  return (
    <div className="flex gap-2.5">
      <div className="flex pt-[3px]">
        <Icon className={cn('size-3.5 flex-none', tone.icon)} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[13px] leading-snug">
          <span
            className={cn(
              'font-medium',
              tone.title,
              part.state === 'input-streaming' || part.state === 'input-available'
                ? 'animate-pulse'
                : '',
            )}
          >
            {pageChat('activity_stream.tool.title', { name })}
          </span>
          <span className={cn('break-words', tone.subtitle)}>
            {pageChat(`activity_stream.tool.state.${part.state}`)}
          </span>
        </div>
        {hasDebug && (
          <Collapsible className="group/timeline-debug mt-1.5">
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="text-muted-foreground/80 hover:text-foreground flex items-center gap-1 text-[10.5px] transition-colors"
              >
                <ChevronRight className="size-3 transition-transform group-data-[state=open]/timeline-debug:rotate-90" />
                <span>{pageChat('activity_stream.debug.title')}</span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-1.5">
              <div className="border-border/60 bg-background/60 grid gap-1.5 rounded-md border px-2.5 py-1.5 text-[11px]">
                {inputPreview && (
                  <div className="grid gap-0.5">
                    <div className="text-muted-foreground/80">
                      {pageChat('activity_stream.debug.command_input')}
                    </div>
                    <pre className="bg-background/80 border-border/40 overflow-x-auto rounded border px-2 py-1 whitespace-pre-wrap break-all">
                      {inputPreview}
                    </pre>
                  </div>
                )}
                {outputPreview && (
                  <div className="grid gap-0.5">
                    <div className="text-muted-foreground/80">
                      {pageChat('activity_stream.debug.result_summary')}
                    </div>
                    <pre
                      className={cn(
                        'bg-background/80 border-border/40 overflow-x-auto rounded border px-2 py-1 whitespace-pre-wrap break-all',
                        part.state === 'output-error' && 'text-destructive',
                      )}
                    >
                      {outputPreview}
                    </pre>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}

function toolStateIcon(state: AgentToolPart['state']) {
  switch (state) {
    case 'input-streaming':
    case 'input-available':
      return Wrench;
    case 'output-available':
      return CheckCircle2;
    case 'output-error':
      return XCircle;
    default:
      return Wrench;
  }
}

function toolStateTone(state: AgentToolPart['state']) {
  if (state === 'output-error') {
    return {
      icon: 'text-destructive',
      title: 'text-destructive',
      subtitle: 'text-destructive/75',
    };
  }
  if (state === 'output-available') {
    return {
      icon: 'text-muted-foreground/70',
      title: 'text-foreground/80',
      subtitle: 'text-muted-foreground',
    };
  }
  return {
    icon: 'text-primary',
    title: 'text-foreground',
    subtitle: 'text-muted-foreground',
  };
}

// ---------------------------------------------------------------------------

const TRANSIENT_INTENTS = [
  'thinking',
  'searching_knowledge',
  'reading_source',
  'comparing_results',
  'writing_answer',
  'waiting',
  'completed',
  'error',
] as const;

type TransientIntent = (typeof TRANSIENT_INTENTS)[number];

function isTransientIntent(value: unknown): value is TransientIntent {
  return (
    typeof value === 'string' &&
    (TRANSIENT_INTENTS as readonly string[]).includes(value)
  );
}

function ActivityIndicator({ activity }: { activity: ActivityData | null }) {
  const pageChat = useTranslations('page_chat');
  if (!activity) return null;
  const rawIntent = activity.activity?.intent || activity.intent;
  const intent: TransientIntent = isTransientIntent(rawIntent)
    ? rawIntent
    : 'thinking';
  const Icon = activityIntentIcon(intent);
  const label = pageChat(`activity_stream.transient.${intent}` as const);
  return (
    <div className="flex items-center gap-2 px-3.5 pt-2 pb-1 text-[12px]">
      <Icon className="text-primary size-3.5 animate-pulse" />
      <span className="text-muted-foreground italic">{label}</span>
    </div>
  );
}

function activityIntentIcon(intent: string) {
  switch (intent) {
    case 'searching_knowledge':
      return Sparkles;
    case 'reading_source':
      return BookOpen;
    case 'comparing_results':
      return Brain;
    case 'writing_answer':
      return Sparkles;
    case 'completed':
      return CheckCircle2;
    case 'error':
      return AlertTriangle;
    default:
      return Clock3;
  }
}

// ---------------------------------------------------------------------------

function ConsentPlaceholder({ part }: { part: AgentToolConsentPart }) {
  const pageChat = useTranslations('page_chat');
  return (
    <div className="border-primary/40 bg-accent-soft text-accent-ink flex items-start gap-2 rounded-md border px-3 py-2 text-[12px]">
      <HandCoins className="text-primary mt-0.5 size-3.5 flex-none" />
      <div className="min-w-0 flex-1">
        <div className="font-medium">
          {pageChat('activity_stream.consent.placeholder_title', {
            name: part.data.toolName,
          })}
        </div>
        <div className="text-accent-ink/70 mt-0.5 break-all">
          {pageChat('activity_stream.consent.placeholder_state', {
            state: part.data.state,
          })}
        </div>
      </div>
    </div>
  );
}

function ElicitationPlaceholder({ part }: { part: AgentElicitationPart }) {
  const pageChat = useTranslations('page_chat');
  return (
    <div className="border-primary/40 bg-accent-soft text-accent-ink flex items-start gap-2 rounded-md border px-3 py-2 text-[12px]">
      <HelpCircle className="text-primary mt-0.5 size-3.5 flex-none" />
      <div className="min-w-0 flex-1">
        <div className="font-medium">{part.data.prompt}</div>
        <div className="text-accent-ink/70 mt-0.5 break-all">
          {pageChat('activity_stream.elicitation.placeholder_state', {
            state: part.data.state,
          })}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

export function AgentTurnRenderer({
  chatId,
  turn,
  parts,
  transientActivity,
  status,
  errorText,
  feedback,
  onFeedback,
  ConsentSlot,
  ElicitationSlot,
}: AgentTurnRendererProps) {
  const pageChat = useTranslations('page_chat');
  const format = useFormatter();

  const grouped = useMemo(() => partitionParts(parts), [parts]);
  const answerText = useMemo(() => joinTextParts(grouped.text), [grouped.text]);
  const references = useMemo<Reference[]>(() => {
    const fromUrls = grouped.sourceUrl.map(sourceUrlToReference);
    const fromDocs = grouped.sourceDoc.map(sourceDocToReference);
    const fromCitations = grouped.citation.map(citationToReference);
    return [...fromUrls, ...fromDocs, ...fromCitations];
  }, [grouped.citation, grouped.sourceDoc, grouped.sourceUrl]);

  const pending = !TERMINAL_STATUSES.has(status);
  const statusKey = STATUS_LABEL_KEY[status];
  const statusTone = STATUS_BADGE_TONE[status];
  const showHeaderStatus = status !== 'completed';

  const timestamp = turn.finished_at || turn.started_at;
  const showAnswerSection =
    Boolean(answerText) || TERMINAL_STATUSES.has(status);
  const showReferences =
    references.length > 0 && !(status === 'completed' && Boolean(answerText));
  const hasTurnDebugDetails =
    status === 'failed' ||
    Boolean(errorText || turn.error_code || turn.error_message);
  const copyText = answerText || errorText || turn.error_message || '';

  const traceMetaParts: string[] = [];
  if (grouped.tool.length > 0) {
    traceMetaParts.push(
      pageChat('activity_stream.meta.steps', { count: grouped.tool.length }),
    );
  }
  if (references.length > 0) {
    traceMetaParts.push(
      pageChat('activity_stream.meta.sources', { count: references.length }),
    );
  }
  const traceMeta = traceMetaParts.join(' · ');

  const hasActivity =
    grouped.tool.length +
      grouped.consent.length +
      grouped.elicitation.length >
    0;

  return (
    <div className="flex w-full flex-row gap-3.5">
      <div>
        <div className="bg-accent-soft text-accent-ink relative flex size-7 items-center justify-center rounded-full">
          {pending && (
            <LoaderCircle className="absolute -inset-1 size-9 animate-spin opacity-20" />
          )}
          <Sparkles className="size-3.5" />
        </div>
      </div>
      <div className="flex min-w-0 max-w-sm flex-1 flex-col gap-3 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        {showHeaderStatus && (
          <div className="flex flex-row items-center gap-2">
            <Badge
              variant={statusTone}
              className="h-5 px-2 text-[10px]"
            >
              {pageChat(
                `activity_stream.status.${statusKey}` as never,
                {} as never,
              )}
            </Badge>
            {timestamp && (
              <div className="text-muted-foreground font-mono text-[11px]">
                {format.dateTime(new Date(timestamp), 'medium')}
              </div>
            )}
          </div>
        )}

        <Collapsible
          defaultOpen
          className="group/activity-stream bg-muted border-border/70 overflow-hidden rounded-xl border"
        >
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="text-muted-foreground hover:text-foreground flex w-full items-center gap-2 px-3.5 py-2.5 text-left transition-colors"
            >
              <Sparkles className="text-primary size-3" />
              <span className="font-mono text-[10.5px] tracking-[0.08em] uppercase">
                {pageChat('activity_stream.label')}
              </span>
              {traceMeta && (
                <span className="text-muted-foreground/80 ml-2 truncate text-[11px]">
                  {traceMeta}
                </span>
              )}
              <ChevronRight className="ml-auto size-3.5 transition-transform group-data-[state=open]/activity-stream:rotate-90" />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ActivityIndicator activity={transientActivity} />
            <div className="border-border/70 flex flex-col gap-2.5 border-t px-4 py-3.5">
              {!hasActivity && (
                <div className="text-muted-foreground py-1 text-[13px]">
                  {pending
                    ? pageChat('activity_stream.empty')
                    : statusKey === 'completed'
                      ? pageChat('activity_stream.completed_empty' as const)
                      : pageChat('activity_stream.pending_empty' as const)}
                </div>
              )}
              {grouped.tool.map((part) => (
                <ToolActivityItem key={part.toolCallId} part={part} />
              ))}
              {grouped.consent.map((part) =>
                ConsentSlot ? (
                  <ConsentSlot
                    key={part.id}
                    chatId={chatId}
                    turnId={turn.turn_id}
                    part={part}
                  />
                ) : (
                  <ConsentPlaceholder key={part.id} part={part} />
                ),
              )}
              {grouped.elicitation.map((part) =>
                ElicitationSlot ? (
                  <ElicitationSlot
                    key={part.id}
                    chatId={chatId}
                    turnId={turn.turn_id}
                    part={part}
                  />
                ) : (
                  <ElicitationPlaceholder key={part.id} part={part} />
                ),
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>

        {showAnswerSection &&
          (status === 'completed' ? (
            <div className="text-[15px] leading-[1.65] tracking-[-0.003em]">
              {answerText ? (
                <Markdown>{answerText}</Markdown>
              ) : (
                <div className="text-muted-foreground text-sm">
                  {pageChat('answer_section.completed_empty' as const)}
                </div>
              )}
            </div>
          ) : (
            <Card
              className={cn(
                'gap-0 overflow-hidden rounded-xl py-0',
                status === 'failed' || status === 'cancelled' || status === 'aborted'
                  ? 'border-destructive/20 bg-destructive/5 shadow-none'
                  : 'border-border/60 bg-background/80',
              )}
            >
              <CardContent className="px-4 py-4 text-sm">
                <div className="text-muted-foreground font-mono mb-2 text-[10.5px] tracking-[0.08em] uppercase">
                  {pageChat(
                    answerSectionTitleKey(status, Boolean(answerText)) as never,
                    {} as never,
                  )}
                </div>
                {answerText ? (
                  <Markdown>{answerText}</Markdown>
                ) : pending ? (
                  <div className="space-y-2">
                    <div className="text-muted-foreground text-sm">
                      {pageChat('answer_section.pending_empty')}
                    </div>
                    <div className="flex flex-row gap-2 py-1">
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0" />
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200" />
                      <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400" />
                    </div>
                  </div>
                ) : (
                  <div className="text-muted-foreground text-sm">
                    {pageChat(emptyAnswerStateKey(status) as never, {} as never)}
                  </div>
                )}
                {errorText && (
                  <div className="text-destructive mt-3 flex items-start gap-2 text-[12px]">
                    <CornerDownRight className="mt-0.5 size-3.5 flex-none" />
                    <span className="break-words">{errorText}</span>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}

        {hasTurnDebugDetails && (
          <Collapsible className="group/details">
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="text-muted-foreground/80 hover:text-foreground flex items-center gap-1.5 text-left text-[11px] transition-colors"
              >
                <ChevronRight className="size-3 transition-transform group-data-[state=open]/details:rotate-90" />
                <span>{pageChat('activity_stream.debug.title')}</span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-1.5">
              <div className="border-border/60 bg-background/60 grid gap-2 rounded-md border px-3 py-2 text-[11px]">
                <div className="grid gap-0.5">
                  <div className="text-muted-foreground/80">
                    {pageChat('activity_stream.debug.turn_id')}
                  </div>
                  <div className="font-mono break-all">{turn.turn_id}</div>
                </div>
                <div className="grid gap-0.5">
                  <div className="text-muted-foreground/80">
                    {pageChat('activity_stream.debug.request_id')}
                  </div>
                  <div className="font-mono break-all">{turn.request_id}</div>
                </div>
                <div className="grid gap-0.5">
                  <div className="text-muted-foreground/80">
                    {pageChat('activity_stream.debug.status')}
                  </div>
                  <div>{statusKey}</div>
                </div>
                {turn.error_code && (
                  <div className="grid gap-0.5">
                    <div className="text-muted-foreground/80">
                      {pageChat('activity_stream.debug.error_code')}
                    </div>
                    <div className="font-mono">{turn.error_code}</div>
                  </div>
                )}
                {(errorText || turn.error_message) && (
                  <div className="grid gap-0.5">
                    <div className="text-muted-foreground/80">
                      {pageChat('activity_stream.debug.error_message')}
                    </div>
                    <div className="break-all">
                      {errorText || turn.error_message}
                    </div>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        <div className="flex flex-row items-center gap-1">
          {showReferences && (
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground h-7 cursor-pointer px-2 text-[12px]"
                >
                  <Badge
                    className="bg-accent-soft text-accent-ink mr-1.5 h-4 min-w-4 rounded-sm px-1 font-mono text-[10px] tabular-nums"
                    variant="outline"
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
                      key={`${turn.turn_id}-ref-${index}`}
                      defaultOpen={index < 3}
                      title={
                        <div className="flex items-center justify-between gap-2">
                          <div className="truncate">
                            {index + 1}.{' '}
                            {String(
                              reference.metadata?.title ||
                                reference.metadata?.uri ||
                                reference.text ||
                                'Reference',
                            )}
                          </div>
                          {typeof reference.score === 'number' && (
                            <div className="text-muted-foreground font-mono text-xs tabular-nums">
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
            turnId={turn.turn_id}
            feedback={feedback}
            onFeedback={onFeedback}
          />

          {copyText && (
            <CopyToClipboard
              variant="ghost"
              className="text-muted-foreground hover:text-foreground h-7 px-2"
              text={copyText}
            />
          )}
        </div>
      </div>
    </div>
  );

}

function answerSectionTitleKey(
  status: AgentStreamStatus,
  hasAnswerText: boolean,
): string {
  if (status === 'failed') {
    return hasAnswerText
      ? 'answer_section.failure_details'
      : 'answer_section.run_failed';
  }
  if (status === 'cancelled' || status === 'aborted') {
    return hasAnswerText
      ? 'answer_section.cancelled_output'
      : 'answer_section.run_cancelled';
  }
  if (status === 'completed') {
    return 'answer_section.final_answer';
  }
  return hasAnswerText
    ? 'answer_section.draft_answer'
    : 'answer_section.answer';
}

function emptyAnswerStateKey(status: AgentStreamStatus): string {
  if (status === 'failed') return 'answer_section.failed_empty';
  if (status === 'cancelled' || status === 'aborted') {
    return 'answer_section.cancelled_empty';
  }
  return 'answer_section.pending_empty';
}

function previewJson(value: unknown, maxLength: number): string | undefined {
  if (value == null) return undefined;
  let raw: string;
  if (typeof value === 'string') {
    raw = value;
  } else {
    try {
      raw = JSON.stringify(value, null, 2);
    } catch {
      raw = String(value);
    }
  }
  const normalized = raw.trim();
  if (!normalized) return undefined;
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}
