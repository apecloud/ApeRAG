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
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
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
import {
  type ActivityData,
  type AgentCitationPart,
  type AgentElicitationPart,
  type AgentMessagePart,
  type AgentReasoningPart,
  type AgentSourceDocumentPart,
  type AgentSourceUrlPart,
  type AgentStreamStatus,
  type AgentTextPart,
  type AgentToolConsentPart,
  type AgentToolPart,
  type AgentTurnEnvelope,
  type CitationLocation,
} from '@/features/agent-runtime';
import type { Feedback, Reference } from '@/features/bot/types';
import { cn } from '@/lib/utils';
import {
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CornerDownRight,
  Globe2,
  HandCoins,
  HelpCircle,
  LoaderCircle,
  Search,
  Sparkles,
  XCircle,
} from 'lucide-react';
import { useFormatter, useTranslations } from 'next-intl';
import { useEffect, useMemo, useRef, useState } from 'react';
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
  const reasoning: AgentReasoningPart[] = [];
  const tool: AgentToolPart[] = [];
  const sourceUrl: AgentSourceUrlPart[] = [];
  const sourceDoc: AgentSourceDocumentPart[] = [];
  const citation: AgentCitationPart[] = [];
  const consent: AgentToolConsentPart[] = [];
  const elicitation: AgentElicitationPart[] = [];
  for (const part of parts) {
    if (part.type === 'text') text.push(part);
    else if (part.type === 'reasoning') reasoning.push(part);
    else if (part.type === 'source-url') sourceUrl.push(part);
    else if (part.type === 'source-document') sourceDoc.push(part);
    else if (part.type === 'data-citation') citation.push(part);
    else if (part.type === 'data-tool-consent') consent.push(part);
    else if (part.type === 'data-elicitation') elicitation.push(part);
    else if (part.type.startsWith('tool-')) tool.push(part as AgentToolPart);
  }
  return {
    text,
    reasoning,
    tool,
    sourceUrl,
    sourceDoc,
    citation,
    consent,
    elicitation,
  };
}

function joinTextParts(parts: AgentTextPart[]): string {
  return parts
    .map((p) => p.text)
    .filter(Boolean)
    .join('\n\n');
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

type ToolBehaviorKind =
  | 'web_search'
  | 'web_read'
  | 'knowledge_search'
  | 'document_read'
  | 'generic';

type ToolBehaviorPhase = 'running' | 'done' | 'error';

function toolDisplayName(part: AgentToolPart): string {
  const metadataToolName =
    typeof part.metadata?.mcpToolName === 'string'
      ? part.metadata.mcpToolName
      : undefined;
  return (
    part.toolName ||
    metadataToolName ||
    part.type.replace(/^tool-/, '') ||
    'tool'
  );
}

function toolBehaviorKind(part: AgentToolPart): ToolBehaviorKind {
  const name = toolDisplayName(part).toLowerCase();
  if (name.includes('web_search') || name.includes('search_web')) {
    return 'web_search';
  }
  if (
    name.includes('web_read') ||
    name.includes('read_web') ||
    name.includes('fetch_url') ||
    name.includes('read_url')
  ) {
    return 'web_read';
  }
  if (
    name.includes('read_document') ||
    name.includes('document_read') ||
    name.includes('read_section') ||
    name.includes('read_chunk')
  ) {
    return 'document_read';
  }
  if (
    name.includes('knowledge') ||
    name.includes('collection') ||
    name.includes('vector') ||
    name.includes('fulltext') ||
    name.includes('hybrid') ||
    name.includes('graph') ||
    name.includes('search')
  ) {
    return 'knowledge_search';
  }
  return 'generic';
}

function toolBehaviorPhase(state: AgentToolPart['state']): ToolBehaviorPhase {
  if (state === 'output-error') return 'error';
  if (state === 'output-available') return 'done';
  return 'running';
}

function toolBehaviorIcon(kind: ToolBehaviorKind, phase: ToolBehaviorPhase) {
  if (phase === 'error') return XCircle;
  if (phase === 'done') return CheckCircle2;
  switch (kind) {
    case 'web_search':
    case 'knowledge_search':
      return Search;
    case 'web_read':
      return Globe2;
    case 'document_read':
      return BookOpen;
    default:
      return Sparkles;
  }
}

function extractStringField(
  value: unknown,
  keys: string[],
  depth = 3,
): string | undefined {
  if (Array.isArray(value)) {
    if (depth <= 0) return undefined;
    for (const item of value) {
      const found = extractStringField(item, keys, depth - 1);
      if (found) return found;
    }
    return undefined;
  }
  if (value == null || typeof value !== 'object') {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  for (const key of keys) {
    const candidate = record[key];
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate.trim();
    }
  }
  if (depth <= 0) return undefined;
  for (const candidate of Object.values(record)) {
    const found = extractStringField(candidate, keys, depth - 1);
    if (found) return found;
  }
  return undefined;
}

function compactUrlLabel(value: string): string {
  try {
    const url = new URL(value);
    const path = url.pathname === '/' ? '' : url.pathname;
    return `${url.hostname}${path}`.replace(/\/$/, '');
  } catch {
    return value;
  }
}

function toolBehaviorDetail(
  part: AgentToolPart,
  kind: ToolBehaviorKind,
  pageChat: ReturnType<typeof useTranslations<'page_chat'>>,
): string {
  if (part.summary?.trim()) return part.summary.trim();

  const display = extractStringField(part.metadata, ['display'], 1);
  if (display) return display;

  const query =
    extractStringField(part.metadata, ['query', 'q', 'keyword', 'keywords']) ||
    extractStringField(part.input, ['query', 'q', 'keyword', 'keywords']) ||
    extractStringField(part.output, ['query', 'q', 'keyword', 'keywords']);
  if (query) {
    return pageChat('activity_stream.tool.detail.search_query', { query });
  }

  const title =
    extractStringField(part.metadata, ['title', 'name']) ||
    extractStringField(part.input, ['title', 'name']) ||
    extractStringField(part.output, ['title', 'name']);
  if (title) {
    return pageChat('activity_stream.tool.detail.source_title', { title });
  }

  const url =
    extractStringField(part.metadata, ['url', 'uri', 'link']) ||
    extractStringField(part.input, ['url', 'uri', 'link']) ||
    extractStringField(part.output, ['url', 'uri', 'link']);
  if (url) {
    return pageChat('activity_stream.tool.detail.web_url', {
      url: compactUrlLabel(url),
    });
  }

  if (part.state === 'output-error') {
    return pageChat('activity_stream.tool.detail.error');
  }
  if (part.state === 'output-available') {
    return pageChat('activity_stream.tool.detail.done');
  }
  return pageChat(`activity_stream.tool.detail.${kind}` as never, {} as never);
}

function isVisibleToolActivity(part: AgentToolPart): boolean {
  const kind = toolBehaviorKind(part);
  return !(kind === 'generic' && part.state === 'output-available');
}

// ---------------------------------------------------------------------------

function ReasoningActivityItem({
  part,
  ordinal,
}: {
  part: AgentReasoningPart;
  ordinal: number;
}) {
  const pageChat = useTranslations('page_chat');
  const text = part.text.trim();
  if (!text) return null;
  const streaming = part.state === 'streaming';

  return (
    <div className="flex gap-2.5">
      <div className="flex pt-[3px]">
        <Sparkles
          className={cn(
            'size-3.5 flex-none',
            streaming ? 'text-primary animate-pulse' : 'text-muted-foreground/70',
          )}
        />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-[13px] leading-snug">
          <span className="text-foreground/80 font-medium">
            {pageChat('activity_stream.reasoning.title', {
              index: String(ordinal),
            })}
          </span>
          {streaming && (
            <span className="text-muted-foreground ml-2 text-[12px]">
              {pageChat('activity_stream.reasoning.streaming')}
            </span>
          )}
        </div>
        <div className="text-muted-foreground border-border/60 bg-background/70 rounded-md border px-3 py-2 text-[12.5px] leading-relaxed whitespace-pre-wrap">
          {text}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function ToolActivityItem({ part }: { part: AgentToolPart }) {
  const pageChat = useTranslations('page_chat');
  if (!isVisibleToolActivity(part)) return null;
  const kind = toolBehaviorKind(part);
  const phase = toolBehaviorPhase(part.state);
  const tone = toolStateTone(part.state);
  const Icon = toolBehaviorIcon(kind, phase);
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
              part.state === 'input-streaming' ||
                part.state === 'input-available'
                ? 'animate-pulse'
                : '',
            )}
          >
            {pageChat(
              `activity_stream.tool.behavior.${kind}.${phase}` as never,
              {} as never,
            )}
          </span>
          <span className={cn('break-words', tone.subtitle)}>
            {toolBehaviorDetail(part, kind, pageChat)}
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
                    <pre className="bg-background/80 border-border/40 overflow-x-auto rounded border px-2 py-1 break-all whitespace-pre-wrap">
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
                        'bg-background/80 border-border/40 overflow-x-auto rounded border px-2 py-1 break-all whitespace-pre-wrap',
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

function activityLabel(
  activity: ActivityData | null,
  pageChat: ReturnType<typeof useTranslations<'page_chat'>>,
): string | null {
  if (!activity) return null;
  const rawIntent = activity.activity?.intent || activity.intent;
  const intent: TransientIntent = isTransientIntent(rawIntent)
    ? rawIntent
    : 'thinking';
  return pageChat(`activity_stream.transient.${intent}` as const);
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
  const [activityOpen, setActivityOpen] = useState(true);
  const autoCollapsedRef = useRef(false);
  const visibleToolParts = useMemo(
    () => grouped.tool.filter(isVisibleToolActivity),
    [grouped.tool],
  );
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
  const activeToolPart =
    visibleToolParts.findLast(
      (part) => toolBehaviorPhase(part.state) === 'running',
    ) || visibleToolParts.at(-1);
  const headerStatusLabel =
    (pending && activityLabel(transientActivity, pageChat)) ||
    (pending && activeToolPart
      ? pageChat(
          `activity_stream.tool.behavior.${toolBehaviorKind(activeToolPart)}.running` as never,
          {} as never,
        )
      : pageChat(`activity_stream.status.${statusKey}` as never, {} as never));

  const timestamp = turn.finished_at || turn.started_at;
  const showAnswerSection =
    TERMINAL_STATUSES.has(status) ||
    ((status === 'failed' || status === 'cancelled' || status === 'aborted') &&
      Boolean(answerText));
  const showReferences =
    references.length > 0 && !(status === 'completed' && Boolean(answerText));
  const hasTurnDebugDetails =
    status === 'failed' ||
    Boolean(errorText || turn.error_code || turn.error_message);
  const copyText = answerText || errorText || turn.error_message || '';
  const traceMetaParts: string[] = [];
  if (grouped.reasoning.length > 0) {
    traceMetaParts.push(
      pageChat('activity_stream.meta.thoughts', {
        count: grouped.reasoning.length,
      }),
    );
  }
  if (visibleToolParts.length > 0) {
    traceMetaParts.push(
      pageChat('activity_stream.meta.steps', {
        count: visibleToolParts.length,
      }),
    );
  }
  if (references.length > 0) {
    traceMetaParts.push(
      pageChat('activity_stream.meta.sources', { count: references.length }),
    );
  }
  const traceMeta = traceMetaParts.join(' · ');

  const hasActivity =
    grouped.reasoning.length +
      visibleToolParts.length +
      grouped.consent.length +
      grouped.elicitation.length >
    0;

  useEffect(() => {
    if (autoCollapsedRef.current || !answerText.trim()) return;
    autoCollapsedRef.current = true;
    setActivityOpen(false);
  }, [answerText]);

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
      <div className="flex max-w-sm min-w-0 flex-1 flex-col gap-3 sm:max-w-lg md:max-w-2xl lg:max-w-3xl xl:max-w-4xl">
        {showHeaderStatus && (
          <div className="flex flex-row items-center gap-2">
            <Badge variant={statusTone} className="h-5 px-2 text-[10px]">
              {headerStatusLabel}
            </Badge>
            {timestamp && (
              <div className="text-muted-foreground font-mono text-[11px]">
                {format.dateTime(new Date(timestamp), 'medium')}
              </div>
            )}
          </div>
        )}

        <Collapsible
          open={activityOpen}
          onOpenChange={setActivityOpen}
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
              {parts.map((part, index) => {
                if (part.type === 'text') {
                  return null;
                }
                if (part.type === 'reasoning') {
                  const ordinal =
                    parts
                      .slice(0, index + 1)
                      .filter((item) => item.type === 'reasoning').length || 1;
                  return (
                    <ReasoningActivityItem
                      key={`reasoning-${part.id ?? index}`}
                      part={part}
                      ordinal={ordinal}
                    />
                  );
                }
                if (part.type.startsWith('tool-')) {
                  return (
                    <ToolActivityItem
                      key={`tool-${(part as AgentToolPart).toolCallId}-${index}`}
                      part={part as AgentToolPart}
                    />
                  );
                }
                if (part.type === 'data-tool-consent') {
                  return ConsentSlot ? (
                    <ConsentSlot
                      key={`consent-${part.id}`}
                      chatId={chatId}
                      turnId={turn.turn_id}
                      part={part}
                    />
                  ) : (
                    <ConsentPlaceholder
                      key={`consent-${part.id}`}
                      part={part}
                    />
                  );
                }
                if (part.type === 'data-elicitation') {
                  return ElicitationSlot ? (
                    <ElicitationSlot
                      key={`elicitation-${part.id}`}
                      chatId={chatId}
                      turnId={turn.turn_id}
                      part={part}
                    />
                  ) : (
                    <ElicitationPlaceholder
                      key={`elicitation-${part.id}`}
                      part={part}
                    />
                  );
                }
                return null;
              })}
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
                status === 'failed' ||
                  status === 'cancelled' ||
                  status === 'aborted'
                  ? 'border-destructive/20 bg-destructive/5 shadow-none'
                  : 'border-border/60 bg-background/80',
              )}
            >
              <CardContent className="px-4 py-4 text-sm">
                <div className="text-muted-foreground mb-2 font-mono text-[10.5px] tracking-[0.08em] uppercase">
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
                    {pageChat(
                      emptyAnswerStateKey(status) as never,
                      {} as never,
                    )}
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
