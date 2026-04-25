'use client';

// FE D8.4c (#78) -- Interactive consent prompt UI.
//
// Renders one `AgentToolConsentPart` and lets the user approve / deny
// the tool call. Wires to `decideToolConsent()` (POST
// `/agent/chats/{chat_id}/turns/{turn_id}/consent/{tool_call_id}`)
// from the AI SDK-compatible client API landed by #76. The
// `<ConsentSlot>` placeholder shape is owned by the renderer (#77,
// PR #1703) and we plug into it without changing the slot props.
//
// D9 §3 + §A7 contract surface (renderer-side):
// * Show only `toolName + argsPreview + risk` -- raw args never reach
//   the FE (BE-side `args_preview()` redacts before emit per #75).
// * Server-driven state machine: `pending -> approved | denied |
//   expired` arrives via the part's `data.state`; we never derive
//   the visible state purely from local optimism.
// * After a successful decide call we still rely on the next streamed
//   `data-tool-consent` part (state="approved"/"denied") to flip the
//   UI; local "submitting" state is just to disable the buttons
//   between click + server ack.
// * All visible text + toast strings go through `useTranslations`
//   (zh-CN / en-US catalogs) per #77 i18n baseline; only dynamic
//   identifiers (toolName, argsPreview, argsHash) stay verbatim.

import { useCallback, useState } from 'react';
import { CheckCircle2, HandCoins, ShieldAlert, XCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { decideToolConsent } from '@/features/agent-runtime/api';
import type {
  AgentToolConsentPart,
  ToolConsentRisk,
} from '@/features/agent-runtime/types';
import type { ConsentSlotProps } from './agent-turn-renderer';

// Risk -> badge tone mapping. The user-facing label comes from i18n
// catalog (`activity_stream.consent.risk.<key>`); we default to
// destructive for system-modifying / admin so the prompt never
// under-sells the consequences of a side-effecting tool call.
const RISK_TONE: Record<
  ToolConsentRisk,
  'default' | 'secondary' | 'destructive'
> = {
  writes_user_data: 'default',
  calls_external_api: 'default',
  modifies_system: 'destructive',
  admin_only: 'destructive',
};

type Decision = 'approved' | 'denied';

export function ConsentPrompt({ chatId, turnId, part }: ConsentSlotProps) {
  const pageChat = useTranslations('page_chat');
  const [submitting, setSubmitting] = useState<Decision | null>(null);

  const data = part.data;
  const state = data.state;
  const tone = RISK_TONE[data.risk] ?? 'default';
  const riskLabel =
    pageChat(`activity_stream.consent.risk.${data.risk}` as const) || data.risk;

  const onDecide = useCallback(
    async (decision: Decision) => {
      if (state !== 'pending' || submitting !== null) return;
      setSubmitting(decision);
      try {
        await decideToolConsent(chatId, turnId, data.toolCallId, decision);
        // Server will emit a follow-up `data-tool-consent` part with
        // the resolved state; we don't optimistically mutate here.
      } catch (err) {
        toast.error(
          err instanceof Error && err.message
            ? err.message
            : pageChat('activity_stream.consent.decision_failed'),
        );
        setSubmitting(null);
      }
    },
    [chatId, turnId, data.toolCallId, state, submitting, pageChat],
  );

  if (state !== 'pending') {
    return <ConsentResolvedRow part={part} />;
  }

  return (
    <div className="border-primary/40 bg-accent-soft text-accent-ink rounded-md border px-3 py-2 text-[13px]">
      <div className="flex items-start gap-2">
        <HandCoins className="text-primary mt-0.5 size-4 flex-none" />
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{data.toolName}</span>
            <Badge variant={tone} className="text-[11px]">
              <ShieldAlert className="mr-1 size-3" />
              {riskLabel}
            </Badge>
          </div>
          {data.argsPreview ? (
            <pre className="bg-background/60 max-h-40 overflow-auto rounded border border-border/60 px-2 py-1.5 text-[12px] leading-snug whitespace-pre-wrap break-all">
              {data.argsPreview}
            </pre>
          ) : null}
          <div className="text-accent-ink/70 text-[11px]">
            {pageChat('activity_stream.consent.args_fingerprint')}:{' '}
            <span className="font-mono">{data.argsHash.slice(0, 12)}…</span>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button
              size="sm"
              onClick={() => onDecide('approved')}
              disabled={submitting !== null}
            >
              <CheckCircle2 className="mr-1 size-4" />
              {submitting === 'approved'
                ? pageChat('activity_stream.consent.approving')
                : pageChat('activity_stream.consent.approve')}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onDecide('denied')}
              disabled={submitting !== null}
            >
              <XCircle className="mr-1 size-4" />
              {submitting === 'denied'
                ? pageChat('activity_stream.consent.denying')
                : pageChat('activity_stream.consent.deny')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConsentResolvedRow({ part }: { part: AgentToolConsentPart }) {
  const pageChat = useTranslations('page_chat');
  const data = part.data;
  const state = data.state;
  const tone =
    state === 'approved'
      ? 'text-emerald-600 dark:text-emerald-400'
      : state === 'denied'
        ? 'text-destructive'
        : 'text-muted-foreground';
  const Icon =
    state === 'approved' ? CheckCircle2 : state === 'denied' ? XCircle : HandCoins;
  // `state` is narrowed to non-'pending' by the caller (the parent
  // component returns this row only when state !== 'pending'), so the
  // catalog lookup below is always valid -- but TS can't see that
  // narrowing across the function boundary, so we fall back to the
  // raw state literal for any future state additions the catalog
  // doesn't yet cover.
  const stateLabel =
    state === 'approved' || state === 'denied' || state === 'expired'
      ? pageChat(`activity_stream.consent.state_label.${state}` as const)
      : state;
  return (
    <div className="border-border/60 bg-background/60 flex items-start gap-2 rounded-md border px-3 py-2 text-[12px]">
      <Icon className={`mt-0.5 size-3.5 flex-none ${tone}`} />
      <div className="min-w-0 flex-1">
        <div className="font-medium">{data.toolName}</div>
        <div className={`mt-0.5 break-all ${tone}`}>
          {pageChat('activity_stream.consent.resolved_status', {
            state: stateLabel,
          })}
        </div>
      </div>
    </div>
  );
}
