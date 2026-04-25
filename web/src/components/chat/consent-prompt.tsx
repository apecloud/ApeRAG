'use client';

// FE D8.4c (#78) -- Interactive consent prompt UI.
//
// Renders one `AgentToolConsentPart` and lets the user approve / deny
// the tool call. Wires to `decideToolConsent()` (POST
// `/agent/chats/{chat_id}/turns/{turn_id}/consent/{tool_call_id}`)
// from the AI SDK-compatible client API landed by #76. The
// `<ConsentSlot>` placeholder shape is owned by the renderer (#77,
// PR #1703 head `b532abcd`) and we plug into it without changing
// the slot props.
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

import { useCallback, useState } from 'react';
import { CheckCircle2, HandCoins, ShieldAlert, XCircle } from 'lucide-react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { decideToolConsent } from '@/features/agent-runtime/api';
import type {
  AgentToolConsentPart,
  ToolConsentRisk,
} from '@/features/agent-runtime/types';
import type { ConsentSlotProps } from './agent-turn-renderer';

// Risk -> badge tone mapping. We default to destructive for anything
// system-modifying / admin so the prompt never under-sells the
// consequences of a side-effecting tool call.
const RISK_TONE: Record<
  ToolConsentRisk,
  { label: string; tone: 'default' | 'secondary' | 'destructive' }
> = {
  writes_user_data: { label: 'Writes user data', tone: 'default' },
  calls_external_api: { label: 'Calls external API', tone: 'default' },
  modifies_system: { label: 'Modifies system', tone: 'destructive' },
  admin_only: { label: 'Admin-only action', tone: 'destructive' },
};

type Decision = 'approved' | 'denied';

function describeError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return 'consent decision failed';
}

export function ConsentPrompt({ chatId, turnId, part }: ConsentSlotProps) {
  const [submitting, setSubmitting] = useState<Decision | null>(null);

  const data = part.data;
  const state = data.state;
  const risk = RISK_TONE[data.risk] ?? {
    label: data.risk,
    tone: 'default' as const,
  };

  const onDecide = useCallback(
    async (decision: Decision) => {
      if (state !== 'pending' || submitting !== null) return;
      setSubmitting(decision);
      try {
        await decideToolConsent(chatId, turnId, data.toolCallId, decision);
        // Server will emit a follow-up `data-tool-consent` part with
        // the resolved state; we don't optimistically mutate here.
      } catch (err) {
        toast.error(describeError(err));
        setSubmitting(null);
      }
    },
    [chatId, turnId, data.toolCallId, state, submitting],
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
            <Badge variant={risk.tone} className="text-[11px]">
              <ShieldAlert className="mr-1 size-3" />
              {risk.label}
            </Badge>
          </div>
          {data.argsPreview ? (
            <pre className="bg-background/60 max-h-40 overflow-auto rounded border border-border/60 px-2 py-1.5 text-[12px] leading-snug whitespace-pre-wrap break-all">
              {data.argsPreview}
            </pre>
          ) : null}
          <div className="text-accent-ink/70 text-[11px]">
            args fingerprint:{' '}
            <span className="font-mono">{data.argsHash.slice(0, 12)}…</span>
          </div>
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button
              size="sm"
              onClick={() => onDecide('approved')}
              disabled={submitting !== null}
            >
              <CheckCircle2 className="mr-1 size-4" />
              {submitting === 'approved' ? 'Approving…' : 'Approve'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onDecide('denied')}
              disabled={submitting !== null}
            >
              <XCircle className="mr-1 size-4" />
              {submitting === 'denied' ? 'Denying…' : 'Deny'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConsentResolvedRow({ part }: { part: AgentToolConsentPart }) {
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
  return (
    <div className="border-border/60 bg-background/60 flex items-start gap-2 rounded-md border px-3 py-2 text-[12px]">
      <Icon className={`mt-0.5 size-3.5 flex-none ${tone}`} />
      <div className="min-w-0 flex-1">
        <div className="font-medium">{data.toolName}</div>
        <div className={`mt-0.5 break-all ${tone}`}>
          consent {state}
        </div>
      </div>
    </div>
  );
}
