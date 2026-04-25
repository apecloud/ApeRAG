'use client';

// FE D8.4c (#78) -- Interactive elicitation form UI.
//
// Renders one `AgentElicitationPart`, generates form fields from the
// JSON-Schema fragment the BE attached, validates the user input
// against required-field presence + basic type coercion, and POSTs
// the response to `submitElicitation()` (POST
// `/agent/chats/{chat_id}/turns/{turn_id}/elicit/{elicitation_id}`)
// from the AI SDK-compatible client API landed by #76. Plugs into
// the renderer's `<ElicitationSlot>` shape (#77 PR #1703 head
// `b532abcd`) without changing slot props.
//
// D9 §5.1 contract surface (renderer-side):
// * `data.schema` is a JSON Schema; we render a minimal field set
//   the BE Pydantic validator can accept (string / number /
//   integer / boolean / enum, plus required-field gating).
// * Server-driven state machine: `pending -> answered | cancelled`.
//   We never derive the visible state purely from local optimism.
// * The BE re-validates on submit (per D9 §A4 #5 schema-validated
//   input). On 422 we leave the form populated so the user can
//   correct + retry; on 403/404/409 we surface a toast and disable
//   the form.

import { useCallback, useMemo, useState } from 'react';
import { CheckCircle2, HelpCircle, XCircle } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { submitElicitation } from '@/features/agent-runtime/api';
import type { AgentElicitationPart } from '@/features/agent-runtime/types';
import type { ElicitationSlotProps } from './agent-turn-renderer';

// Minimal JSON Schema property shape we recognise. Anything outside
// this set falls back to a free-form text input — the server-side
// Pydantic validator is the source of truth, so the FE form is just
// a UX accelerator, not the gate.
type JsonSchemaProperty = {
  type?: 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array';
  title?: string;
  description?: string;
  enum?: ReadonlyArray<string | number | boolean>;
  default?: unknown;
  format?: string;
};

type JsonSchemaObject = {
  type?: 'object';
  required?: string[];
  properties?: Record<string, JsonSchemaProperty>;
};

function describeError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return 'elicitation submit failed';
}

function coerceFieldValue(
  prop: JsonSchemaProperty,
  raw: unknown,
): { value: unknown; ok: boolean } {
  // Pydantic on the BE will re-coerce, but we do best-effort coercion
  // so the wire payload is shaped right (a `number` field arrives as
  // an actual number, not a string).
  if (prop.enum && prop.enum.length > 0) {
    if (raw === '' || raw == null) return { value: undefined, ok: true };
    return { value: raw, ok: true };
  }
  switch (prop.type) {
    case 'boolean':
      return { value: Boolean(raw), ok: true };
    case 'integer':
    case 'number': {
      if (raw === '' || raw == null) return { value: undefined, ok: true };
      const n =
        prop.type === 'integer' ? Number.parseInt(String(raw), 10) : Number(raw);
      return Number.isFinite(n) ? { value: n, ok: true } : { value: raw, ok: false };
    }
    case 'string':
    default:
      return { value: raw == null ? '' : String(raw), ok: true };
  }
}

function buildResponse(
  schema: JsonSchemaObject | undefined,
  formState: Record<string, unknown>,
): { response: Record<string, unknown>; missing: string[]; invalid: string[] } {
  const required = schema?.required ?? [];
  const properties = schema?.properties ?? {};
  const response: Record<string, unknown> = {};
  const missing: string[] = [];
  const invalid: string[] = [];

  // Coerce + collect typed values.
  for (const [field, raw] of Object.entries(formState)) {
    const prop = properties[field] ?? {};
    const { value, ok } = coerceFieldValue(prop, raw);
    if (!ok) {
      invalid.push(field);
      continue;
    }
    if (value === undefined || value === '') continue;
    response[field] = value;
  }

  // Required-field gate: the BE default validator already does this
  // (per `tools/elicitation.py::_required_fields_validator`), but the
  // FE check avoids a 422 round-trip on the first attempt.
  for (const field of required) {
    const present = response[field] !== undefined && response[field] !== '';
    const declaredAsBoolean =
      properties[field]?.type === 'boolean' && response[field] !== undefined;
    if (!present && !declaredAsBoolean) missing.push(field);
  }

  return { response, missing, invalid };
}

export function ElicitationForm({ chatId, turnId, part }: ElicitationSlotProps) {
  const data = part.data;
  const state = data.state;
  const schema = (data.schema ?? {}) as JsonSchemaObject;
  const properties = schema.properties ?? {};
  const required = schema.required ?? [];

  // Initialize form state from schema defaults so the user starts
  // with sensible values (especially for `default: false` on
  // checkboxes, which would otherwise read as undefined).
  const initialState = useMemo(() => {
    const init: Record<string, unknown> = {};
    for (const [field, prop] of Object.entries(properties)) {
      if (prop.default !== undefined) init[field] = prop.default;
      else if (prop.type === 'boolean') init[field] = false;
    }
    return init;
  }, [properties]);

  const [formState, setFormState] = useState<Record<string, unknown>>(initialState);
  const [submitting, setSubmitting] = useState(false);

  const setField = useCallback(
    (field: string, value: unknown) =>
      setFormState((prev) => ({ ...prev, [field]: value })),
    [],
  );

  const onSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (state !== 'pending' || submitting) return;
      const { response, missing, invalid } = buildResponse(schema, formState);
      if (invalid.length > 0) {
        toast.error(`Invalid value for: ${invalid.join(', ')}`);
        return;
      }
      if (missing.length > 0) {
        toast.error(`Missing required fields: ${missing.join(', ')}`);
        return;
      }
      setSubmitting(true);
      try {
        await submitElicitation(chatId, turnId, data.elicitationId, response);
      } catch (err) {
        toast.error(describeError(err));
      } finally {
        setSubmitting(false);
      }
    },
    [chatId, turnId, data.elicitationId, schema, formState, state, submitting],
  );

  if (state !== 'pending') {
    return <ElicitationResolvedRow part={part} />;
  }

  const fieldEntries = Object.entries(properties);

  return (
    <form
      onSubmit={onSubmit}
      className="border-primary/40 bg-accent-soft text-accent-ink space-y-3 rounded-md border px-3 py-2 text-[13px]"
    >
      <div className="flex items-start gap-2">
        <HelpCircle className="text-primary mt-0.5 size-4 flex-none" />
        <div className="min-w-0 flex-1 space-y-1">
          <div className="font-medium">{data.prompt}</div>
          {data.serverName ? (
            <div className="text-accent-ink/70 text-[11px]">
              from {data.serverName}
            </div>
          ) : null}
        </div>
      </div>

      {fieldEntries.length === 0 ? (
        <div className="text-accent-ink/70 text-[12px]">
          no schema fields declared; submit empty response
        </div>
      ) : (
        <div className="space-y-2">
          {fieldEntries.map(([field, prop]) => (
            <ElicitationField
              key={field}
              name={field}
              prop={prop}
              required={required.includes(field)}
              value={formState[field]}
              disabled={submitting}
              onChange={(v) => setField(field, v)}
            />
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <Button size="sm" type="submit" disabled={submitting}>
          <CheckCircle2 className="mr-1 size-4" />
          {submitting ? 'Submitting…' : 'Submit'}
        </Button>
      </div>
    </form>
  );
}

type ElicitationFieldProps = {
  name: string;
  prop: JsonSchemaProperty;
  required: boolean;
  value: unknown;
  disabled: boolean;
  onChange: (value: unknown) => void;
};

function ElicitationField({
  name,
  prop,
  required,
  value,
  disabled,
  onChange,
}: ElicitationFieldProps) {
  const label = prop.title ?? name;
  const id = `elicit-${name}`;

  if (prop.enum && prop.enum.length > 0) {
    return (
      <div className="space-y-1">
        <FieldLabel htmlFor={id} label={label} required={required} description={prop.description} />
        <Select
          value={value == null ? '' : String(value)}
          onValueChange={(v) => onChange(v)}
          disabled={disabled}
        >
          <SelectTrigger id={id} className="w-full">
            <SelectValue placeholder="Select…" />
          </SelectTrigger>
          <SelectContent>
            {prop.enum.map((opt) => (
              <SelectItem key={String(opt)} value={String(opt)}>
                {String(opt)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    );
  }

  if (prop.type === 'boolean') {
    return (
      <div className="flex items-start gap-2">
        <Checkbox
          id={id}
          checked={Boolean(value)}
          onCheckedChange={(checked) => onChange(checked === true)}
          disabled={disabled}
        />
        <div className="min-w-0 flex-1">
          <FieldLabel htmlFor={id} label={label} required={required} description={prop.description} inline />
        </div>
      </div>
    );
  }

  // String / number / integer / fallback all use a single-line input
  // unless the schema asks for a multi-line `textarea` format.
  const isMultiline = prop.type === 'string' && prop.format === 'textarea';
  const inputType =
    prop.type === 'integer' || prop.type === 'number' ? 'number' : 'text';

  return (
    <div className="space-y-1">
      <FieldLabel htmlFor={id} label={label} required={required} description={prop.description} />
      {isMultiline ? (
        <Textarea
          id={id}
          value={value == null ? '' : String(value)}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          rows={4}
        />
      ) : (
        <Input
          id={id}
          type={inputType}
          value={value == null ? '' : String(value)}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
        />
      )}
    </div>
  );
}

function FieldLabel({
  htmlFor,
  label,
  required,
  description,
  inline = false,
}: {
  htmlFor: string;
  label: string;
  required: boolean;
  description?: string;
  inline?: boolean;
}) {
  return (
    <div>
      <Label htmlFor={htmlFor} className={inline ? 'text-[13px]' : 'text-[12px]'}>
        {label}
        {required ? <span className="text-destructive ml-0.5">*</span> : null}
      </Label>
      {description ? (
        <div className="text-accent-ink/70 text-[11px]">{description}</div>
      ) : null}
    </div>
  );
}

function ElicitationResolvedRow({ part }: { part: AgentElicitationPart }) {
  const data = part.data;
  const state = data.state;
  const tone =
    state === 'answered'
      ? 'text-emerald-600 dark:text-emerald-400'
      : 'text-muted-foreground';
  const Icon = state === 'answered' ? CheckCircle2 : XCircle;
  return (
    <div className="border-border/60 bg-background/60 flex items-start gap-2 rounded-md border px-3 py-2 text-[12px]">
      <Icon className={`mt-0.5 size-3.5 flex-none ${tone}`} />
      <div className="min-w-0 flex-1">
        <div className="font-medium">{data.prompt}</div>
        <div className={`mt-0.5 break-all ${tone}`}>elicitation {state}</div>
      </div>
    </div>
  );
}
