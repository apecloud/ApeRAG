'use client';

// task #61 P1-D3 (PR for #87): read-only display of the deployment
// vector backend identity + static capability matrix.
//
// Important contract (per architect msg=0044261f + dongdong msg=c2593fdd):
// `vector_backend` is an OUTPUT projection on the Collection detail
// schema, NOT an input field. The BE projects it from
// `settings.vector_db_type` (deployment-wide), so every collection in a
// given deployment shows the same identity + capability matrix. There
// is no per-collection override knob; the display below is intentionally
// not editable. If a future task introduces a per-collection vector
// backend choice, the projection helper would gain a fallback path on
// the BE; the FE display below would not need to change because it
// already binds to the projected `vector_backend` field.
//
// dongdong picks up rendering polish (responsive + dark mode + final
// copy) on the same PR per the joint A4-style split (cuiwenbo contract
// layer + dongdong rendering polish + CR pair).

import { useCollectionContext } from '@/components/providers/collection-provider';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import type {
  VectorBackendCapabilities,
  VectorBackendInfo,
  VectorBackendType,
} from '@/features/collection/types';
import { Check, Database, X } from 'lucide-react';
import { useTranslations } from 'next-intl';

const BACKEND_LABEL: Record<VectorBackendType, string> = {
  pgvector: 'PGVector',
  qdrant: 'Qdrant',
};

type CapabilityKey = keyof VectorBackendCapabilities;

const CAPABILITY_LABEL_KEYS = {
  supports_atomic_batch_upsert: 'vector_backend_capability_atomic_batch_upsert',
  supports_filter_or_with_empty_parts:
    'vector_backend_capability_filter_or_empty_parts',
  supports_legacy_mode: 'vector_backend_capability_legacy_mode',
} as const satisfies Record<CapabilityKey, string>;

const CAPABILITY_DESCRIPTION_KEYS = {
  supports_atomic_batch_upsert:
    'vector_backend_capability_atomic_batch_upsert_description',
  supports_filter_or_with_empty_parts:
    'vector_backend_capability_filter_or_empty_parts_description',
  supports_legacy_mode: 'vector_backend_capability_legacy_mode_description',
} as const satisfies Record<CapabilityKey, string>;

const CAPABILITY_ORDER: CapabilityKey[] = [
  'supports_atomic_batch_upsert',
  'supports_filter_or_with_empty_parts',
  'supports_legacy_mode',
];

const CapabilityRow = ({
  label,
  description,
  supported,
  supportedLabel,
  unsupportedLabel,
}: {
  label: string;
  description: string;
  supported: boolean;
  supportedLabel: string;
  unsupportedLabel: string;
}) => (
  <div className="border-border/70 flex flex-row items-start justify-between gap-3 border-t py-2 first:border-t-0">
    <div className="min-w-0">
      <div className="text-sm font-medium">{label}</div>
      <div className="text-muted-foreground mt-0.5 text-xs">{description}</div>
    </div>
    <div className="shrink-0">
      {supported ? (
        <Badge
          variant="outline"
          className="border-green-300 bg-green-50 text-green-700"
        >
          <Check className="size-3.5" />
          <span className="ml-1">{supportedLabel}</span>
        </Badge>
      ) : (
        <Badge
          variant="outline"
          className="text-muted-foreground border-muted bg-muted"
        >
          <X className="size-3.5" />
          <span className="ml-1">{unsupportedLabel}</span>
        </Badge>
      )}
    </div>
  </div>
);

export const CollectionVectorBackendCard = () => {
  const { collection } = useCollectionContext();
  const page_collections = useTranslations('page_collections');
  // ``Collection`` is the legacy permissive shape (Wave 8 pre-rename),
  // so ``vector_backend`` is typed loose; do a runtime narrow before
  // rendering. ``null`` means the BE could not project a static
  // capability matrix (unknown deployment vector backend) — render the
  // placeholder so the user knows the field exists but is unmapped.
  const rawVectorBackend = (collection as { vector_backend?: unknown })
    .vector_backend;
  const vectorBackend =
    rawVectorBackend && typeof rawVectorBackend === 'object'
      ? (rawVectorBackend as VectorBackendInfo)
      : null;

  return (
    <Card className="border-border/70 gap-0 overflow-hidden rounded-xl py-0 shadow-sm">
      <CardHeader className="gap-3 p-5">
        <div className="flex flex-row items-center gap-2">
          <div className="bg-accent-soft text-accent-ink flex size-9 shrink-0 items-center justify-center rounded-lg">
            <Database className="size-4" />
          </div>
          <div className="min-w-0">
            <CardTitle className="text-base font-medium">
              {page_collections('vector_backend_card_title')}
            </CardTitle>
            <CardDescription className="mt-0.5 text-xs">
              {page_collections('vector_backend_card_description')}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="border-border/70 border-t p-5">
        {vectorBackend ? (
          <>
            <div className="flex flex-row items-center justify-between gap-3 pb-3">
              <div className="text-sm font-medium">
                {page_collections('vector_backend_identity')}
              </div>
              <Badge
                variant="outline"
                className="bg-accent-soft text-accent-ink border-accent-soft"
              >
                {BACKEND_LABEL[vectorBackend.type] ?? vectorBackend.type}
              </Badge>
            </div>
            <div className="border-border/70 border-t pt-2">
              {CAPABILITY_ORDER.map((capability) => (
                <CapabilityRow
                  key={capability}
                  label={page_collections(CAPABILITY_LABEL_KEYS[capability])}
                  description={page_collections(
                    CAPABILITY_DESCRIPTION_KEYS[capability],
                  )}
                  supported={Boolean(vectorBackend.capabilities[capability])}
                  supportedLabel={page_collections(
                    'vector_backend_capability_supported',
                  )}
                  unsupportedLabel={page_collections(
                    'vector_backend_capability_unsupported',
                  )}
                />
              ))}
            </div>
          </>
        ) : (
          <div className="text-muted-foreground py-3 text-center text-sm">
            {page_collections('vector_backend_unknown_placeholder')}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
