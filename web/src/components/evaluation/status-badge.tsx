'use client';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { useTranslations } from 'next-intl';

const toneByStatus: Record<string, string> = {
  queued: 'border-slate-300 bg-slate-100 text-slate-700',
  pending: 'border-slate-300 bg-slate-100 text-slate-700',
  running: 'border-sky-300 bg-sky-100 text-sky-800',
  completed: 'border-emerald-300 bg-emerald-100 text-emerald-800',
  failed: 'border-rose-300 bg-rose-100 text-rose-800',
  cancelled: 'border-amber-300 bg-amber-100 text-amber-800',
};

const fallbackLabel = (status?: string) => {
  if (!status) return '--';
  return status
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
};

const getStatusLabel = (
  status: string | undefined,
  t: ReturnType<typeof useTranslations>,
) => {
  if (!status) return '--';

  const key = `status.${status.toLowerCase()}`;
  try {
    return t(key as never);
  } catch {
    return fallbackLabel(status);
  }
};

export const EvaluationStatusBadge = ({ status }: { status?: string }) => {
  const t = useTranslations('page_bot_evaluation');
  return (
    <Badge
      variant="outline"
      className={cn(
        'border px-2.5 py-1 text-xs font-medium',
        toneByStatus[status?.toLowerCase() || 'queued'],
      )}
    >
      {getStatusLabel(status, t)}
    </Badge>
  );
};
