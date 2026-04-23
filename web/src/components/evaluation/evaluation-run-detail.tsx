'use client';

import { useEffect, useMemo, useState, useTransition } from 'react';

import { FormatDate } from '@/components/format-date';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import {
  cancelEvaluationRun,
  retryEvaluationRunItem,
} from '@/features/evaluation/client-api';
import type {
  EvaluationRunDetailResponse,
  EvaluationRunItem,
} from '@/features/evaluation/types';
import { EvaluationApiNotice } from './api-notice';
import { EvaluationEmptyState } from './empty-state';
import { EvaluationStatusBadge } from './status-badge';

const NON_TERMINAL_RUN_STATUSES = new Set(['queued', 'running']);

const getRunProgress = (detail?: EvaluationRunDetailResponse | null) => {
  if (typeof detail?.progress?.percent === 'number') {
    return detail.progress.percent;
  }

  if (!detail?.summary?.total || detail.summary.total <= 0) {
    return 0;
  }

  const resolved =
    (detail.summary.completed || 0) +
    (detail.summary.failed || 0) +
    (detail.summary.cancelled || 0);

  return Math.round((resolved / detail.summary.total) * 100);
};

const matchesSearch = (item: EvaluationRunItem, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [
    item.case_key,
    item.status,
    item.error,
    item.input_message,
    item.latest_attempt?.agent_chat_id,
    item.latest_attempt?.agent_turn_id,
  ].some((value) => String(value ?? '').toLowerCase().includes(query));
};

export const EvaluationRunDetail = ({
  botId,
  detail,
  items,
  unavailable,
  error,
}: {
  botId?: string;
  detail: EvaluationRunDetailResponse | null;
  items: EvaluationRunItem[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_bot_evaluation');
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [isPending, startTransition] = useTransition();

  const run = detail?.run;
  const filteredItems = useMemo(() => {
    return items.filter((item) => matchesSearch(item, searchValue));
  }, [items, searchValue]);

  useEffect(() => {
    if (!run?.status || !NON_TERMINAL_RUN_STATUSES.has(run.status)) {
      return;
    }

    const timer = window.setTimeout(() => {
      startTransition(() => {
        router.refresh();
      });
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [router, run?.status, startTransition]);

  const refreshPage = () => {
    startTransition(() => {
      router.refresh();
    });
  };

  const handleCancelRun = async () => {
    if (!run?.id) return;

    try {
      await cancelEvaluationRun(run.id);
      toast.success(t('cancel_success'));
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error ? actionError.message : t('action_failed'),
      );
    }
  };

  const handleRetryItem = async (itemId?: string) => {
    if (!run?.id || !itemId) return;

    try {
      await retryEvaluationRunItem(run.id, itemId);
      toast.success(t('retry_success'));
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error ? actionError.message : t('action_failed'),
      );
    }
  };

  if (unavailable) {
    return (
      <EvaluationApiNotice
        title={t('not_available_title')}
        description={error || t('not_available_description')}
      />
    );
  }

  if (error) {
    return (
      <EvaluationApiNotice
        title={t('error_title')}
        description={error || t('error_description')}
      />
    );
  }

  if (!run) {
    return (
      <EvaluationEmptyState
        title={t('empty_detail_title')}
        description={t('empty_detail_description')}
      />
    );
  }

  const resolvedBotId = run.bot_id || botId;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 xl:grid-cols-[1.4fr_repeat(4,1fr)]">
        <Card className="xl:col-span-1">
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <CardDescription>{t('run_detail')}</CardDescription>
                <CardTitle className="text-2xl">
                  {run.name || run.id || '--'}
                </CardTitle>
              </div>
              <EvaluationStatusBadge status={run.status} />
            </div>
            <CardDescription>
              {t('dataset')}: {run.dataset_name || run.dataset_id || '--'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{t('progress')}</span>
                <span>{getRunProgress(detail)}%</span>
              </div>
              <Progress value={getRunProgress(detail)} />
            </div>
            <div className="text-muted-foreground flex flex-wrap gap-6 text-sm">
              <div>
                {t('created_at')}:{' '}
                {run.created_at ? (
                  <FormatDate datetime={new Date(run.created_at)} />
                ) : (
                  '--'
                )}
              </div>
              <div>
                {t('updated_at')}:{' '}
                {run.updated_at ? (
                  <FormatDate datetime={new Date(run.updated_at)} />
                ) : (
                  '--'
                )}
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_total')}</CardDescription>
            <CardTitle className="text-3xl">
              {detail?.summary?.total ?? 0}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_running')}</CardDescription>
            <CardTitle className="text-3xl">
              {detail?.summary?.running ?? 0}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_completed')}</CardDescription>
            <CardTitle className="text-3xl">
              {detail?.summary?.completed ?? 0}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('avg_score')}</CardDescription>
            <CardTitle className="text-3xl">
              {typeof detail?.summary?.avg_score === 'number'
                ? detail.summary.avg_score.toFixed(2)
                : '--'}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle>{t('items')}</CardTitle>
            <CardDescription>{t('items_description')}</CardDescription>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              className="w-full sm:w-72"
              placeholder={t('search_placeholder')}
              value={searchValue}
              onChange={(event) => setSearchValue(event.currentTarget.value)}
            />
            {NON_TERMINAL_RUN_STATUSES.has(run.status || '') && (
              <Button
                variant="outline"
                onClick={handleCancelRun}
                disabled={isPending}
              >
                {t('cancel_run')}
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {filteredItems.length === 0 ? (
            <EvaluationEmptyState
              title={t('empty_items_title')}
              description={t('empty_items_description')}
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t('case_key')}</TableHead>
                  <TableHead>{t('status_label')}</TableHead>
                  <TableHead>{t('score')}</TableHead>
                  <TableHead>{t('latest_attempt')}</TableHead>
                  <TableHead>{t('trace')}</TableHead>
                  <TableHead>{t('error')}</TableHead>
                  <TableHead className="text-right">{t('actions')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredItems.map((item) => (
                  <TableRow key={item.id}>
                    <TableCell className="font-medium">
                      {item.case_key || '--'}
                    </TableCell>
                    <TableCell>
                      <EvaluationStatusBadge status={item.status} />
                    </TableCell>
                    <TableCell>{item.best_score ?? '--'}</TableCell>
                    <TableCell>
                      <div className="space-y-1">
                        <div>
                          {t('attempt_label')} #
                          {item.latest_attempt?.attempt_no ?? '--'}
                        </div>
                        {item.latest_attempt?.finished_at && (
                          <div className="text-muted-foreground text-xs">
                            <FormatDate
                              datetime={
                                new Date(item.latest_attempt.finished_at)
                              }
                            />
                          </div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      {item.latest_attempt?.agent_chat_id && resolvedBotId ? (
                        <Link
                          className="text-primary text-sm underline-offset-4 hover:underline"
                          href={`/workspace/bots/${resolvedBotId}/chats/${item.latest_attempt.agent_chat_id}`}
                        >
                          {t('open_chat')}
                        </Link>
                      ) : (
                        '--'
                      )}
                    </TableCell>
                    <TableCell className="max-w-60 text-sm whitespace-normal">
                      {item.error || item.latest_attempt?.error || '--'}
                    </TableCell>
                    <TableCell className="text-right">
                      {(item.status === 'failed' ||
                        item.status === 'cancelled') && (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={isPending}
                          onClick={() => handleRetryItem(item.id)}
                        >
                          {t('retry_item')}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
