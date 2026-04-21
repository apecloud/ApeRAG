'use client';

import { useMemo, useState } from 'react';

import type { Bot } from '@/api';
import { FormatDate } from '@/components/format-date';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import React from 'react';

import { EvaluationApiNotice } from './api-notice';
import { EvaluationEmptyState } from './empty-state';
import { EvaluationStatusBadge } from './status-badge';
import type { EvaluationRun } from './types';

const getRunProgress = (run: EvaluationRun) => {
  if (typeof run.progress?.percent === 'number') {
    return run.progress.percent;
  }

  if (!run.summary?.total || run.summary.total <= 0) {
    return 0;
  }

  const resolved =
    (run.summary.completed || 0) +
    (run.summary.failed || 0) +
    (run.summary.cancelled || 0);

  return Math.round((resolved / run.summary.total) * 100);
};

const matchesSearch = (run: EvaluationRun, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [run.id, run.dataset_version_id, run.status].some((value) =>
    value?.toLowerCase().includes(query),
  );
};

export const EvaluationRunsPanel = ({
  bot,
  runs,
  unavailable,
  error,
}: {
  bot?: Bot;
  runs: EvaluationRun[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_bot_evaluation');
  const [searchValue, setSearchValue] = useState('');

  const filteredRuns = useMemo(() => {
    return runs.filter((run) => matchesSearch(run, searchValue));
  }, [runs, searchValue]);

  const summary = useMemo(() => {
    return {
      total: runs.length,
      running: runs.filter((run) => run.status === 'running').length,
      failed: runs.filter((run) => run.status === 'failed').length,
      completed: runs.filter((run) => run.status === 'completed').length,
    };
  }, [runs]);

  if (unavailable) {
    return (
      <EvaluationApiNotice
        title={t('not_available_title')}
        description={error || t('not_available_description')}
      />
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 xl:grid-cols-[1.4fr_repeat(4,1fr)]">
        <Card className="xl:col-span-1">
          <CardHeader>
            <CardDescription>{t('bot_scope')}</CardDescription>
            <CardTitle className="text-2xl">{bot?.title || bot?.id || '--'}</CardTitle>
            <CardDescription>{t('metadata.description')}</CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('runs')}</CardDescription>
            <CardTitle className="text-3xl">{summary.total}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_running')}</CardDescription>
            <CardTitle className="text-3xl">{summary.running}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_completed')}</CardDescription>
            <CardTitle className="text-3xl">{summary.completed}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_failed')}</CardDescription>
            <CardTitle className="text-3xl">{summary.failed}</CardTitle>
          </CardHeader>
        </Card>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-semibold">{t('metadata.title')}</h2>
          <p className="text-muted-foreground text-sm">{t('metadata.description')}</p>
        </div>
        <Input
          className="w-full md:max-w-sm"
          placeholder={t('search_placeholder')}
          value={searchValue}
          onChange={(event) => setSearchValue(event.currentTarget.value)}
        />
      </div>

      {filteredRuns.length === 0 ? (
        <EvaluationEmptyState
          title={t('empty_title')}
          description={t('empty_description')}
        />
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          {filteredRuns.map((run) => {
            const key =
              run.id || `${run.dataset_version_id}-${run.created_at || 'run'}`;
            const content = (
              <Card className="h-full transition-colors hover:bg-accent/30">
                <CardHeader className="gap-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                      <CardTitle className="text-lg">{run.id || '--'}</CardTitle>
                      <CardDescription>
                        {t('dataset_version')}: {run.dataset_version_id || '--'}
                      </CardDescription>
                    </div>
                    <EvaluationStatusBadge status={run.status} />
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{t('progress')}</span>
                      <span>{getRunProgress(run)}%</span>
                    </div>
                    <Progress value={getRunProgress(run)} />
                  </div>

                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div className="rounded-xl border p-3">
                      <div className="text-muted-foreground text-xs">
                        {t('summary_total')}
                      </div>
                      <div className="mt-2 text-lg font-semibold">
                        {run.summary?.total ?? '--'}
                      </div>
                    </div>
                    <div className="rounded-xl border p-3">
                      <div className="text-muted-foreground text-xs">
                        {t('summary_running')}
                      </div>
                      <div className="mt-2 text-lg font-semibold">
                        {run.summary?.running ?? '--'}
                      </div>
                    </div>
                    <div className="rounded-xl border p-3">
                      <div className="text-muted-foreground text-xs">
                        {t('summary_completed')}
                      </div>
                      <div className="mt-2 text-lg font-semibold">
                        {run.summary?.completed ?? '--'}
                      </div>
                    </div>
                    <div className="rounded-xl border p-3">
                      <div className="text-muted-foreground text-xs">
                        {t('avg_score')}
                      </div>
                      <div className="mt-2 text-lg font-semibold">
                        {typeof run.summary?.avg_score === 'number'
                          ? run.summary.avg_score.toFixed(2)
                          : '--'}
                      </div>
                    </div>
                  </div>

                  <div className="text-muted-foreground flex items-center justify-between text-sm">
                    <div>
                      {t('created_at')}:{' '}
                      {run.created_at ? (
                        <FormatDate datetime={new Date(run.created_at)} />
                      ) : (
                        '--'
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );

            if (!run.id) {
              return <React.Fragment key={key}>{content}</React.Fragment>;
            }

            return (
              <Link
                key={key}
                href={`/workspace/bots/${bot?.id}/evaluation/runs/${run.id}`}
              >
                {content}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
};
