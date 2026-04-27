'use client';

import { useMemo, useState } from 'react';

import { FormatDate } from '@/components/format-date';
import { Badge } from '@/components/ui/badge';
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
import { ArrowRight, FlaskConical } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import React from 'react';

import type { Bot } from '@/features/bot/types';
import type { EvaluationRun } from '@/features/evaluation/types';
import { EvaluationApiNotice } from './api-notice';
import { EvaluationStatusBadge } from './status-badge';

const getRunProgress = (run: EvaluationRun) => {
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

  return [run.id, run.dataset_name, run.dataset_id, run.status, run.name].some(
    (value) => String(value ?? '').toLowerCase().includes(query),
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

  const collectionEvaluationHref = '/workspace/collections';

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

  return (
    <div className="flex flex-col gap-6">
      <section className="relative overflow-hidden rounded-[2rem] border border-sky-200/70 bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.16),_transparent_35%),radial-gradient(circle_at_bottom_right,_rgba(59,130,246,0.12),_transparent_38%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(255,255,255,0.92))] p-6 shadow-[0_24px_70px_-35px_rgba(15,23,42,0.35)] sm:p-8">
        <div className="grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
          <div className="space-y-5">
            <Badge
              variant="secondary"
              className="rounded-full border border-sky-200 bg-white/85 px-3 py-1 text-[11px] tracking-[0.18em] text-slate-700 uppercase"
            >
              {t('runs_badge')}
            </Badge>
            <div className="space-y-3">
              <h1 className="max-w-3xl text-4xl leading-none font-semibold tracking-[-0.04em] text-balance text-slate-950 sm:text-5xl">
                {t('hero_title')}
              </h1>
              <p className="max-w-[64ch] text-sm leading-7 text-slate-600 sm:text-base">
                {t('hero_description')}
              </p>
            </div>
          </div>

          <Card className="border-white/70 bg-white/82 shadow-[inset_0_1px_0_rgba(255,255,255,0.72)] backdrop-blur">
            <CardHeader className="space-y-3">
              <div>
                <CardTitle className="text-xl">{t('bot_scope')}</CardTitle>
                <CardDescription>{t('runs_scope_helper')}</CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-2xl border border-slate-200/80 bg-white/90 p-4">
                <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                  {t('bot_scope')}
                </div>
                <div className="mt-2 text-lg font-semibold tracking-tight text-slate-950">
                  {bot?.title || bot?.id || '--'}
                </div>
              </div>
              <Button
                asChild
                variant="outline"
                className="w-full rounded-full"
              >
                <Link href={collectionEvaluationHref}>
                  {t('open_collection_evaluation')}
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[1.4fr_repeat(4,1fr)]">
        <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)] xl:col-span-1">
          <CardHeader>
            <CardDescription>{t('runs_panel_title')}</CardDescription>
            <CardTitle className="text-2xl tracking-tight">
              {bot?.title || bot?.id || '--'}
            </CardTitle>
            <CardDescription>{t('runs_panel_description')}</CardDescription>
          </CardHeader>
        </Card>
        <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
          <CardHeader className="pb-2">
            <CardDescription>{t('runs')}</CardDescription>
            <CardTitle className="text-3xl tracking-tight">
              {summary.total}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_running')}</CardDescription>
            <CardTitle className="text-3xl tracking-tight">
              {summary.running}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_completed')}</CardDescription>
            <CardTitle className="text-3xl tracking-tight">
              {summary.completed}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
          <CardHeader className="pb-2">
            <CardDescription>{t('summary_failed')}</CardDescription>
            <CardTitle className="text-3xl tracking-tight">
              {summary.failed}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <section className="flex flex-col gap-4 rounded-[1.75rem] border border-slate-200/80 bg-white/90 p-6 shadow-[0_24px_70px_-45px_rgba(15,23,42,0.4)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs tracking-[0.18em] text-slate-500 uppercase">
              <FlaskConical className="size-4" />
              {t('runs')}
            </div>
            <h2 className="text-3xl leading-none font-semibold tracking-[-0.03em] text-slate-950">
              {t('runs_section_title')}
            </h2>
            <p className="max-w-[70ch] text-sm leading-7 text-slate-600 sm:text-base">
              {t('runs_section_description')}
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <Input
              className="h-11 rounded-full border-slate-200 bg-white/90 sm:w-72"
              placeholder={t('search_placeholder')}
              value={searchValue}
              onChange={(event) => setSearchValue(event.currentTarget.value)}
            />
          </div>
        </div>

        {filteredRuns.length === 0 ? (
          <div className="grid gap-3 rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-50/70 p-6">
            <h3 className="text-xl font-semibold tracking-[-0.03em] text-slate-950">
              {t('empty_title')}
            </h3>
            <p className="max-w-[62ch] text-sm leading-7 text-slate-600 sm:text-base">
              {t('empty_description')}
            </p>
            <div>
              <Button
                asChild
                variant="outline"
                className="rounded-full bg-white"
              >
                <Link href={collectionEvaluationHref}>
                  {t('open_collection_evaluation')}
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {filteredRuns.map((run) => {
              const href = bot?.id
                ? `/workspace/bots/${bot.id}/evaluation/runs/${run.id}`
                : null;
              const content = (
                <Card className="h-full overflow-hidden border-slate-200/80 bg-white shadow-[0_22px_60px_-40px_rgba(15,23,42,0.35)] transition-colors hover:bg-sky-50/50">
                  <CardHeader className="gap-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="space-y-1">
                        <CardTitle className="text-xl tracking-[-0.03em]">
                          {run.name || run.id || '--'}
                        </CardTitle>
                        <CardDescription>
                          {t('dataset')}:{' '}
                          {run.dataset_name || run.dataset_id || '--'}
                        </CardDescription>
                      </div>
                      <EvaluationStatusBadge status={run.status} />
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-500">{t('progress')}</span>
                        <span>{getRunProgress(run)}%</span>
                      </div>
                      <Progress value={getRunProgress(run)} />
                    </div>

                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3">
                        <div className="text-xs tracking-[0.14em] text-slate-500 uppercase">
                          {t('summary_total')}
                        </div>
                        <div className="mt-2 text-lg font-semibold">
                          {run.summary?.total ?? '--'}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3">
                        <div className="text-xs tracking-[0.14em] text-slate-500 uppercase">
                          {t('summary_running')}
                        </div>
                        <div className="mt-2 text-lg font-semibold">
                          {run.summary?.running ?? '--'}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3">
                        <div className="text-xs tracking-[0.14em] text-slate-500 uppercase">
                          {t('summary_completed')}
                        </div>
                        <div className="mt-2 text-lg font-semibold">
                          {run.summary?.completed ?? '--'}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-3">
                        <div className="text-xs tracking-[0.14em] text-slate-500 uppercase">
                          {t('avg_score')}
                        </div>
                        <div className="mt-2 text-lg font-semibold">
                          {typeof run.summary?.avg_score === 'number'
                            ? run.summary.avg_score.toFixed(2)
                            : '--'}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-sm text-slate-500">
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

              if (!href) {
                return <React.Fragment key={run.id}>{content}</React.Fragment>;
              }

              return (
                <Link key={run.id} href={href}>
                  {content}
                </Link>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
};
