'use client';

import { useEffect, useMemo, useState, useTransition } from 'react';

import type { Bot } from '@/api';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ArrowRight, FlaskConical, PlayCircle, Sparkles } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import React from 'react';
import { toast } from 'sonner';

import { postEvaluationAction } from './action-utils';
import { EvaluationApiNotice } from './api-notice';
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

type CreateRunFormState = {
  datasetVersionId: string;
  name: string;
};

export const EvaluationRunsPanel = ({
  bot,
  runs,
  unavailable,
  error,
  initialDatasetVersionId,
}: {
  bot?: Bot;
  runs: EvaluationRun[];
  unavailable: boolean;
  error?: string;
  initialDatasetVersionId?: string;
}) => {
  const t = useTranslations('page_bot_evaluation');
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [createRunOpen, setCreateRunOpen] = useState(false);
  const [createRunForm, setCreateRunForm] = useState<CreateRunFormState>({
    datasetVersionId: initialDatasetVersionId || '',
    name: '',
  });
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    if (!initialDatasetVersionId) return;
    setCreateRunForm((prev) => ({
      ...prev,
      datasetVersionId: prev.datasetVersionId || initialDatasetVersionId,
    }));
  }, [initialDatasetVersionId]);

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

  const refreshPage = () => {
    startTransition(() => {
      router.refresh();
    });
  };

  const handleCreateRun = async () => {
    if (!bot?.id) {
      toast.error(t('missing_bot'));
      return;
    }

    if (!createRunForm.datasetVersionId.trim()) {
      toast.error(t('dataset_version_id_required'));
      return;
    }

    try {
      const payload = await postEvaluationAction<{ run?: EvaluationRun }>(
        '/api/v2/evaluation-runs',
        {
          bot_id: bot.id,
          dataset_version_id: createRunForm.datasetVersionId.trim(),
          name: createRunForm.name.trim() || undefined,
        },
      );

      toast.success(t('create_run_success'));
      setCreateRunOpen(false);
      setCreateRunForm({
        datasetVersionId: createRunForm.datasetVersionId.trim(),
        name: '',
      });

      if (payload.run?.id && bot.id) {
        router.push(
          `/workspace/bots/${bot.id}/evaluation/runs/${payload.run.id}`,
        );
        return;
      }

      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('create_run_failed'),
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

  return (
    <>
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
              <div className="flex flex-wrap gap-3">
                <Badge className="rounded-full bg-slate-950 px-4 py-1.5 text-white hover:bg-slate-950">
                  1. {t('step_prepare_dataset')}
                </Badge>
                <Badge
                  variant="secondary"
                  className="rounded-full border border-slate-200 bg-white/85 px-4 py-1.5 text-slate-700"
                >
                  2. {t('step_create_run')}
                </Badge>
                <Badge
                  variant="secondary"
                  className="rounded-full border border-slate-200 bg-white/85 px-4 py-1.5 text-slate-700"
                >
                  3. {t('step_review_results')}
                </Badge>
              </div>
            </div>

            <Card className="border-white/70 bg-white/82 shadow-[inset_0_1px_0_rgba(255,255,255,0.72)] backdrop-blur">
              <CardHeader className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="flex size-10 items-center justify-center rounded-2xl bg-slate-950 text-white">
                    <Sparkles className="size-5" />
                  </div>
                  <div>
                    <CardTitle className="text-xl">
                      {t('runs_scope_title')}
                    </CardTitle>
                    <CardDescription>
                      {t('runs_scope_description')}
                    </CardDescription>
                  </div>
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
                  <p className="mt-2 text-sm text-slate-600">
                    {t('runs_scope_helper')}
                  </p>
                </div>
                {initialDatasetVersionId && (
                  <div className="rounded-2xl border border-emerald-200 bg-emerald-50/70 p-4">
                    <div className="text-xs tracking-[0.16em] text-emerald-700 uppercase">
                      {t('prefilled_dataset_version')}
                    </div>
                    <div className="mt-2 font-medium break-all text-slate-900">
                      {initialDatasetVersionId}
                    </div>
                  </div>
                )}
                <Button
                  className="w-full rounded-full"
                  onClick={() => setCreateRunOpen(true)}
                >
                  <PlayCircle className="size-4" />
                  {t('create_run')}
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
              <Button
                className="h-11 rounded-full"
                onClick={() => setCreateRunOpen(true)}
              >
                <PlayCircle className="size-4" />
                {t('create_run')}
              </Button>
            </div>
          </div>

          {filteredRuns.length === 0 ? (
            <div className="grid gap-4 rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-50/70 p-6 lg:grid-cols-[1.15fr_0.85fr]">
              <div className="space-y-3">
                <h3 className="text-2xl font-semibold tracking-[-0.03em] text-slate-950">
                  {t('empty_title')}
                </h3>
                <p className="max-w-[62ch] text-sm leading-7 text-slate-600 sm:text-base">
                  {t('empty_description')}
                </p>
                <div className="flex flex-wrap gap-3">
                  <Button
                    className="rounded-full"
                    onClick={() => setCreateRunOpen(true)}
                  >
                    <PlayCircle className="size-4" />
                    {t('create_run')}
                  </Button>
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
              <div className="grid gap-3">
                <div className="rounded-2xl border bg-white p-4">
                  <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                    1
                  </div>
                  <div className="mt-2 font-medium text-slate-900">
                    {t('step_prepare_dataset')}
                  </div>
                </div>
                <div className="rounded-2xl border bg-white p-4">
                  <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                    2
                  </div>
                  <div className="mt-2 font-medium text-slate-900">
                    {t('step_create_run')}
                  </div>
                </div>
                <div className="rounded-2xl border bg-white p-4">
                  <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                    3
                  </div>
                  <div className="mt-2 font-medium text-slate-900">
                    {t('step_review_results')}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid gap-4 xl:grid-cols-2">
              {filteredRuns.map((run) => {
                const key =
                  run.id ||
                  `${run.dataset_version_id}-${run.created_at || 'run'}`;
                const content = (
                  <Card className="h-full overflow-hidden border-slate-200/80 bg-white shadow-[0_22px_60px_-40px_rgba(15,23,42,0.35)] transition-colors hover:bg-sky-50/50">
                    <CardHeader className="gap-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-1">
                          <CardTitle className="text-xl tracking-[-0.03em]">
                            {run.id || '--'}
                          </CardTitle>
                          <CardDescription>
                            {t('dataset_version')}:{' '}
                            {run.dataset_version_id || '--'}
                          </CardDescription>
                        </div>
                        <EvaluationStatusBadge status={run.status} />
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-slate-500">
                            {t('progress')}
                          </span>
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
        </section>
      </div>

      <Dialog open={createRunOpen} onOpenChange={setCreateRunOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t('create_run_title')}</DialogTitle>
            <DialogDescription>{t('create_run_description')}</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4 text-sm text-slate-600">
              {t('create_run_helper')}
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('dataset_version_id_label')}
              </label>
              <Input
                value={createRunForm.datasetVersionId}
                placeholder={t('dataset_version_id_placeholder')}
                onChange={(event) =>
                  setCreateRunForm((prev) => ({
                    ...prev,
                    datasetVersionId: event.currentTarget.value,
                  }))
                }
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('run_name_label')}
              </label>
              <Input
                value={createRunForm.name}
                placeholder={t('run_name_placeholder')}
                onChange={(event) =>
                  setCreateRunForm((prev) => ({
                    ...prev,
                    name: event.currentTarget.value,
                  }))
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateRunOpen(false)}
              disabled={isPending}
            >
              {t('cancel')}
            </Button>
            <Button onClick={handleCreateRun} disabled={isPending}>
              {t('create_run')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
