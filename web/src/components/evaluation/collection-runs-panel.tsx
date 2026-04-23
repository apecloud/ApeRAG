'use client';

import { useMemo, useState, useTransition } from 'react';

import { FormatDate } from '@/components/format-date';
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { FlaskConical, PlayCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import { createEvaluationRun } from '@/features/evaluation/client-api';
import type {
  EvaluationDataset,
  EvaluationRun,
} from '@/features/evaluation/types';
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

const isBotMissingError = (message: string): boolean => {
  const lower = message.toLowerCase();
  return (
    lower.includes('no default bot') ||
    lower.includes('no_default_bot') ||
    lower.includes('default agent bot') ||
    lower.includes('no bot available') ||
    lower.includes('bot_id') ||
    lower.includes('bot not found')
  );
};

type StartRunFormState = {
  datasetId: string;
  name: string;
};

const defaultStartRunForm: StartRunFormState = {
  datasetId: '',
  name: '',
};

export const CollectionRunsPanel = ({
  collectionId,
  datasets,
  runs,
  unavailable,
  error,
}: {
  collectionId: string;
  datasets: EvaluationDataset[];
  runs: EvaluationRun[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_collection_evaluations');
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [startRunOpen, setStartRunOpen] = useState(false);
  const [startRunForm, setStartRunForm] =
    useState<StartRunFormState>(defaultStartRunForm);
  const [isPending, startTransition] = useTransition();

  const refreshPage = () => {
    startTransition(() => {
      router.refresh();
    });
  };

  const selectableDatasets = useMemo(
    () => datasets.filter((dataset) => (dataset.item_count ?? 0) > 0),
    [datasets],
  );

  const selectedDataset = useMemo(
    () =>
      datasets.find((dataset) => dataset.id === startRunForm.datasetId) ?? null,
    [datasets, startRunForm.datasetId],
  );

  const startRunDisabled = selectableDatasets.length === 0;

  const filteredRuns = useMemo(() => {
    return runs.filter((run) => matchesSearch(run, searchValue));
  }, [runs, searchValue]);

  const handleStartRun = async () => {
    if (!startRunForm.datasetId) {
      toast.error(t('start_run_empty_dataset'));
      return;
    }

    if (!selectedDataset || (selectedDataset.item_count ?? 0) <= 0) {
      toast.error(t('start_run_empty_dataset'));
      return;
    }

    try {
      const payload = await createEvaluationRun({
        dataset_id: startRunForm.datasetId,
        name: startRunForm.name.trim() || undefined,
      });

      toast.success(t('start_run_success'));
      setStartRunOpen(false);
      setStartRunForm(defaultStartRunForm);

      if (payload?.id) {
        router.push(
          `/workspace/collections/${collectionId}/evaluations/${payload.id}`,
        );
        return;
      }

      refreshPage();
    } catch (actionError) {
      const rawMessage =
        actionError instanceof Error
          ? actionError.message
          : t('start_run_failed');
      const userFacing =
        actionError instanceof Error && isBotMissingError(actionError.message)
          ? t('start_run_no_bot_error')
          : rawMessage;
      toast.error(userFacing);
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
      <section className="flex flex-col gap-4 rounded-[1.75rem] border border-slate-200/80 bg-white/90 p-6 shadow-[0_24px_70px_-45px_rgba(15,23,42,0.4)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs tracking-[0.18em] text-slate-500 uppercase">
              <FlaskConical className="size-4" />
              {t('runs_badge')}
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
              placeholder={t('search_runs_placeholder')}
              value={searchValue}
              onChange={(event) => setSearchValue(event.currentTarget.value)}
            />
            <Button
              className="h-11 rounded-full"
              disabled={startRunDisabled}
              title={
                startRunDisabled ? t('start_run_empty_dataset') : undefined
              }
              onClick={() => {
                setStartRunForm({
                  ...defaultStartRunForm,
                  datasetId: selectableDatasets[0]?.id ?? '',
                });
                setStartRunOpen(true);
              }}
            >
              <PlayCircle className="size-4" />
              {t('start_run')}
            </Button>
          </div>
        </div>

        {filteredRuns.length === 0 ? (
          <div className="grid gap-3 rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-50/70 p-6">
            <h3 className="text-xl font-semibold tracking-[-0.03em] text-slate-950">
              {t('empty_runs_title')}
            </h3>
            <p className="max-w-[62ch] text-sm leading-7 text-slate-600 sm:text-base">
              {t('empty_runs_description')}
            </p>
          </div>
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {filteredRuns.map((run) => (
              <Link
                key={run.id}
                href={`/workspace/collections/${collectionId}/evaluations/${run.id}`}
              >
                <Card className="h-full overflow-hidden border-slate-200/80 bg-white shadow-[0_22px_60px_-40px_rgba(15,23,42,0.35)] transition-colors hover:bg-sky-50/50">
                  <CardHeader className="gap-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="space-y-1">
                        <CardTitle className="text-xl tracking-[-0.03em]">
                          {run.name || run.id || '--'}
                        </CardTitle>
                        <CardDescription>
                          {t('dataset_name_column')}:{' '}
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
              </Link>
            ))}
          </div>
        )}
      </section>

      <Dialog open={startRunOpen} onOpenChange={setStartRunOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t('start_run_title')}</DialogTitle>
            <DialogDescription>{t('start_run_description')}</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('dataset_select_label')}
              </label>
              <Select
                value={startRunForm.datasetId}
                onValueChange={(value) =>
                  setStartRunForm((prev) => ({ ...prev, datasetId: value }))
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder={t('dataset_select_placeholder')} />
                </SelectTrigger>
                <SelectContent>
                  {selectableDatasets.map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.item_count ?? 0})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('run_name_label')}
              </label>
              <Input
                value={startRunForm.name}
                placeholder={t('run_name_placeholder')}
                onChange={(event) =>
                  setStartRunForm((prev) => ({
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
              onClick={() => setStartRunOpen(false)}
              disabled={isPending}
            >
              {t('cancel')}
            </Button>
            <Button onClick={handleStartRun} disabled={isPending}>
              {t('start_run')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
