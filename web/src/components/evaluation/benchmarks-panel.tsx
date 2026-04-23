'use client';

import { useMemo, useState, useTransition } from 'react';

import { CopyToClipboard } from '@/components/copy-to-clipboard';
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
import { Textarea } from '@/components/ui/textarea';
import { Database, FolderPlus, Sparkles } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import {
  createBenchmarkDataset,
  createBenchmarkDatasetVersion,
} from '@/features/evaluation/client-api';
import type { BenchmarkDataset } from '@/features/evaluation/types';
import { EvaluationApiNotice } from './api-notice';
import { DatasetVersionStatusBadge } from './status-badge';

const matchesSearch = (dataset: BenchmarkDataset, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [
    dataset.name,
    dataset.description,
    dataset.source_type,
    dataset.latest_version?.version_name,
    dataset.latest_version?.version,
  ].some((value) => String(value ?? '').toLowerCase().includes(query));
};

const getCaseCount = (dataset: BenchmarkDataset) => {
  return dataset.latest_version?.case_count ?? 0;
};

const getVersionCount = (dataset?: BenchmarkDataset | null) => {
  const version = dataset?.latest_version?.version;
  if (!version) return undefined;
  const match = String(version).match(/(\d+)$/);
  return match ? Number(match[1]) : undefined;
};

type DatasetFormState = {
  name: string;
  description: string;
};

type VersionFormState = {
  versionName: string;
  caseKey: string;
  inputMessage: string;
  expectedAnswer: string;
  referenceContext: string;
};

type CreatedVersionState = {
  datasetId: string;
  datasetName: string;
  versionId: string;
  versionName?: string;
};

const defaultDatasetForm: DatasetFormState = {
  name: '',
  description: '',
};

const defaultVersionForm: VersionFormState = {
  versionName: '',
  caseKey: '',
  inputMessage: '',
  expectedAnswer: '',
  referenceContext: '',
};

const getVersionSeed = (dataset?: BenchmarkDataset | null) => {
  const nextVersion = (getVersionCount(dataset) ?? 0) + 1;
  return `v${nextVersion}`;
};

export const BenchmarksPanel = ({
  collectionId,
  items,
  unavailable,
  error,
}: {
  collectionId: string;
  items: BenchmarkDataset[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_benchmarks');
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [createDatasetOpen, setCreateDatasetOpen] = useState(false);
  const [createVersionOpen, setCreateVersionOpen] = useState(false);
  const [datasetForm, setDatasetForm] =
    useState<DatasetFormState>(defaultDatasetForm);
  const [versionForm, setVersionForm] =
    useState<VersionFormState>(defaultVersionForm);
  const [versionTarget, setVersionTarget] = useState<BenchmarkDataset | null>(
    null,
  );
  const [latestCreatedVersion, setLatestCreatedVersion] =
    useState<CreatedVersionState | null>(null);
  const [isPending, startTransition] = useTransition();

  const filteredItems = useMemo(() => {
    return items.filter((dataset) => matchesSearch(dataset, searchValue));
  }, [items, searchValue]);

  const summary = useMemo(() => {
    return {
      datasets: items.length,
      readyDatasets: items.filter(
        (dataset) => dataset.latest_version?.status === 'published',
      ).length,
      totalCases: items.reduce(
        (sum, dataset) => sum + getCaseCount(dataset),
        0,
      ),
    };
  }, [items]);

  const refreshPage = () => {
    startTransition(() => {
      router.refresh();
    });
  };

  const openCreateVersion = (dataset?: BenchmarkDataset | null) => {
    const resolvedDataset = dataset || null;
    setVersionTarget(resolvedDataset);
    setVersionForm({
      ...defaultVersionForm,
      versionName: getVersionSeed(resolvedDataset),
      caseKey: resolvedDataset?.name
        ? `${String(resolvedDataset.name).toLowerCase().replace(/\s+/g, '-')}-001`
        : '',
    });
    setCreateVersionOpen(true);
  };

  const handleCreateDataset = async () => {
    if (!datasetForm.name.trim()) {
      toast.error(t('create_dataset_name_required'));
      return;
    }

    try {
      const payload = await createBenchmarkDataset({
        name: datasetForm.name.trim(),
        description: datasetForm.description.trim() || undefined,
        collection_id: collectionId,
        source_type: 'manual',
      });

      if (!payload?.id) {
        throw new Error(t('create_dataset_missing_id'));
      }

      const nextDataset: BenchmarkDataset = {
        ...payload,
        name: payload.name || datasetForm.name.trim(),
      };

      toast.success(t('create_dataset_success'));
      setCreateDatasetOpen(false);
      setDatasetForm(defaultDatasetForm);
      openCreateVersion(nextDataset);
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('create_dataset_failed'),
      );
    }
  };

  const handleCreateVersion = async () => {
    if (!versionTarget?.id) return;
    if (!versionForm.inputMessage.trim()) {
      toast.error(t('create_version_input_required'));
      return;
    }

    try {
      const payload = await createBenchmarkDatasetVersion(versionTarget.id, {
        version_name: versionForm.versionName.trim() || undefined,
        cases: [
          {
            case_key: versionForm.caseKey.trim() || undefined,
            input_message: versionForm.inputMessage.trim(),
            expected_answer: versionForm.expectedAnswer.trim() || undefined,
            reference_context: versionForm.referenceContext.trim() || undefined,
            sort_key: 0,
          },
        ],
      });

      if (!payload?.id) {
        throw new Error(t('create_version_missing_id'));
      }

      toast.success(t('create_version_success'));
      setLatestCreatedVersion({
        datasetId: versionTarget.id,
        datasetName: versionTarget.name || t('datasets'),
        versionId: payload.id,
        versionName:
          payload.version_name ||
          versionForm.versionName.trim() ||
          String(payload.version),
      });
      setCreateVersionOpen(false);
      setVersionForm(defaultVersionForm);
      setVersionTarget(null);
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('create_version_failed'),
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
        <section className="relative overflow-hidden rounded-[2rem] border border-amber-200/60 bg-[radial-gradient(circle_at_top_left,_rgba(245,158,11,0.16),_transparent_35%),radial-gradient(circle_at_bottom_right,_rgba(16,185,129,0.14),_transparent_40%),linear-gradient(180deg,rgba(255,255,255,0.96),rgba(255,255,255,0.92))] p-6 shadow-[0_24px_60px_-28px_rgba(15,23,42,0.28)] sm:p-8">
          <div className="pointer-events-none absolute inset-y-0 right-0 hidden w-1/3 bg-[radial-gradient(circle_at_center,_rgba(250,204,21,0.18),_transparent_55%)] lg:block" />
          <div className="grid gap-8 lg:grid-cols-[1.3fr_0.9fr]">
            <div className="space-y-5">
              <Badge
                variant="secondary"
                className="rounded-full border border-amber-200 bg-white/80 px-3 py-1 text-[11px] tracking-[0.18em] text-slate-700 uppercase"
              >
                {t('entry_badge')}
              </Badge>
              <div className="space-y-3">
                <h1 className="max-w-3xl text-4xl leading-none font-semibold tracking-[-0.04em] text-balance text-slate-950 sm:text-5xl">
                  {t('hero_title')}
                </h1>
                <p className="max-w-[62ch] text-sm leading-7 text-slate-600 sm:text-base">
                  {t('hero_description')}
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <Badge className="rounded-full bg-slate-950 px-4 py-1.5 text-white hover:bg-slate-950">
                  1. {t('step_create_dataset')}
                </Badge>
                <Badge
                  variant="secondary"
                  className="rounded-full border border-slate-200 bg-white/85 px-4 py-1.5 text-slate-700"
                >
                  2. {t('step_publish_version')}
                </Badge>
                <Badge
                  variant="secondary"
                  className="rounded-full border border-slate-200 bg-white/85 px-4 py-1.5 text-slate-700"
                >
                  3. {t('step_start_run')}
                </Badge>
              </div>
            </div>

            <Card className="border-white/70 bg-white/80 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)] backdrop-blur">
              <CardHeader className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="flex size-10 items-center justify-center rounded-2xl bg-slate-950 text-white">
                    <Sparkles className="size-5" />
                  </div>
                  <div>
                    <CardTitle className="text-xl">
                      {t('first_use_title')}
                    </CardTitle>
                    <CardDescription>
                      {t('first_use_description')}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-3">
                  <div className="rounded-2xl border border-slate-200/80 bg-white/85 p-4">
                    <div className="text-xs font-medium tracking-[0.16em] text-slate-500 uppercase">
                      {t('datasets')}
                    </div>
                    <div className="mt-2 text-sm text-slate-700">
                      {t('first_use_dataset_hint')}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/80 bg-white/85 p-4">
                    <div className="text-xs font-medium tracking-[0.16em] text-slate-500 uppercase">
                      {t('versions')}
                    </div>
                    <div className="mt-2 text-sm text-slate-700">
                      {t('first_use_version_hint')}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-slate-200/80 bg-white/85 p-4">
                    <div className="text-xs font-medium tracking-[0.16em] text-slate-500 uppercase">
                      {t('runs')}
                    </div>
                    <div className="mt-2 text-sm text-slate-700">
                      {t('first_use_run_hint')}
                    </div>
                  </div>
                </div>
                <Button
                  className="w-full rounded-full"
                  onClick={() => setCreateDatasetOpen(true)}
                >
                  <FolderPlus className="size-4" />
                  {t('create_dataset')}
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        {latestCreatedVersion && (
          <Card className="border-emerald-200 bg-emerald-50/70 shadow-[0_20px_60px_-40px_rgba(5,150,105,0.5)]">
            <CardHeader className="gap-4">
              <div className="space-y-2">
                <CardTitle className="text-xl">
                  {t('created_version_title')}
                </CardTitle>
                <CardDescription className="max-w-[70ch] text-slate-700">
                  {t('created_version_description', {
                    dataset: latestCreatedVersion.datasetName,
                    version:
                      latestCreatedVersion.versionName ||
                      latestCreatedVersion.versionId,
                  })}
                </CardDescription>
              </div>
              <div className="rounded-2xl border border-emerald-200/80 bg-white/80 p-4">
                <div className="text-xs font-medium tracking-[0.16em] text-emerald-700 uppercase">
                  {t('published_version_id')}
                </div>
                <div className="mt-2 flex items-center gap-3">
                  <code className="min-w-0 flex-1 truncate rounded-xl bg-emerald-100/80 px-3 py-2 text-sm text-emerald-950">
                    {latestCreatedVersion.versionId}
                  </code>
                  <CopyToClipboard
                    text={latestCreatedVersion.versionId}
                    variant="outline"
                    className="border-emerald-200 bg-white text-emerald-900 hover:bg-emerald-50"
                  />
                </div>
                <p className="mt-3 text-sm text-slate-700">
                  {t('choose_bot_hint')}
                </p>
              </div>
            </CardHeader>
          </Card>
        )}

        <div className="grid gap-4 md:grid-cols-3">
          <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
            <CardHeader className="pb-2">
              <CardDescription>{t('datasets')}</CardDescription>
              <CardTitle className="text-3xl tracking-tight">
                {summary.datasets}
              </CardTitle>
            </CardHeader>
          </Card>
          <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
            <CardHeader className="pb-2">
              <CardDescription>{t('ready_datasets')}</CardDescription>
              <CardTitle className="text-3xl tracking-tight">
                {summary.readyDatasets}
              </CardTitle>
            </CardHeader>
          </Card>
          <Card className="border-slate-200/70 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.38)]">
            <CardHeader className="pb-2">
              <CardDescription>{t('cases')}</CardDescription>
              <CardTitle className="text-3xl tracking-tight">
                {summary.totalCases}
              </CardTitle>
            </CardHeader>
          </Card>
        </div>

        <section className="flex flex-col gap-4 rounded-[1.75rem] border border-slate-200/80 bg-white/90 p-6 shadow-[0_24px_70px_-45px_rgba(15,23,42,0.4)]">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs tracking-[0.18em] text-slate-500 uppercase">
                <Database className="size-4" />
                {t('datasets')}
              </div>
              <h2 className="text-3xl leading-none font-semibold tracking-[-0.03em] text-slate-950">
                {t('datasets_section_title')}
              </h2>
              <p className="max-w-[70ch] text-sm leading-7 text-slate-600 sm:text-base">
                {t('datasets_section_description')}
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
                onClick={() => setCreateDatasetOpen(true)}
              >
                <FolderPlus className="size-4" />
                {t('create_dataset')}
              </Button>
            </div>
          </div>

          {filteredItems.length === 0 ? (
            <div className="grid gap-4 rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-50/70 p-6 lg:grid-cols-[1.2fr_0.8fr]">
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
                    onClick={() => setCreateDatasetOpen(true)}
                  >
                    <FolderPlus className="size-4" />
                    {t('create_dataset')}
                  </Button>
                </div>
              </div>
              <div className="grid gap-3">
                <div className="rounded-2xl border bg-white p-4">
                  <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                    1
                  </div>
                  <div className="mt-2 font-medium text-slate-900">
                    {t('step_create_dataset')}
                  </div>
                </div>
                <div className="rounded-2xl border bg-white p-4">
                  <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                    2
                  </div>
                  <div className="mt-2 font-medium text-slate-900">
                    {t('step_publish_version')}
                  </div>
                </div>
                <div className="rounded-2xl border bg-white p-4">
                  <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                    3
                  </div>
                  <div className="mt-2 font-medium text-slate-900">
                    {t('step_start_run')}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid gap-4 xl:grid-cols-2">
              {filteredItems.map((dataset) => (
                <Card
                  key={
                    dataset.id ||
                    dataset.name ||
                    `${dataset.source_type || 'dataset'}-${dataset.created_at || dataset.latest_version?.version || 'unknown'}`
                  }
                  className="h-full overflow-hidden border-slate-200/80 bg-white shadow-[0_22px_60px_-40px_rgba(15,23,42,0.35)]"
                >
                  <CardHeader className="gap-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div className="space-y-1">
                        <CardTitle className="text-2xl tracking-[-0.03em]">
                          {dataset.name || dataset.id || '--'}
                        </CardTitle>
                        <CardDescription className="max-w-2xl text-sm leading-7">
                          {dataset.description || t('no_description')}
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <DatasetVersionStatusBadge
                          status={dataset.latest_version?.status}
                        />
                        <Button
                          variant="outline"
                          className="rounded-full bg-white"
                          onClick={() => openCreateVersion(dataset)}
                        >
                          {dataset.latest_version
                            ? t('create_version')
                            : t('publish_first_version')}
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="grid gap-4">
                    <div className="grid gap-3 sm:grid-cols-3">
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4">
                        <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                          {t('source_type')}
                        </div>
                        <div className="mt-2 font-medium text-slate-900">
                          {dataset.source_type || '--'}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4">
                        <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                          {t('latest_version')}
                        </div>
                        <div className="mt-2 font-medium text-slate-900">
                          {dataset.latest_version?.version_name ||
                            dataset.latest_version?.version ||
                            '--'}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4">
                        <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                          {t('cases')}
                        </div>
                        <div className="mt-2 font-medium text-slate-900">
                          {getCaseCount(dataset)}
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-6 text-sm text-slate-500">
                      <div>
                        {t('version_count')}: {getVersionCount(dataset) ?? '--'}
                      </div>
                      <div>
                        {t('created_at')}:{' '}
                        {dataset.created_at ? (
                          <FormatDate datetime={new Date(dataset.created_at)} />
                        ) : (
                          '--'
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </section>
      </div>

      <Dialog open={createDatasetOpen} onOpenChange={setCreateDatasetOpen}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t('create_dataset_title')}</DialogTitle>
            <DialogDescription>
              {t('create_dataset_description')}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('dataset_name_label')}
              </label>
              <Input
                value={datasetForm.name}
                placeholder={t('dataset_name_placeholder')}
                onChange={(event) =>
                  setDatasetForm((prev) => ({
                    ...prev,
                    name: event.currentTarget.value,
                  }))
                }
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('dataset_description_label')}
              </label>
              <Textarea
                rows={4}
                value={datasetForm.description}
                placeholder={t('dataset_description_placeholder')}
                onChange={(event) =>
                  setDatasetForm((prev) => ({
                    ...prev,
                    description: event.currentTarget.value,
                  }))
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateDatasetOpen(false)}
              disabled={isPending}
            >
              {t('cancel')}
            </Button>
            <Button onClick={handleCreateDataset} disabled={isPending}>
              {t('create_dataset')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={createVersionOpen} onOpenChange={setCreateVersionOpen}>
        <DialogContent className="sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>{t('create_version_title')}</DialogTitle>
            <DialogDescription>
              {versionTarget?.name
                ? t('create_version_description_named', {
                    dataset: versionTarget.name,
                  })
                : t('create_version_description')}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('version_name_label')}
              </label>
              <Input
                value={versionForm.versionName}
                placeholder={t('version_name_placeholder')}
                onChange={(event) =>
                  setVersionForm((prev) => ({
                    ...prev,
                    versionName: event.currentTarget.value,
                  }))
                }
              />
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="grid gap-2">
                <label className="text-sm font-medium text-slate-900">
                  {t('case_key_label')}
                </label>
                <Input
                  value={versionForm.caseKey}
                  placeholder={t('case_key_placeholder')}
                  onChange={(event) =>
                    setVersionForm((prev) => ({
                      ...prev,
                      caseKey: event.currentTarget.value,
                    }))
                  }
                />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4 text-sm text-slate-600">
                {t('create_version_helper')}
              </div>
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('input_message_label')}
              </label>
              <Textarea
                rows={4}
                value={versionForm.inputMessage}
                placeholder={t('input_message_placeholder')}
                onChange={(event) =>
                  setVersionForm((prev) => ({
                    ...prev,
                    inputMessage: event.currentTarget.value,
                  }))
                }
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('expected_answer_label')}
              </label>
              <Textarea
                rows={3}
                value={versionForm.expectedAnswer}
                placeholder={t('expected_answer_placeholder')}
                onChange={(event) =>
                  setVersionForm((prev) => ({
                    ...prev,
                    expectedAnswer: event.currentTarget.value,
                  }))
                }
              />
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('reference_context_label')}
              </label>
              <Textarea
                rows={3}
                value={versionForm.referenceContext}
                placeholder={t('reference_context_placeholder')}
                onChange={(event) =>
                  setVersionForm((prev) => ({
                    ...prev,
                    referenceContext: event.currentTarget.value,
                  }))
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setCreateVersionOpen(false)}
              disabled={isPending}
            >
              {t('cancel')}
            </Button>
            <Button onClick={handleCreateVersion} disabled={isPending}>
              {t('publish_first_version')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};
