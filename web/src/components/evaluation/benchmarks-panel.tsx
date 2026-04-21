'use client';

import { useMemo, useState } from 'react';

import { FormatDate } from '@/components/format-date';
import { Input } from '@/components/ui/input';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { useTranslations } from 'next-intl';

import { EvaluationApiNotice } from './api-notice';
import { EvaluationEmptyState } from './empty-state';
import { DatasetVersionStatusBadge } from './status-badge';
import type { BenchmarkDataset } from './types';

const matchesSearch = (dataset: BenchmarkDataset, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [
    dataset.name,
    dataset.description,
    dataset.source_type,
    dataset.latest_version?.version_name,
    dataset.latest_version?.version,
  ].some((value) => value?.toLowerCase().includes(query));
};

const getCaseCount = (dataset: BenchmarkDataset) => {
  return dataset.case_count ?? dataset.latest_version?.case_count ?? 0;
};

export const BenchmarksPanel = ({
  items,
  unavailable,
  error,
}: {
  items: BenchmarkDataset[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_benchmarks');
  const [searchValue, setSearchValue] = useState('');

  const filteredItems = useMemo(() => {
    return items.filter((dataset) => matchesSearch(dataset, searchValue));
  }, [items, searchValue]);

  const summary = useMemo(() => {
    return {
      datasets: items.length,
      readyDatasets: items.filter(
        (dataset) => dataset.latest_version?.status === 'published',
      ).length,
      totalCases: items.reduce((sum, dataset) => sum + getCaseCount(dataset), 0),
    };
  }, [items]);

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
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('datasets')}</CardDescription>
            <CardTitle className="text-3xl">{summary.datasets}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('ready_datasets')}</CardDescription>
            <CardTitle className="text-3xl">{summary.readyDatasets}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>{t('cases')}</CardDescription>
            <CardTitle className="text-3xl">{summary.totalCases}</CardTitle>
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

      {filteredItems.length === 0 ? (
        <EvaluationEmptyState
          title={t('empty_title')}
          description={t('empty_description')}
        />
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          {filteredItems.map((dataset) => (
            <Card
              key={
                dataset.id ||
                dataset.name ||
                `${dataset.source_type || 'dataset'}-${dataset.created_at || dataset.latest_version?.version || 'unknown'}`
              }
            >
              <CardHeader className="gap-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-xl">
                      {dataset.name || dataset.id || '--'}
                    </CardTitle>
                    <CardDescription className="max-w-2xl">
                      {dataset.description || t('no_description')}
                    </CardDescription>
                  </div>
                  <DatasetVersionStatusBadge
                    status={dataset.latest_version?.status}
                  />
                </div>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border p-4">
                    <div className="text-muted-foreground text-xs uppercase tracking-[0.2em]">
                      {t('source_type')}
                    </div>
                    <div className="mt-2 font-medium">
                      {dataset.source_type || '--'}
                    </div>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-muted-foreground text-xs uppercase tracking-[0.2em]">
                      {t('latest_version')}
                    </div>
                    <div className="mt-2 font-medium">
                      {dataset.latest_version?.version_name ||
                        dataset.latest_version?.version ||
                        '--'}
                    </div>
                  </div>
                  <div className="rounded-xl border p-4">
                    <div className="text-muted-foreground text-xs uppercase tracking-[0.2em]">
                      {t('cases')}
                    </div>
                    <div className="mt-2 font-medium">{getCaseCount(dataset)}</div>
                  </div>
                </div>

                <div className="text-muted-foreground flex flex-wrap items-center gap-6 text-sm">
                  <div>
                    {t('version_count')}: {dataset.version_count ?? '--'}
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
    </div>
  );
};
