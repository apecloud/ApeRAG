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
import { Textarea } from '@/components/ui/textarea';
import { Clock3, FolderPlus, ListChecks } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import { createEvaluationDataset } from '@/features/evaluation/client-api';
import type { EvaluationDataset } from '@/features/evaluation/types';
import { EvaluationApiNotice } from './api-notice';

const matchesSearch = (dataset: EvaluationDataset, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [dataset.name, dataset.description].some((value) =>
    String(value ?? '')
      .toLowerCase()
      .includes(query),
  );
};

type DatasetFormState = {
  name: string;
  description: string;
};

const defaultDatasetForm: DatasetFormState = {
  name: '',
  description: '',
};

const datasetMetaIconClass =
  'mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-500 shadow-sm';

export const EvaluationDatasetsPanel = ({
  collectionId,
  items,
  unavailable,
  error,
}: {
  collectionId: string;
  items: EvaluationDataset[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_collection_evaluations');
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [createDatasetOpen, setCreateDatasetOpen] = useState(false);
  const [datasetForm, setDatasetForm] =
    useState<DatasetFormState>(defaultDatasetForm);
  const [isPending, startTransition] = useTransition();

  const filteredItems = useMemo(() => {
    return items.filter((dataset) => matchesSearch(dataset, searchValue));
  }, [items, searchValue]);

  const refreshPage = () => {
    startTransition(() => {
      router.refresh();
    });
  };

  const handleCreateDataset = async () => {
    if (!datasetForm.name.trim()) {
      toast.error(t('create_dataset_name_required'));
      return;
    }

    try {
      const payload = await createEvaluationDataset({
        name: datasetForm.name.trim(),
        description: datasetForm.description.trim() || undefined,
        collection_id: collectionId,
      });

      if (!payload?.id) {
        throw new Error(t('create_dataset_missing_id'));
      }

      toast.success(t('create_dataset_success'));
      setCreateDatasetOpen(false);
      setDatasetForm(defaultDatasetForm);
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('create_dataset_failed'),
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
      <section className="flex flex-col gap-4 rounded-[1.75rem] border border-slate-200/80 bg-white/90 p-6 shadow-[0_24px_70px_-45px_rgba(15,23,42,0.4)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-2">
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
              placeholder={t('search_datasets_placeholder')}
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
          <div className="grid gap-3 rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-50/70 p-6">
            <h3 className="text-xl font-semibold tracking-[-0.03em] text-slate-950">
              {t('empty_datasets_title')}
            </h3>
            <p className="max-w-[62ch] text-sm leading-7 text-slate-600 sm:text-base">
              {t('empty_datasets_description')}
            </p>
            <div>
              <Button
                className="rounded-full"
                onClick={() => setCreateDatasetOpen(true)}
              >
                <FolderPlus className="size-4" />
                {t('create_dataset')}
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {filteredItems.map((dataset) => (
              <Link
                key={dataset.id}
                className="block focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 focus-visible:outline-none"
                href={`/workspace/collections/${collectionId}/evaluations/datasets/${dataset.id}`}
              >
                <Card className="h-full overflow-hidden border-slate-200/80 bg-white shadow-[0_22px_60px_-40px_rgba(15,23,42,0.35)] transition-colors hover:bg-slate-50/80">
                  <CardHeader className="gap-4">
                    <div className="space-y-1">
                      <CardTitle className="text-2xl tracking-[-0.03em]">
                        {dataset.name || dataset.id || '--'}
                      </CardTitle>
                      {dataset.description ? (
                        <CardDescription className="max-w-2xl text-sm leading-7">
                          {dataset.description}
                        </CardDescription>
                      ) : null}
                    </div>
                  </CardHeader>
                  <CardContent className="grid gap-4">
                    <dl className="grid gap-4 border-y border-slate-100 py-4 sm:grid-cols-2 sm:gap-0">
                      <div className="flex gap-3 sm:border-r sm:border-slate-100 sm:pr-5">
                        <span
                          className={datasetMetaIconClass}
                          aria-hidden="true"
                        >
                          <ListChecks className="size-4" />
                        </span>
                        <div className="min-w-0">
                          <dt className="text-xs font-medium text-slate-500">
                            {t('dataset_item_count')}
                          </dt>
                          <dd className="mt-1 text-lg leading-6 font-semibold text-slate-950">
                            {dataset.item_count ?? 0}
                          </dd>
                        </div>
                      </div>
                      <div className="flex gap-3 sm:pl-5">
                        <span
                          className={datasetMetaIconClass}
                          aria-hidden="true"
                        >
                          <Clock3 className="size-4" />
                        </span>
                        <div className="min-w-0">
                          <dt className="text-xs font-medium text-slate-500">
                            {t('dataset_created_at')}
                          </dt>
                          <dd className="mt-1 truncate text-lg leading-6 font-semibold text-slate-950">
                            {dataset.created_at ? (
                              <FormatDate
                                datetime={new Date(dataset.created_at)}
                              />
                            ) : (
                              '--'
                            )}
                          </dd>
                        </div>
                      </div>
                    </dl>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        )}
      </section>

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
    </>
  );
};
