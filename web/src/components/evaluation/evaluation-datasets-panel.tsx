'use client';

import { useMemo, useState, useTransition } from 'react';

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
import { Database, FolderPlus, Trash2 } from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import {
  createEvaluationDataset,
  deleteEvaluationDataset,
} from '@/features/evaluation/client-api';
import type { EvaluationDataset } from '@/features/evaluation/types';
import { EvaluationApiNotice } from './api-notice';

const matchesSearch = (dataset: EvaluationDataset, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [dataset.name, dataset.description, dataset.source_type].some(
    (value) => String(value ?? '').toLowerCase().includes(query),
  );
};

const sourceTypeLabelKey = (
  source_type: EvaluationDataset['source_type'],
): 'source_type.manual' | 'source_type.import' | 'source_type.generated' => {
  switch (source_type) {
    case 'import':
      return 'source_type.import';
    case 'generated':
      return 'source_type.generated';
    case 'manual':
    default:
      return 'source_type.manual';
  }
};

type DatasetFormState = {
  name: string;
  description: string;
};

const defaultDatasetForm: DatasetFormState = {
  name: '',
  description: '',
};

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
        source_type: 'manual',
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

  const handleDeleteDataset = async (dataset: EvaluationDataset) => {
    if (!dataset.id) return;
    if (!window.confirm(t('delete_dataset_confirm'))) return;

    try {
      await deleteEvaluationDataset(dataset.id);
      toast.success(t('delete_dataset_success'));
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('delete_dataset_failed'),
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
            <div className="flex items-center gap-2 text-xs tracking-[0.18em] text-slate-500 uppercase">
              <Database className="size-4" />
              {t('datasets_badge')}
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
              <Card
                key={dataset.id}
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
                    <Badge
                      variant="outline"
                      className="rounded-full border-slate-200 bg-white px-3 py-1 text-xs"
                    >
                      {t(sourceTypeLabelKey(dataset.source_type))}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="grid gap-4">
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4">
                      <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                        {t('dataset_item_count')}
                      </div>
                      <div className="mt-2 font-medium text-slate-900">
                        {dataset.item_count ?? 0}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4">
                      <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                        {t('dataset_source_type')}
                      </div>
                      <div className="mt-2 font-medium text-slate-900">
                        {t(sourceTypeLabelKey(dataset.source_type))}
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200/80 bg-slate-50/70 p-4">
                      <div className="text-xs tracking-[0.16em] text-slate-500 uppercase">
                        {t('dataset_created_at')}
                      </div>
                      <div className="mt-2 font-medium text-slate-900">
                        {dataset.created_at ? (
                          <FormatDate datetime={new Date(dataset.created_at)} />
                        ) : (
                          '--'
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap justify-end gap-2">
                    <Button
                      asChild
                      variant="outline"
                      className="rounded-full"
                      disabled={isPending}
                    >
                      <Link
                        href={`/workspace/collections/${collectionId}/evaluations/datasets/${dataset.id}`}
                      >
                        {t('manage_dataset_items')}
                      </Link>
                    </Button>
                    <Button
                      variant="outline"
                      className="rounded-full text-rose-700 hover:text-rose-800"
                      disabled={isPending}
                      onClick={() => handleDeleteDataset(dataset)}
                    >
                      <Trash2 className="size-4" />
                      {t('delete_dataset')}
                    </Button>
                  </div>
                </CardContent>
              </Card>
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
