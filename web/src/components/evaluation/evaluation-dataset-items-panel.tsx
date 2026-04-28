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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Textarea } from '@/components/ui/textarea';
import { FilePlus2, Sparkles, Trash2 } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

import {
  appendEvaluationDatasetItems,
  deleteEvaluationDatasetItem,
} from '@/features/evaluation/client-api';
import type {
  EvaluationDataset,
  EvaluationDatasetItem,
} from '@/features/evaluation/types';
import { EvaluationAIGenerateDialog } from './evaluation-ai-generate-dialog';
import { EvaluationApiNotice } from './api-notice';

type ItemFormState = {
  inputMessage: string;
  expectedAnswer: string;
};

const defaultItemForm: ItemFormState = {
  inputMessage: '',
  expectedAnswer: '',
};

const matchesSearch = (item: EvaluationDatasetItem, searchValue: string) => {
  const query = searchValue.trim().toLowerCase();
  if (!query) return true;

  return [item.case_key, item.input_message, item.expected_answer].some(
    (value) => String(value ?? '').toLowerCase().includes(query),
  );
};

export const EvaluationDatasetItemsPanel = ({
  dataset,
  items,
  unavailable,
  error,
}: {
  dataset: EvaluationDataset;
  items: EvaluationDatasetItem[];
  unavailable: boolean;
  error?: string;
}) => {
  const t = useTranslations('page_collection_evaluations');
  const router = useRouter();
  const [searchValue, setSearchValue] = useState('');
  const [addItemOpen, setAddItemOpen] = useState(false);
  const [aiGenerateOpen, setAiGenerateOpen] = useState(false);
  const [itemForm, setItemForm] = useState<ItemFormState>(defaultItemForm);
  const [isPending, startTransition] = useTransition();

  const filteredItems = useMemo(() => {
    return items.filter((item) => matchesSearch(item, searchValue));
  }, [items, searchValue]);

  const refreshPage = () => {
    startTransition(() => {
      router.refresh();
    });
  };

  const handleAddItem = async () => {
    if (!itemForm.inputMessage.trim()) {
      toast.error(t('question_required'));
      return;
    }

    try {
      await appendEvaluationDatasetItems(dataset.id, [
        {
          input_message: itemForm.inputMessage.trim(),
          expected_answer: itemForm.expectedAnswer.trim() || undefined,
          // case_key + reference_context are intentionally omitted from
          // the form per earayu2 msg=64ff1a52 + Weston msg=f92ba261:
          // Manual-add and AI-auto-generate datasets share the same
          // shape (question + expected_answer); case_key is generated
          // server-side, reference_context is Phase 3 plumbing the BE
          // does not consume yet.
          sort_key: items.length,
        },
      ]);

      toast.success(t('question_saved'));
      setAddItemOpen(false);
      setItemForm(defaultItemForm);
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('question_save_failed'),
      );
    }
  };

  const handleDeleteItem = async (itemId: string) => {
    if (!window.confirm(t('delete_question_confirm'))) return;

    try {
      await deleteEvaluationDatasetItem(dataset.id, itemId);
      toast.success(t('question_deleted'));
      refreshPage();
    } catch (actionError) {
      toast.error(
        actionError instanceof Error
          ? actionError.message
          : t('question_delete_failed'),
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
      <Card>
        <CardHeader className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <CardTitle>{t('dataset_detail_title')}</CardTitle>
            <CardDescription>{t('dataset_detail_description')}</CardDescription>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              className="w-full sm:w-72"
              placeholder={t('search_datasets_placeholder')}
              value={searchValue}
              onChange={(event) => setSearchValue(event.currentTarget.value)}
            />
            <Button
              variant="outline"
              onClick={() => setAiGenerateOpen(true)}
              disabled={isPending || !dataset.collection_id}
              title={
                dataset.collection_id ? undefined : t('ai_generate_no_collection')
              }
            >
              <Sparkles className="size-4" />
              {t('ai_generate_button')}
            </Button>
            <Button onClick={() => setAddItemOpen(true)} disabled={isPending}>
              <FilePlus2 className="size-4" />
              {t('add_question')}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {filteredItems.length === 0 ? (
            <div className="grid gap-3 rounded-[1.5rem] border border-dashed border-slate-300 bg-slate-50/70 p-6">
              <h3 className="text-xl font-semibold tracking-[-0.03em] text-slate-950">
                {t('empty_items_title')}
              </h3>
              <p className="max-w-[62ch] text-sm leading-7 text-slate-600 sm:text-base">
                {t('empty_items_description')}
              </p>
              <div>
                <Button onClick={() => setAddItemOpen(true)}>
                  <FilePlus2 className="size-4" />
                  {t('add_question')}
                </Button>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t('case_key_label')}</TableHead>
                  <TableHead>{t('input_message_label')}</TableHead>
                  <TableHead>{t('expected_answer_label')}</TableHead>
                  <TableHead>{t('created_at')}</TableHead>
                  <TableHead className="text-right">{t('actions')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredItems.map((item) => (
                  <TableRow key={item.id}>
                    <TableCell className="font-medium">
                      {item.case_key || '--'}
                    </TableCell>
                    <TableCell className="max-w-96 whitespace-normal">
                      {item.input_message || '--'}
                    </TableCell>
                    <TableCell className="max-w-60 whitespace-normal">
                      {item.expected_answer || '--'}
                    </TableCell>
                    <TableCell>
                      {item.created_at ? (
                        <FormatDate datetime={new Date(item.created_at)} />
                      ) : (
                        '--'
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={isPending}
                        onClick={() => handleDeleteItem(item.id)}
                      >
                        <Trash2 className="size-4" />
                        {t('delete')}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Dialog open={addItemOpen} onOpenChange={setAddItemOpen}>
        <DialogContent className="sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>{t('add_question_title')}</DialogTitle>
            <DialogDescription>
              {t('add_question_description')}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('input_message_label')}
              </label>
              <Textarea
                rows={4}
                value={itemForm.inputMessage}
                placeholder={t('input_message_placeholder')}
                onChange={(event) =>
                  setItemForm((prev) => ({
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
                value={itemForm.expectedAnswer}
                placeholder={t('expected_answer_placeholder')}
                onChange={(event) =>
                  setItemForm((prev) => ({
                    ...prev,
                    expectedAnswer: event.currentTarget.value,
                  }))
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setAddItemOpen(false)}
              disabled={isPending}
            >
              {t('cancel')}
            </Button>
            <Button onClick={handleAddItem} disabled={isPending}>
              {t('save_question')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {dataset.collection_id && (
        <EvaluationAIGenerateDialog
          open={aiGenerateOpen}
          onOpenChange={setAiGenerateOpen}
          datasetId={dataset.id}
          collectionId={dataset.collection_id}
          collectionLanguage={null}
          existingItemCount={items.length}
          onSaved={refreshPage}
        />
      )}
    </>
  );
};
