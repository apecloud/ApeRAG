'use client';

import { useState } from 'react';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import {
  appendEvaluationDatasetItems,
  generateEvaluationDatasetItemsPreview,
  type EvaluationDatasetItemDraft,
} from '@/features/evaluation/client-api';
import { ChevronDown, ChevronRight, Loader2, Sparkles } from 'lucide-react';
import { useLocale, useTranslations } from 'next-intl';
import { toast } from 'sonner';

const SUPPORTED_LANGUAGES = ['zh-CN', 'en-US', 'ja-JP', 'ko-KR'] as const;
const DEFAULT_COUNT = 10;
const MAX_COUNT = 100;
const DEFAULT_PROMPT_PLACEHOLDER =
  'Optional: override the built-in QA-pair generation prompt. Leave empty for the default.';

type Phase = 'form' | 'loading' | 'preview';

type DraftRow = EvaluationDatasetItemDraft & {
  selected: boolean;
};

export const EvaluationAIGenerateDialog = ({
  open,
  onOpenChange,
  datasetId,
  collectionId,
  collectionLanguage,
  existingItemCount,
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  collectionId: string;
  collectionLanguage?: string | null;
  existingItemCount: number;
  onSaved: () => void;
}) => {
  const t = useTranslations('page_collection_evaluations');
  const uiLocale = useLocale();
  const defaultLanguage = (() => {
    const candidate = collectionLanguage || uiLocale;
    return SUPPORTED_LANGUAGES.includes(
      candidate as (typeof SUPPORTED_LANGUAGES)[number],
    )
      ? candidate
      : 'zh-CN';
  })();

  const [phase, setPhase] = useState<Phase>('form');
  const [count, setCount] = useState(DEFAULT_COUNT);
  const [language, setLanguage] = useState<string>(defaultLanguage);
  const [promptTemplate, setPromptTemplate] = useState('');
  const [drafts, setDrafts] = useState<DraftRow[]>([]);
  const [expandedIndices, setExpandedIndices] = useState<Set<number>>(
    () => new Set(),
  );
  const [saving, setSaving] = useState(false);

  const resetAll = () => {
    setPhase('form');
    setCount(DEFAULT_COUNT);
    setLanguage(defaultLanguage);
    setPromptTemplate('');
    setDrafts([]);
    setExpandedIndices(new Set());
    setSaving(false);
  };

  const handleClose = (next: boolean) => {
    onOpenChange(next);
    if (!next) {
      // Defer reset so the close animation doesn't flash an empty form.
      setTimeout(resetAll, 200);
    }
  };

  const handleGenerate = async () => {
    if (!collectionId) {
      toast.error(t('ai_generate_collection_missing'));
      return;
    }
    setPhase('loading');
    try {
      const response = await generateEvaluationDatasetItemsPreview(datasetId, {
        collection_id: collectionId,
        count: Math.max(1, Math.min(MAX_COUNT, Math.floor(count))),
        language,
        prompt_template: promptTemplate.trim() || undefined,
      });
      const rows: DraftRow[] = (response.items ?? []).map((item) => ({
        question: item.question ?? '',
        expected_answer: item.expected_answer ?? '',
        reference_context: item.reference_context ?? '',
        selected: true,
      }));
      if (rows.length === 0) {
        toast.error(t('ai_generate_empty_response'));
        setPhase('form');
        return;
      }
      setDrafts(rows);
      setPhase('preview');
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t('ai_generate_failed'),
      );
      setPhase('form');
    }
  };

  const toggleExpand = (index: number) => {
    setExpandedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const updateDraft = (index: number, patch: Partial<DraftRow>) => {
    setDrafts((prev) =>
      prev.map((row, i) => (i === index ? { ...row, ...patch } : row)),
    );
  };

  const selectedCount = drafts.filter((row) => row.selected).length;

  const handleSave = async () => {
    const selected = drafts.filter(
      (row) => row.selected && row.question.trim(),
    );
    if (selected.length === 0) {
      toast.error(t('ai_generate_no_selection'));
      return;
    }
    setSaving(true);
    try {
      await appendEvaluationDatasetItems(
        datasetId,
        selected.map((row, i) => ({
          input_message: row.question.trim(),
          expected_answer: row.expected_answer.trim() || undefined,
          reference_context: row.reference_context.trim() || undefined,
          sort_key: existingItemCount + i,
        })),
      );
      toast.success(
        t('ai_generate_save_success', { count: String(selected.length) }),
      );
      handleClose(false);
      onSaved();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : t('ai_generate_save_failed'),
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('ai_generate_title')}</DialogTitle>
          <DialogDescription>{t('ai_generate_description')}</DialogDescription>
        </DialogHeader>

        {phase === 'form' && (
          <div className="grid gap-4">
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('ai_generate_count_label')}
              </label>
              <Input
                type="number"
                min={1}
                max={MAX_COUNT}
                value={count}
                onChange={(event) => {
                  const next = Number(event.currentTarget.value);
                  if (Number.isFinite(next)) setCount(next);
                }}
              />
              <p className="text-muted-foreground text-xs">
                {t('ai_generate_count_helper', { max: String(MAX_COUNT) })}
              </p>
            </div>
            <div className="grid gap-2">
              <label className="text-sm font-medium text-slate-900">
                {t('ai_generate_language_label')}
              </label>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SUPPORTED_LANGUAGES.map((lang) => (
                    <SelectItem key={lang} value={lang}>
                      {t(`ai_generate_language_${lang}`)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Collapsible className="grid gap-2">
              <CollapsibleTrigger className="text-muted-foreground hover:text-foreground group flex items-center gap-1.5 text-xs">
                <ChevronRight className="size-3.5 group-data-[state=open]:hidden" />
                <ChevronDown className="hidden size-3.5 group-data-[state=open]:inline" />
                {t('ai_generate_advanced')}
              </CollapsibleTrigger>
              <CollapsibleContent className="grid gap-3 pt-1">
                <div className="grid gap-2">
                  <label className="text-sm font-medium text-slate-900">
                    {t('ai_generate_prompt_label')}
                  </label>
                  <Textarea
                    rows={4}
                    value={promptTemplate}
                    placeholder={DEFAULT_PROMPT_PLACEHOLDER}
                    onChange={(event) =>
                      setPromptTemplate(event.currentTarget.value)
                    }
                  />
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        )}

        {phase === 'loading' && (
          <div className="flex flex-col items-center justify-center gap-3 py-12">
            <Loader2 className="text-muted-foreground size-6 animate-spin" />
            <p className="text-muted-foreground text-sm">
              {t('ai_generate_loading')}
            </p>
          </div>
        )}

        {phase === 'preview' && (
          <div className="grid gap-3">
            <p className="text-muted-foreground text-xs">
              {t('ai_generate_preview_helper', {
                count: String(drafts.length),
                selected: String(selectedCount),
              })}
            </p>
            <div className="max-h-[60vh] overflow-y-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 sticky top-0 z-10">
                  <tr>
                    <th className="w-10 px-3 py-2 text-left">
                      <Checkbox
                        checked={
                          drafts.length > 0 &&
                          drafts.every((row) => row.selected)
                        }
                        onCheckedChange={(value) =>
                          setDrafts((prev) =>
                            prev.map((row) => ({
                              ...row,
                              selected: value === true,
                            })),
                          )
                        }
                      />
                    </th>
                    <th className="w-10 px-2 py-2 text-left" />
                    <th className="px-3 py-2 text-left font-medium">
                      {t('input_message_label')}
                    </th>
                    <th className="px-3 py-2 text-left font-medium">
                      {t('expected_answer_label')}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {drafts.map((row, index) => {
                    const expanded = expandedIndices.has(index);
                    return (
                      <>
                        <tr
                          key={`row-${index}`}
                          className="hover:bg-muted/30 border-t align-top"
                        >
                          <td className="px-3 py-2">
                            <Checkbox
                              checked={row.selected}
                              onCheckedChange={(value) =>
                                updateDraft(index, { selected: value === true })
                              }
                            />
                          </td>
                          <td className="px-2 py-2">
                            <button
                              type="button"
                              className="text-muted-foreground hover:text-foreground"
                              onClick={() => toggleExpand(index)}
                              title={t('ai_generate_show_source')}
                            >
                              {expanded ? (
                                <ChevronDown className="size-3.5" />
                              ) : (
                                <ChevronRight className="size-3.5" />
                              )}
                            </button>
                          </td>
                          <td className="px-3 py-2">
                            <Textarea
                              rows={2}
                              value={row.question}
                              onChange={(event) =>
                                updateDraft(index, {
                                  question: event.currentTarget.value,
                                })
                              }
                            />
                          </td>
                          <td className="px-3 py-2">
                            <Textarea
                              rows={2}
                              value={row.expected_answer}
                              onChange={(event) =>
                                updateDraft(index, {
                                  expected_answer: event.currentTarget.value,
                                })
                              }
                            />
                          </td>
                        </tr>
                        {expanded && (
                          <tr
                            key={`source-${index}`}
                            className="bg-muted/20 border-t"
                          >
                            <td colSpan={4} className="px-6 py-3">
                              <p className="text-muted-foreground mb-1 text-xs font-medium uppercase tracking-wider">
                                {t('ai_generate_source_chunk')}
                              </p>
                              <p className="text-foreground/80 max-h-48 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed">
                                {row.reference_context || (
                                  <span className="italic">
                                    {t('ai_generate_source_empty')}
                                  </span>
                                )}
                              </p>
                            </td>
                          </tr>
                        )}
                      </>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <DialogFooter>
          {phase === 'form' && (
            <>
              <Button
                variant="outline"
                onClick={() => handleClose(false)}
                disabled={false}
              >
                {t('cancel')}
              </Button>
              <Button onClick={handleGenerate}>
                <Sparkles className="size-4" /> {t('ai_generate_submit')}
              </Button>
            </>
          )}
          {phase === 'loading' && (
            <Button
              variant="outline"
              onClick={() => setPhase('form')}
              disabled
            >
              {t('cancel')}
            </Button>
          )}
          {phase === 'preview' && (
            <>
              <Button
                variant="outline"
                onClick={() => {
                  setPhase('form');
                  setDrafts([]);
                  setExpandedIndices(new Set());
                }}
                disabled={saving}
              >
                {t('ai_generate_back')}
              </Button>
              <Button onClick={handleSave} disabled={saving}>
                {saving && <Loader2 className="size-4 animate-spin" />}
                {t('ai_generate_save', { count: String(selectedCount) })}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
