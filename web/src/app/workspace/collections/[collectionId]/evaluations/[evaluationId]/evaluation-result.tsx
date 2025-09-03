'use client';

import { EvaluationDetail, EvaluationItem } from '@/api';
import { Markdown } from '@/components/markdown';
import { useCollectionContext } from '@/components/providers/collection-provider';
import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Separator } from '@/components/ui/separator';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import _ from 'lodash';
import {
  ArrowLeft,
  ChevronRight,
  EllipsisVertical,
  ListRestart,
  LoaderCircle,
  RotateCcw,
  Trash,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { EvaluationDeleteItem } from './evaluation-delete';
import { EvaluationRetryItem } from './evaluation-retry-item';

const EvaluationResultStatus = ({ item }: { item: EvaluationItem }) => {
  if (item.status === 'COMPLETED') {
    return (
      <div
        data-score={item.llm_judge_score}
        className={cn(
          'ml-auto flex size-8 flex-col justify-center rounded-full bg-gray-500/30 text-center text-white',
          'data-[score=5]:bg-green-700',
          'data-[score=4]:bg-cyan-700',
          'data-[score=3]:bg-amber-700',
          'data-[score=2]:bg-fuchsia-700',
          'data-[score=1]:bg-rose-700',
        )}
      >
        {item.llm_judge_score}
      </div>
    );
  } else if (item.status === 'RUNNING') {
    return (
      <div className="ml-auto flex size-8 flex-col justify-center rounded-full bg-gray-500/30 text-center text-white">
        <LoaderCircle className="size-8 animate-spin opacity-50" />
      </div>
    );
  } else {
    return (
      <div className="text-muted-foreground ml-auto flex flex-row items-center gap-2">
        <div
          data-status={item.status}
          className={cn(
            'size-2 rounded-lg bg-gray-500',
            'data-[status=COMPLETED]:bg-green-700',
            'data-[status=FAILED]:bg-red-500',
            'data-[status=PENDING]:bg-gray-500',
            'data-[status=RUNNING]:bg-sky-500',
          )}
        />
        {_.upperFirst(_.lowerCase(item.status))}
      </div>
    );
  }
};

export const EvaluationResult = ({
  evaluation: initData,
}: {
  evaluation: EvaluationDetail;
}) => {
  const { collection } = useCollectionContext();
  const [evaluation, setEvaluation] = useState<EvaluationDetail>(initData);
  const page_evaluation = useTranslations('page_evaluation');
  const loadData = useCallback(async () => {
    if (!evaluation.id) return;
    const res =
      await apiClient.evaluationApi.getEvaluationApiV1EvaluationsEvalIdGet({
        evalId: evaluation.id,
      });
    if (res.data.id) {
      setEvaluation(res.data);
    }
  }, [evaluation.id]);

  useEffect(() => {
    if (
      evaluation.items?.some((item) =>
        ['RUNNING', 'PENDING'].includes(item.status || ''),
      )
    ) {
      setTimeout(loadData, 5000);
    }
  }, [evaluation, loadData]);

  return (
    <>
      <div className="mb-4 flex flex-row items-center justify-between gap-4">
        <div className="flex flex-1 flex-row items-center gap-2 truncate text-lg font-bold">
          <Button size="icon" variant="secondary">
            <Link href={`/workspace/collections/${collection.id}/evaluations`}>
              <ArrowLeft />
            </Link>
          </Button>
          {evaluation.name}
        </div>
        <div className="flex flex-row items-center gap-4 text-sm">
          <div className="flex flex-row items-center gap-2">
            <div
              data-status={evaluation.status}
              className={cn(
                'size-2 rounded-lg',
                'data-[status=COMPLETED]:bg-green-700',
                'data-[status=FAILED]:bg-red-500',
                'data-[status=PENDING]:bg-gray-500',
                'data-[status=PAUSED]:bg-amber-500',
                'data-[status=RUNNING]:bg-sky-500',
              )}
            />
            {_.upperFirst(_.lowerCase(evaluation.status))}
          </div>

          <div className="flex flex-row gap-1">
            <span className="text-muted-foreground">
              {page_evaluation('question_set')}:{' '}
            </span>
            <span>
              <Link
                href={`/workspace/collections/${collection.id}/questions/${evaluation.config?.question_set_id}`}
                className="hover:text-primary underline"
              >
                {_.truncate(evaluation.question_set_name, { length: 20 })}
              </Link>
            </span>
          </div>

          <div className="flex flex-row items-center">
            <span className="text-muted-foreground text-sm">
              {page_evaluation('avg_score')}: &nbsp;
            </span>
            <span className="text-2xl font-bold">
              {evaluation.average_score || 0}
            </span>
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="icon" variant="ghost">
                <EllipsisVertical />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-60">
              <EvaluationRetryItem
                onRetry={loadData}
                scope={'failed'}
                evaluation={evaluation}
              >
                <ListRestart />

                {page_evaluation('retry_failed_evaluation')}
              </EvaluationRetryItem>
              <EvaluationRetryItem
                onRetry={loadData}
                scope="all"
                evaluation={evaluation}
              >
                <RotateCcw />
                {page_evaluation('retry_all_evaluation')}
              </EvaluationRetryItem>

              <DropdownMenuSeparator />
              <EvaluationDeleteItem evaluation={evaluation}>
                <DropdownMenuItem variant="destructive">
                  <Trash /> {page_evaluation('delete_evaluation')}
                </DropdownMenuItem>
              </EvaluationDeleteItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        {evaluation.items?.map((item, index) => {
          return (
            <Collapsible
              key={item.id}
              className="group/collapsible flex flex-col gap-2"
            >
              <CollapsibleTrigger asChild>
                <Button
                  size="lg"
                  variant="secondary"
                  className="h-14 w-full cursor-pointer justify-start"
                >
                  <ChevronRight className="transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                  <span className="flex-1 truncate text-left">
                    {index + 1}. {item.question_text}
                  </span>
                  <EvaluationResultStatus item={item} />
                </Button>
              </CollapsibleTrigger>

              <CollapsibleContent className="flex flex-col gap-6 rounded-lg border p-6 text-sm">
                <div>
                  <div className="text-muted-foreground mb-4">
                    {page_evaluation('judge_reason')}
                  </div>
                  <div
                    data-score={item.llm_judge_score}
                    className={cn(
                      'data-[score=5]:text-green-700',
                      'data-[score=4]:text-cyan-700',
                      'data-[score=3]:text-amber-700',
                      'data-[score=2]:text-fuchsia-700',
                      'data-[score=1]:text-rose-700',
                    )}
                  >
                    <Markdown>{item.llm_judge_reasoning}</Markdown>
                  </div>
                </div>
                <Separator />
                <div>
                  <div className="text-muted-foreground mb-4">
                    {page_evaluation('ground_truth')}
                  </div>
                  <div>{item.ground_truth}</div>
                </div>
                <Separator />
                <div>
                  <div className="text-muted-foreground">
                    {page_evaluation('rag_answer')}
                  </div>
                  <Markdown>{item.rag_answer}</Markdown>
                </div>
              </CollapsibleContent>
            </Collapsible>
          );
        })}
      </div>
    </>
  );
};
