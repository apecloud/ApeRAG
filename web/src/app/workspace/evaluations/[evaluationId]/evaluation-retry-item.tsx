'use client';
import {
  Evaluation,
  RetryEvaluationApiV1EvaluationsEvalIdRetryPostScopeEnum,
} from '@/api';
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { DropdownMenuItem } from '@/components/ui/dropdown-menu';
import { apiClient } from '@/lib/api/client';
import { useTranslations } from 'next-intl';

import { useCallback, useState } from 'react';

export const EvaluationRetryItem = ({
  scope,
  evaluation,
  children,
  onRetry,
}: {
  scope: RetryEvaluationApiV1EvaluationsEvalIdRetryPostScopeEnum;
  evaluation: Evaluation;
  children: React.ReactNode;
  onRetry: () => void;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const page_evaluation = useTranslations('page_evaluation');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const handleRetry = useCallback(async () => {
    if (!evaluation.id) return;

    await apiClient.evaluationApi.retryEvaluationApiV1EvaluationsEvalIdRetryPost(
      {
        evalId: evaluation.id,
        scope,
      },
    );
    setVisible(false);

    onRetry();
  }, [evaluation.id, onRetry, scope]);

  return (
    <AlertDialog open={visible} onOpenChange={() => setVisible(false)}>
      <AlertDialogTrigger asChild>
        <DropdownMenuItem
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </DropdownMenuItem>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
          <AlertDialogDescription>
            {scope === 'all'
              ? page_evaluation('retry_all_evaluation_confirm')
              : page_evaluation('retry_failed_evaluation_confirm')}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogDescription></AlertDialogDescription>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            {common_action('cancel')}
          </AlertDialogCancel>
          <Button onClick={() => handleRetry()}>
            {common_action('continue')}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
