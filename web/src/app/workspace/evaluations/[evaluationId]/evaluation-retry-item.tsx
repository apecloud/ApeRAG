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
          <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
          <AlertDialogDescription>
            Are you sure you want to re-evaluate these{' '}
            {scope === 'all' ? 'all' : 'failed'} questions?
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogDescription></AlertDialogDescription>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            Cancel
          </AlertDialogCancel>
          <Button onClick={() => handleRetry()}>Continue</Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
