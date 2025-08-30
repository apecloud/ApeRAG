'use client';

import { Evaluation } from '@/api';
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
import { apiClient } from '@/lib/api/client';
import { Slot } from '@radix-ui/react-slot';
import { useTranslations } from 'next-intl';
import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';

export const EvaluationDeleteItem = ({
  evaluation,
  children,
}: {
  evaluation: Evaluation;
  children?: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const page_evaluation = useTranslations('page_evaluation');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
  const handleDelete = useCallback(async () => {
    if (evaluation?.id) {
      await apiClient.evaluationApi.deleteEvaluationApiV1EvaluationsEvalIdDelete(
        {
          evalId: evaluation.id,
        },
      );
      setVisible(false);
      router.push('/workspace/evaluations');
    }
  }, [evaluation.id, router]);

  return (
    <AlertDialog open={visible} onOpenChange={() => setVisible(false)}>
      <AlertDialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
          <AlertDialogDescription>
            {page_evaluation('delete_evaluation_confirm')}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogDescription></AlertDialogDescription>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            {common_action('cancel')}
          </AlertDialogCancel>
          <Button variant="destructive" onClick={() => handleDelete()}>
            {common_action('continue')}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
