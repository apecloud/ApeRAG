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
          <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
          <AlertDialogDescription>
            This action cannot be undone. This will permanently delete
            collection and remove your evaluation from our servers.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogDescription></AlertDialogDescription>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={() => setVisible(false)}>
            Cancel
          </AlertDialogCancel>
          <Button variant="destructive" onClick={() => handleDelete()}>
            Continue
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
