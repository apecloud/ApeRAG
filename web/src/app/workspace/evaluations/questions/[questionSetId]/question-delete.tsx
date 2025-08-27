'use client';

import { Question, QuestionSet } from '@/api';
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

export const QuestionDelete = ({
  questionSet,
  question,
  children,
}: {
  questionSet: QuestionSet;
  question: Question;
  children: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();

  const handleDelete = useCallback(async () => {
    if (question?.id && questionSet.id) {
      await apiClient.evaluationApi.deleteQuestionApiV1QuestionSetsQsIdQuestionsQIdDelete(
        {
          qsId: questionSet.id,
          qId: question.id,
        },
      );
      setVisible(false);
      router.refresh();
    }
  }, [question.id, questionSet.id, router]);

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
            This action cannot be undone. This will permanently delete question
            and remove your question from our servers.
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
