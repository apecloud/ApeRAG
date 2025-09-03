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
import { useTranslations } from 'next-intl';
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
  const page_question_set = useTranslations('page_question_set');
  const common_action = useTranslations('common.action');
  const common_tips = useTranslations('common.tips');
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
          <AlertDialogTitle>{common_tips('confirm')}</AlertDialogTitle>
          <AlertDialogDescription>
            {page_question_set('delete_question_confirm')}
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
