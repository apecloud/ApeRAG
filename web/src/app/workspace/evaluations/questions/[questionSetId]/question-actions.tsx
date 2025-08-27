'use client';
import { Question, QuestionSet } from '@/api';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from '@/components/ui/form';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';

const questionSchema = z.object({
  question_text: z.string().min(1),
  ground_truth: z.string().min(1),
  // question_type: z.enum(['FACTUAL', 'INFERENTIAL', 'USER_DEFINED']),
});

export const QuestionActions = ({
  questionSet,
  question,
  action,
  children,
}: {
  questionSet: QuestionSet;
  question?: Question;
  action: 'add' | 'edit';
  children: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const form = useForm<z.infer<typeof questionSchema>>({
    resolver: zodResolver(questionSchema),
    defaultValues: {
      question_text: question?.question_text || '',
      ground_truth: question?.ground_truth || '',
    },
  });

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof questionSchema>) => {
      if (!questionSet?.id) return;
      if (action === 'add') {
        await apiClient.evaluationApi.addQuestionsApiV1QuestionSetsQsIdQuestionsPost(
          {
            qsId: questionSet.id,
            questionsAdd: {
              questions: [
                {
                  ...values,
                  // question_type: 'USER_DEFINED',
                },
              ],
            },
          },
        );
      }
      if (action === 'edit') {
        if (!question?.id || !questionSet?.id) return;
        await apiClient.evaluationApi.updateQuestionApiV1QuestionSetsQsIdQuestionsQIdPut(
          {
            qsId: questionSet.id,
            qId: question.id,
            questionUpdate: {
              ...values,
              // question_type: 'USER_DEFINED',
            },
          },
        );
      }
      router.refresh();
      setVisible(false);
    },
    [action, question?.id, questionSet?.id, router],
  );

  useEffect(() => {
    if (visible) {
      form.reset();
    }
  }, [form, visible]);

  return (
    <Dialog open={visible} onOpenChange={() => setVisible(false)}>
      <DialogTrigger asChild>
        <Slot
          onClick={(e) => {
            setVisible(true);
            e.preventDefault();
          }}
        >
          {children}
        </Slot>
      </DialogTrigger>
      <DialogContent>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleCreateOrUpdate)}
            className="space-y-6"
          >
            <DialogHeader>
              <DialogTitle>
                {action === 'add' ? 'Add Question' : 'Update Question'}
              </DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>

            <FormField
              control={form.control}
              name="question_text"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Question</FormLabel>
                  <FormControl>
                    <Textarea {...field} placeholder="Enter your question..." />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="ground_truth"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Ground Truth</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Enter the ground truth..."
                      {...field}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisible(false)}
              >
                Cancel
              </Button>
              <Button type="submit">
                {action === 'add' ? 'Create' : 'Update'}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};
