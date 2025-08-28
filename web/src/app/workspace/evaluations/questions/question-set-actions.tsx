'use client';
import { QuestionSet } from '@/api';
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
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { zodResolver } from '@hookform/resolvers/zod';
import { Slot } from '@radix-ui/react-slot';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';

const questionSetSchema = z.object({
  name: z.string().min(1),
  description: z.string(),
});

export const QuestionSetActions = ({
  questionSet,
  action,
  children,
}: {
  questionSet?: QuestionSet;
  action: 'add' | 'edit';
  children: React.ReactNode;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const router = useRouter();
  const form = useForm<z.infer<typeof questionSetSchema>>({
    resolver: zodResolver(questionSetSchema),
    defaultValues: {
      name: questionSet?.name || '',
      description: questionSet?.description || '',
    },
  });

  const handleCreateOrUpdate = useCallback(
    async (values: z.infer<typeof questionSetSchema>) => {
      if (action === 'add') {
        await apiClient.evaluationApi.createQuestionSetApiV1QuestionSetsPost({
          questionSetCreate: {
            ...values,
            questions: [],
          },
        });
      }
      if (action === 'edit') {
        if (!questionSet?.id) return;
        await apiClient.evaluationApi.updateQuestionSetApiV1QuestionSetsQsIdPut(
          {
            qsId: questionSet.id,
            questionSetUpdate: values,
          },
        );
      }
      router.refresh();
      setVisible(false);
    },
    [action, questionSet?.id, router],
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
                {action === 'add' ? 'Add Question Set' : 'Update Question Set'}
              </DialogTitle>
              <DialogDescription></DialogDescription>
            </DialogHeader>

            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Question set name</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      placeholder="Enter your question set name."
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Enter your question set description."
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
