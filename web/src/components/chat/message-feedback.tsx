import {
  FEEDBACK_TAGS,
  type Feedback,
  type FeedbackType,
} from '@/features/bot/types';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { zodResolver } from '@hookform/resolvers/zod';
import { ThumbsDown, ThumbsUp } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useCallback, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

const feedbackSchema = z.object({
  tag: z.enum(FEEDBACK_TAGS, {
    error: 'You need to select a feedback type.',
  }),
  message: z.string().optional(),
});

export const MessageFeedback = ({
  turnId,
  feedback,
  onFeedback,
}: {
  turnId?: string;
  feedback?: Feedback;
  onFeedback: (turnId: string, feedback: Feedback) => void;
}) => {
  const [visible, setVisible] = useState<boolean>(false);
  const form = useForm<z.infer<typeof feedbackSchema>>({
    resolver: zodResolver(feedbackSchema),
  });
  const page_chat = useTranslations('page_chat');
  const common_action = useTranslations('common.action');

  const handleVote = useCallback(
    (type: FeedbackType) => {
      if (!turnId) return;
      if (type === feedback?.type) {
        onFeedback(turnId, {});
      } else if (type === 'good') {
        onFeedback(turnId, { type });
      } else {
        setVisible(true);
      }
    },
    [feedback?.type, onFeedback, turnId],
  );

  const handleBadVote = useCallback(
    (values: z.infer<typeof feedbackSchema>) => {
      if (!turnId) return;
      onFeedback(turnId, {
        type: 'bad',
        tag: values.tag,
        message: values.message,
      });
      setVisible(false);
    },
    [onFeedback, turnId],
  );

  return (
    <div className="flex flex-row items-center gap-1">
      <Dialog open={visible} onOpenChange={() => setVisible(false)}>
        <DialogContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(handleBadVote)}>
              <DialogHeader>
                <DialogTitle>{page_chat('feedback.title')}</DialogTitle>
                <DialogDescription></DialogDescription>
              </DialogHeader>
              <div className="space-y-8 py-8">
                <FormField
                  control={form.control}
                  name="tag"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>
                        {page_chat('feedback.choose_a_category')}
                      </FormLabel>
                      <FormControl>
                        <RadioGroup
                          onValueChange={field.onChange}
                          defaultValue={field.value}
                          className="flex flex-col gap-2"
                        >
                          {FEEDBACK_TAGS.map((key) => {
                            return (
                              <Label key={key} className="block">
                                <FormItem
                                  className={cn(
                                    'hover:bg-accent flex items-center gap-3 rounded-md border p-3',
                                    key === field.value ? 'bg-accent/80' : '',
                                  )}
                                >
                                  <FormControl>
                                    <RadioGroupItem value={key} />
                                  </FormControl>
                                  <div>
                                    <div className="mb-1">
                                      {page_chat(`feedback.${key}.title`)}
                                    </div>
                                    <div className="text-muted-foreground">
                                      {page_chat(`feedback.${key}.description`)}
                                    </div>
                                  </div>
                                </FormItem>
                              </Label>
                            );
                          })}
                        </RadioGroup>
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="message"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>
                        {page_chat('feedback.share_your_feedback')}
                      </FormLabel>
                      <FormControl>
                        <Textarea
                          {...field}
                          className="min-h-25"
                          placeholder={page_chat(
                            'feedback.share_your_feedback_placeholder',
                          )}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setVisible(false)}
                >
                  {common_action('cancel')}
                </Button>
                <Button type="submit">{common_action('save')}</Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>
      <Button
        variant="ghost"
        size="icon"
        disabled={!turnId}
        className={cn(
          'cursor-pointer',
          feedback?.type === 'good'
            ? 'text-green-500 hover:text-green-600'
            : 'text-muted-foreground',
        )}
        onClick={() => handleVote('good')}
      >
        <ThumbsUp />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        disabled={!turnId}
        className={cn(
          'cursor-pointer',
          feedback?.type === 'bad'
            ? 'text-rose-500 hover:text-rose-600'
            : 'text-muted-foreground',
        )}
        onClick={() => handleVote('bad')}
      >
        <ThumbsDown />
      </Button>
    </div>
  );
};
